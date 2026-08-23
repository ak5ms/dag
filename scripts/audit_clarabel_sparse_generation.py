"""Reject large dense arrays during fresh direct-Clarabel MPO generation.

The default 150-asset, eight-horizon model previously made CVXPYgen 1.0 attempt
a roughly 34.9 GiB ``sparse.toarray()`` result. This audit wraps that call and
``numpy.zeros`` with a 512 MiB rejection threshold, then reports direct backend
generation time and process peak RSS.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import resource
import time
from unittest.mock import patch

import cvxpy as cp
import numpy as np
from scipy import sparse

from trading_dsl_engine.cpp_stream.optimizer import (
    ClarabelNativePaths,
    build_current_clarabel,
)
from trading_dsl_engine.cpp_stream.optimizer.direct_clarabel import (
    generate_clarabel_artifact,
)


N_ASSETS = int(os.environ.get("CLARABEL_AUDIT_ASSETS", "150"))
N_HORIZONS = int(os.environ.get("CLARABEL_AUDIT_HORIZONS", "8"))
DENSE_LIMIT_BYTES = int(
    os.environ.get(
        "CLARABEL_AUDIT_DENSE_LIMIT_BYTES",
        str(512 * 1024 * 1024),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        "CLARABEL_AUDIT_OUTPUT_DIR",
        f".generated/clarabel-sharded-audit-{N_ASSETS}x{N_HORIZONS}",
    )
)
PARAMETER_SHARD_SIZE = int(
    os.environ.get(
        "CLARABEL_AUDIT_PARAMETER_SHARD_SIZE",
        os.environ.get("CLARABEL_PARAMETER_SHARD_SIZE", "512"),
    )
)


def _clarabel() -> ClarabelNativePaths:
    include = os.environ.get("CLARABEL_INCLUDE_DIR")
    library = os.environ.get("CLARABEL_STATIC_LIBRARY")
    if include and library:
        return ClarabelNativePaths(Path(include), Path(library))
    if bool(include) != bool(library):
        raise ValueError(
            "set both CLARABEL_INCLUDE_DIR and CLARABEL_STATIC_LIBRARY"
        )
    return build_current_clarabel()


def _problem() -> cp.Problem:
    weights = cp.Variable((N_HORIZONS, N_ASSETS), name="weights")
    turnover = cp.Variable((N_HORIZONS, N_ASSETS), name="turnover")
    expected_returns = cp.Parameter(
        (N_HORIZONS, N_ASSETS), name="expected_returns"
    )
    half_spread = cp.Parameter(
        (N_HORIZONS, N_ASSETS), nonneg=True, name="half_spread"
    )
    current_weights = cp.Parameter(N_ASSETS, name="current_weights")
    risk_radius = cp.Parameter(N_HORIZONS, nonneg=True, name="risk_radius")
    risk_factor = cp.Parameter((N_ASSETS, N_ASSETS), name="risk_factor")
    delta = weights - cp.vstack([current_weights, weights[:-1]])
    constraints = [turnover >= delta, turnover >= -delta]
    constraints.extend(
        cp.SOC(risk_radius[horizon], risk_factor @ weights[horizon])
        for horizon in range(N_HORIZONS)
    )
    return cp.Problem(
        cp.Minimize(
            -cp.sum(cp.multiply(expected_returns, weights))
            + cp.sum(cp.multiply(half_spread, turnover))
        ),
        constraints,
    )


def main() -> None:
    rejected: list[dict[str, object]] = []
    original_zeros = np.zeros

    def guarded_zeros(shape, *args, **kwargs):
        dtype = kwargs.get("dtype", args[0] if args else float)
        count = int(np.prod(shape, dtype=np.int64)) if shape != () else 1
        size = count * np.dtype(dtype).itemsize
        if size >= DENSE_LIMIT_BYTES:
            rejected.append(
                {"operation": "numpy.zeros", "shape": shape, "bytes": size}
            )
            raise MemoryError(
                f"rejected numpy.zeros allocation of {size} bytes"
            )
        return original_zeros(shape, *args, **kwargs)

    def guarded_toarray(method, class_name):
        def call(matrix, *args, **kwargs):
            size = (
                int(np.prod(matrix.shape, dtype=np.int64))
                * matrix.dtype.itemsize
            )
            if size >= DENSE_LIMIT_BYTES:
                rejected.append(
                    {
                        "operation": f"{class_name}.toarray",
                        "shape": matrix.shape,
                        "bytes": size,
                    }
                )
                raise MemoryError(
                    f"rejected {class_name}.toarray allocation of {size} bytes"
                )
            return method(matrix, *args, **kwargs)

        return call

    sparse_classes = tuple(
        cls
        for cls in (
            sparse.csr_matrix,
            sparse.csc_matrix,
            sparse.coo_matrix,
            sparse.lil_matrix,
            getattr(sparse, "csr_array", None),
            getattr(sparse, "csc_array", None),
            getattr(sparse, "coo_array", None),
        )
        if cls is not None
    )

    baseline_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    started = time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(patch.object(np, "zeros", guarded_zeros))
        for cls in sparse_classes:
            if "toarray" not in cls.__dict__:
                continue
            original = cls.__dict__["toarray"]
            stack.enter_context(
                patch.object(
                    cls,
                    "toarray",
                    guarded_toarray(original, cls.__name__),
                )
            )
        artifact = generate_clarabel_artifact(
            _problem(),
            code_dir=OUTPUT_DIR,
            clarabel=_clarabel(),
            class_name=f"SparseAuditMpo{N_ASSETS}",
            prefix=f"sparse_audit_mpo_{N_ASSETS}_",
            instrument_count=N_ASSETS,
            parameter_shard_size=PARAMETER_SHARD_SIZE,
            force=True,
        )

    peak_rss_kib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {
        "assets": N_ASSETS,
        "horizons": N_HORIZONS,
        "dense_allocation_guard_bytes": DENSE_LIMIT_BYTES,
        "parameter_shard_size": PARAMETER_SHARD_SIZE,
        "rejected_dense_allocations": rejected,
        "elapsed_seconds": time.perf_counter() - started,
        "baseline_rss_kib": baseline_rss_kib,
        "peak_rss_kib": peak_rss_kib,
        "generation_rss_growth_kib": peak_rss_kib - baseline_rss_kib,
        "generated_header_bytes": artifact.instance_header.stat().st_size,
        "manifest": str(artifact.manifest_path),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if rejected:
        raise RuntimeError("optimizer generation attempted an oversized allocation")


if __name__ == "__main__":
    main()
