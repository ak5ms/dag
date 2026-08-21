from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import numpy as np

from trading_dsl_engine.cpp_stream.optimizer.node import CvxpyNodeDefinition


@dataclass(frozen=True)
class CvxpygenArtifact:
    """CVXPYgen sources and cpp_stream's stable parameter/output ABI manifest."""

    code_dir: Path
    manifest_path: Path
    solver: str


def generate_cvxpygen_artifact(
    definition: CvxpyNodeDefinition,
    sample_parameters: Mapping[str, Any],
    code_dir: str | Path,
    *,
    solver: str = "CLARABEL",
    wrapper: bool = False,
) -> CvxpygenArtifact:
    """Generate native solver sources at formula-compile time.

    Sample values establish static parameter sparsity only. Live values are supplied
    by the DAG through the manifest's named buffers. The execution backend uses one
    persistent current-Clarabel workspace per worker and updates A/q/b because the
    risk factor is allowed to change on every problem.
    """
    from cvxpygen import cpg

    build = definition._fresh_build()
    missing = set(build.parameters) - set(sample_parameters)
    extra = set(sample_parameters) - set(build.parameters)
    if missing or extra:
        raise KeyError(
            f"sample parameter mismatch: missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for name, parameter in build.parameters.items():
        value = np.asarray(sample_parameters[name], dtype=np.float64)
        expected = tuple(int(v) for v in parameter.shape)
        if value.shape != expected:
            raise ValueError(
                f"sample parameter {name!r} has shape {value.shape}; "
                f"expected {expected}"
            )
        parameter.value = value

    destination = Path(code_dir).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    cpg.generate_code(
        build.problem,
        code_dir=str(destination),
        solver=solver,
        wrapper=wrapper,
    )
    manifest = {
        "schema_version": 1,
        "name": definition.name,
        "solver": solver,
        "update_regime": "A/q/b",
        "parameters": {
            name: {
                "shape": list(parameter.shape),
                "dtype": "float64",
            }
            for name, parameter in build.parameters.items()
        },
        "outputs": {
            name: {
                "kind": type(spec).__name__,
                "source": spec.name,
            }
            for name, spec in definition.outputs.items()
        },
        "runtime": {
            "workspace_ownership": "one persistent solver per worker",
            "batch_order": "stable",
            "sequential_mode": "single worker",
        },
    }
    manifest_path = destination / "cpp_stream_optimizer_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return CvxpygenArtifact(destination, manifest_path, solver)
