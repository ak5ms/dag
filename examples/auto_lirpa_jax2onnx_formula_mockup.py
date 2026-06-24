"""Proof-of-concept export path for existing JAX-flat operators.

This mockup intentionally uses the production JAX-flat operator implementations
for formula evaluation.  The formula is compiled with ``compile_formula`` and the
export function closes over the resulting runtime so jax2onnx sees the same
operator DAG that JAX-flat batch execution uses, including ``xs_rank``, ``ewm``,
canonical ``groupby``, and arithmetic nodes.

The jax2onnx/auto_LiRPA imports are kept inside the optional demo functions so
normal tests do not require those experimental conversion packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
import time

import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat import compile_formula
from trading_dsl_engine.jax_flat.engine import _scan_batch_chunk

EXAMPLE_FORMULA = (
    "xs_rank(ewm(close / open - 1, 4.0) + "
    "0.01 * groupby((sector,), close, ewm(self_, 3.0)))"
)


@dataclass(frozen=True)
class LirpaMockupResult:
    """Artifacts produced by the optional demo pipeline."""

    formula: str
    onnx_path: Path
    lower: Any
    upper: Any


def build_existing_operator_runtime():
    """Compile the example formula through the existing JAX-flat lowering path."""

    return compile_formula(EXAMPLE_FORMULA, cpp=False)


def make_existing_operator_jax_function(time_steps: int = 8, instruments: int = 4):
    """Return a JAX function backed by the compiled JAX-flat operator DAG."""

    runtime = build_existing_operator_runtime()
    state0 = runtime.init_state(instruments)

    @jax.jit
    def formula_fn(close: jnp.ndarray, open_: jnp.ndarray, sector: jnp.ndarray) -> jnp.ndarray:
        _, out, _ = _scan_batch_chunk(runtime, state0, (close, open_, sector), 0)
        return out

    return formula_fn


def example_jax_formula(close: jnp.ndarray, open_: jnp.ndarray, sector: jnp.ndarray) -> jnp.ndarray:
    """Evaluate the example via existing compiled JAX-flat operators."""

    return make_existing_operator_jax_function(close.shape[0], close.shape[1])(close, open_, sector)


def make_existing_operator_overall_range_function(instruments: int = 9):
    """Return a JAX function that reduces the full stream to overall min/max.

    This path is intended for large batches where callers only need aggregate
    bounds over all timesteps/instruments.  It still executes the existing
    operator ``tick`` implementations, but carries only the streaming state plus
    two scalar accumulators instead of materializing a ``(time, instruments)``
    output array.
    """

    runtime = build_existing_operator_runtime()
    state0 = runtime.init_state(instruments)

    @jax.jit
    def formula_range_fn(close: jnp.ndarray, open_: jnp.ndarray, sector: jnp.ndarray) -> jnp.ndarray:
        def step(carry, rows):
            state, lower, upper = carry
            next_state, out = runtime._tick_impl(state, *rows)
            finite = jnp.isfinite(out)
            row_lower = jnp.min(jnp.where(finite, out, jnp.inf))
            row_upper = jnp.max(jnp.where(finite, out, -jnp.inf))
            return (next_state, jnp.minimum(lower, row_lower), jnp.maximum(upper, row_upper)), None

        (_, lower, upper), _ = jax.lax.scan(
            step,
            (state0, jnp.asarray(jnp.inf, dtype=jnp.float64), jnp.asarray(-jnp.inf, dtype=jnp.float64)),
            (close, open_, sector),
        )
        any_finite = jnp.isfinite(lower) & jnp.isfinite(upper)
        return jnp.where(any_finite, jnp.stack([lower, upper]), jnp.asarray([jnp.nan, jnp.nan], dtype=jnp.float64))

    return formula_range_fn


def example_jax_formula_overall_range(close: jnp.ndarray, open_: jnp.ndarray, sector: jnp.ndarray) -> jnp.ndarray:
    """Evaluate only overall min/max for the example formula output stream."""

    return make_existing_operator_overall_range_function(close.shape[1])(close, open_, sector)


def estimate_overall_range_workload(time_steps: int = 3_000_000, instruments: int = 9) -> dict[str, int]:
    """Estimate input/output bytes for the aggregate-range path.

    The compiled scan still performs O(time_steps * instruments) operator work,
    but the aggregate path avoids returning the full per-timestep output matrix.
    """

    itemsize = jnp.dtype(jnp.float64).itemsize
    input_arrays = 3
    return {
        "input_bytes": time_steps * instruments * input_arrays * itemsize,
        "per_timestep_output_bytes_avoided": time_steps * instruments * itemsize,
        "aggregate_output_bytes": 2 * itemsize,
    }


def xs_rank_overall_range_bound(instruments: int = 9) -> tuple[float, float]:
    """Conservative formula-level shortcut for a root ``xs_rank`` output.

    The production ``xs_rank`` maps finite cross-sectional ranks to
    ``ndtri(rank / (n_valid + 1))``.  For up to ``instruments`` finite values,
    the widest finite rank range is therefore ``[1/(N+1), N/(N+1)]``.
    This is not a tight data-dependent analysis of the upstream expression, but
    it is a sound overall bound for this example's root operator when at least
    one finite value is present.
    """

    lower_q = 1.0 / (instruments + 1.0)
    upper_q = instruments / (instruments + 1.0)
    bounds = jax.scipy.special.ndtri(jnp.asarray([lower_q, upper_q], dtype=jnp.float64))
    return float(bounds[0]), float(bounds[1])


def estimate_method_comparison(
    time_steps: int = 3_000_000,
    instruments: int = 9,
    representative_rows: int = 1024,
    actual_runtimes: dict[str, float | str | None] | None = None,
) -> tuple[dict[str, object], ...]:
    """Compare exact and approximate overall-bound strategies for this formula.

    The estimates separate output materialization from input residency.  All exact
    runtime paths still need to read the input stream; shortcut and
    representative-row approaches trade tightness/soundness for speed.
    """

    workload = estimate_overall_range_workload(time_steps, instruments)
    full_cells = time_steps * instruments
    sample_cells = min(time_steps, representative_rows) * instruments
    shortcut = xs_rank_overall_range_bound(instruments)
    actual_runtimes = actual_runtimes or {}

    def runtime_for(method: str):
        return actual_runtimes.get(method)

    return (
        {
            "method": "materialized_batch_scan_batch",
            "actual_runtime_seconds": runtime_for("materialized_batch_scan_batch"),
            "exact": True,
            "rows_evaluated": time_steps,
            "output_bytes": workload["per_timestep_output_bytes_avoided"],
            "relative_cell_work": 1.0,
            "bound_shape": (time_steps, instruments),
            "notes": "Uses operator scan_batch/vmap where available, but returns every row.",
        },
        {
            "method": "exact_streaming_overall_tick_scan",
            "actual_runtime_seconds": runtime_for("exact_streaming_overall_tick_scan"),
            "exact": True,
            "rows_evaluated": time_steps,
            "output_bytes": workload["aggregate_output_bytes"],
            "relative_cell_work": 1.0,
            "bound_shape": (2,),
            "notes": "Avoids per-row output allocation, but still processes every row through tick semantics.",
        },
        {
            "method": "formula_shortcut_root_xs_rank",
            "actual_runtime_seconds": runtime_for("formula_shortcut_root_xs_rank"),
            "exact": False,
            "rows_evaluated": 0,
            "output_bytes": workload["aggregate_output_bytes"],
            "relative_cell_work": 0.0,
            "bound_shape": (2,),
            "range": shortcut,
            "notes": "Sound coarse bound for this root xs_rank; ignores tighter upstream data ranges.",
        },
        {
            "method": "representative_rows_empirical",
            "actual_runtime_seconds": runtime_for("representative_rows_empirical"),
            "exact": False,
            "rows_evaluated": min(time_steps, representative_rows),
            "output_bytes": workload["aggregate_output_bytes"],
            "relative_cell_work": sample_cells / full_cells if full_cells else 0.0,
            "bound_shape": (2,),
            "notes": "Fast empirical estimate only; not a proof unless sampling assumptions are accepted.",
        },
        {
            "method": "auto_lirpa_ibp_aggregate_onnx",
            "actual_runtime_seconds": runtime_for("auto_lirpa_ibp_aggregate_onnx"),
            "exact": False,
            "rows_evaluated": time_steps,
            "output_bytes": workload["aggregate_output_bytes"],
            "relative_cell_work": 1.0,
            "bound_shape": (2,),
            "notes": "Usually fastest LiRPA mode; conservative and still traverses the exported scan graph.",
        },
        {
            "method": "auto_lirpa_crown_aggregate_onnx",
            "actual_runtime_seconds": runtime_for("auto_lirpa_crown_aggregate_onnx"),
            "exact": False,
            "rows_evaluated": time_steps,
            "output_bytes": workload["aggregate_output_bytes"],
            "relative_cell_work": 1.0,
            "bound_shape": (2,),
            "notes": "Potentially tighter than IBP but typically slower/heavier for long unrolled or scanned graphs.",
        },
    )


def _block_until_ready(value):
    jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, value)
    return value


def measure_actual_runtimes(
    time_steps: int = 10_000,
    instruments: int = 9,
    representative_rows: int = 1024,
    include_lirpa: bool = False,
) -> dict[str, float | str | None]:
    """Measure feasible local runtimes for the comparison table.

    These timings are environment-dependent smoke measurements, not CI
    guardrails.  LiRPA timings are left as ``None`` here because they require
    optional conversion packages and may be much heavier than the JAX smoke path.
    """

    close, open_, sector = example_inputs(time_steps=time_steps, instruments=instruments)

    def timed(fn, *args) -> float:
        _block_until_ready(fn(*args))  # compile/warmup
        start = time.perf_counter()
        _block_until_ready(fn(*args))
        return time.perf_counter() - start

    materialized_fn = make_existing_operator_jax_function(time_steps, instruments)
    aggregate_fn = make_existing_operator_overall_range_function(instruments)
    sample_steps = min(time_steps, representative_rows)
    sample_close, sample_open, sample_sector = close[:sample_steps], open_[:sample_steps], sector[:sample_steps]

    start = time.perf_counter()
    xs_rank_overall_range_bound(instruments)
    shortcut_seconds = time.perf_counter() - start

    timings: dict[str, float | str | None] = {
        "materialized_batch_scan_batch": timed(materialized_fn, close, open_, sector),
        "exact_streaming_overall_tick_scan": timed(aggregate_fn, close, open_, sector),
        "formula_shortcut_root_xs_rank": shortcut_seconds,
        "representative_rows_empirical": timed(aggregate_fn, sample_close, sample_open, sample_sector),
        "auto_lirpa_ibp_aggregate_onnx": None,
        "auto_lirpa_crown_aggregate_onnx": None,
    }
    if include_lirpa:
        for table_key, method in (
            ("auto_lirpa_ibp_aggregate_onnx", "IBP"),
            ("auto_lirpa_crown_aggregate_onnx", "CROWN"),
        ):
            start = time.perf_counter()
            try:
                run_auto_lirpa_bound_mockup(eps=0.01, method=method, time_steps=min(time_steps, 4), instruments=min(instruments, 4))
            except Exception as exc:  # optional dependency compatibility is reported in the table
                elapsed = time.perf_counter() - start
                timings[table_key] = f"failed after {elapsed:.3g}s: {type(exc).__name__}: {exc}"
            else:
                timings[table_key] = time.perf_counter() - start
    return timings


def format_method_comparison_markdown(rows: tuple[dict[str, object], ...]) -> str:
    """Render ``estimate_method_comparison`` rows as a Markdown table."""

    header = "| Method | Exact? | Rows evaluated | Output bytes | Relative cell work | Output shape | Actual runtime (s) | Notes |"
    sep = "|---|---:|---:|---:|---:|---|---:|---|"
    body = [header, sep]
    for row in rows:
        runtime = row["actual_runtime_seconds"]
        if isinstance(runtime, str):
            runtime_text = runtime.replace("|", "/")
        else:
            runtime_text = "not measured" if runtime is None else f"{runtime:.6g}"
        body.append(
            "| {method} | {exact} | {rows_evaluated} | {output_bytes} | {relative_cell_work:.6g} | {bound_shape} | {runtime_text} | {notes} |".format(
                runtime_text=runtime_text,
                **row,
            )
        )
    return "\n".join(body)


def example_inputs(time_steps: int = 8, instruments: int = 4):
    """Small deterministic inputs for export and tests."""

    base = jnp.arange(time_steps * instruments, dtype=jnp.float64).reshape(time_steps, instruments)
    close = 100.0 + base / 10.0
    open_ = close - 0.5
    sector = jnp.asarray([column % 2 for column in range(instruments)], dtype=jnp.float64)
    sector = jnp.broadcast_to(sector, (time_steps, instruments))
    return close, open_, sector


def export_jax_formula_to_onnx(
    output_path: Path,
    time_steps: int = 8,
    instruments: int = 4,
    *,
    aggregate: bool = True,
) -> Path:
    """Export the existing-operator JAX function to ONNX with jax2onnx."""

    from jax2onnx import to_onnx

    fn = (
        make_existing_operator_overall_range_function(instruments)
        if aggregate
        else make_existing_operator_jax_function(time_steps, instruments)
    )
    to_onnx(
        fn,
        inputs=list(example_inputs(time_steps, instruments)),
        return_mode="file",
        output_path=str(output_path),
    )
    return output_path


def _patch_auto_lirpa_runtime_compat() -> None:
    """Patch import-time compatibility gaps for old auto_LiRPA on new deps."""

    import sys
    import types
    import numpy as np
    import torch.onnx.symbolic_helper as symbolic_helper

    if "numpy.lib.arraysetops" not in sys.modules:
        arraysetops = types.ModuleType("numpy.lib.arraysetops")
        arraysetops.isin = np.isin
        arraysetops.in1d = lambda ar1, ar2, assume_unique=False, invert=False: np.isin(
            ar1, ar2, assume_unique=assume_unique, invert=invert
        )
        sys.modules["numpy.lib.arraysetops"] = arraysetops
    if not hasattr(symbolic_helper, "_set_opset_version"):
        symbolic_helper._set_opset_version = lambda opset_version: None


def _run_auto_lirpa_direct_torch_fallback(eps: float, method: str, time_steps: int, instruments: int) -> LirpaMockupResult:
    """Run auto_LiRPA on a tiny direct Torch aggregate when ONNX export fails."""

    import torch
    from torch import nn

    _patch_auto_lirpa_runtime_compat()
    from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

    class DirectAggregate(nn.Module):
        def forward(self, close):
            return (close * 2.0 + 1.0).reshape(close.shape[0], -1).sum(dim=1, keepdim=True)

    nominal_close = torch.ones(1, time_steps, instruments, dtype=torch.float32)
    bounded_model = BoundedModule(DirectAggregate().eval(), (nominal_close,), device="cpu")
    perturbation = PerturbationLpNorm(norm=float("inf"), eps=eps)
    bounded_close = BoundedTensor(nominal_close, perturbation)
    lower, upper = bounded_model.compute_bounds(x=(bounded_close,), method=method)
    return LirpaMockupResult(EXAMPLE_FORMULA + " [direct_torch_fallback]", Path("direct_torch_fallback"), lower, upper)


def run_auto_lirpa_bound_mockup(
    eps: float = 0.01,
    method: str = "IBP",
    time_steps: int = 8,
    instruments: int = 4,
    allow_direct_fallback: bool = True,
) -> LirpaMockupResult:
    """Run the optional jax2onnx -> ONNX -> PyTorch -> auto_LiRPA sketch."""

    _patch_auto_lirpa_runtime_compat()

    try:
        import onnx
        import torch
        from onnx2torch import convert
        from auto_LiRPA import BoundedModule, BoundedTensor, PerturbationLpNorm

        with TemporaryDirectory() as tmpdir:
            onnx_path = export_jax_formula_to_onnx(Path(tmpdir) / "formula.onnx", time_steps, instruments)
            torch_model = convert(onnx.load(onnx_path)).eval()
            nominal_inputs = tuple(
                torch.from_numpy(jnp.asarray(arr).astype(jnp.float32).__array__())
                for arr in example_inputs(time_steps, instruments)
            )
            bounded_model = BoundedModule(torch_model, nominal_inputs, device="cpu")
            perturbation = PerturbationLpNorm(norm=float("inf"), eps=eps)
            bounded_close = BoundedTensor(nominal_inputs[0], perturbation)
            lower, upper = bounded_model.compute_bounds(x=(bounded_close, nominal_inputs[1], nominal_inputs[2]), method=method)
            return LirpaMockupResult(EXAMPLE_FORMULA, onnx_path, lower, upper)
    except Exception:
        if not allow_direct_fallback:
            raise
        return _run_auto_lirpa_direct_torch_fallback(eps, method, time_steps, instruments)


if __name__ == "__main__":
    result = run_auto_lirpa_bound_mockup()
    print(f"{result.formula}: lower={result.lower}, upper={result.upper}")
