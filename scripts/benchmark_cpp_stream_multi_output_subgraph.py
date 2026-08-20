from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("CPP_STREAM_MULTI_ROWS", "5000000"))
HET_ROWS = int(os.environ.get("CPP_STREAM_MULTI_HET_ROWS", "1000000"))
FUSED_ROWS = int(os.environ.get("CPP_STREAM_MULTI_FUSED_ROWS", "1000000"))
DUP_ROWS = int(os.environ.get("CPP_STREAM_MULTI_DUP_ROWS", "1000000"))
INSTRUMENTS = int(os.environ.get("CPP_STREAM_MULTI_INSTRUMENTS", "9"))
WARMUPS = int(os.environ.get("CPP_STREAM_MULTI_WARMUPS", "1"))
RUNS = int(os.environ.get("CPP_STREAM_MULTI_RUNS", "10"))
OUTPUT_DIR = Path(os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR", tempfile.gettempdir()))

SUBGRAPH = "ewm(x + 1, span=32)"
TOP = f"xs_rank({SUBGRAPH})"
CAT_BOTH = f"cat({SUBGRAPH}, {TOP})"

# A materially heterogeneous pair: first public result is (N,), second is (N, 8).
# The equivalent Cat emits the same nine float64 values per instrument and row.
HET_FEATURES = (
    SUBGRAPH,
    TOP,
    "ewm(x + 1, span=64)",
    "xs_rank(ewm(x + 1, span=64))",
    "ewm(x + 1, span=128)",
    "xs_rank(ewm(x + 1, span=128))",
    "ewm(x + 1, span=256)",
    "xs_rank(ewm(x + 1, span=256))",
)
HET_WIDE = "cat(" + ", ".join(HET_FEATURES) + ")"
HET_CAT = f"cat({SUBGRAPH}, {HET_WIDE})"

# Same-span EWMs are automatically bundled and a trailing Cat may be fused into
# the bundle as an epilogue. Exposing FUSED_X must not silently destroy that
# ceiling without us measuring it.
FUSED_X = "ewm(x, span=32)"
FUSED_Y = "ewm(y, span=32)"
FUSED_TOP = f"cat({FUSED_X}, {FUSED_Y})"
FUSED_EQUIVALENT_CAT = f"cat({FUSED_X}, {FUSED_TOP})"

# Exact duplicate roots exercise packed-output reuse; distinct sibling roots
# exercise the generic projection bundle's cross-root expression cache.
DUP_EXPR = (
    "sqrt(abs(sin(x * 1.000001 + y * 0.999999))) "
    "+ tanh((x - y) * (x + y)) + exp(abs(x - y) * -1.0)"
)
SIBLING_SHARED = "sin(x * 1.000001 + y * 0.999999) + tanh(x - y)"
SIBLING_LEFT = f"sqrt(abs({SIBLING_SHARED}))"
SIBLING_RIGHT = f"({SIBLING_SHARED}) * 2 + 1"
SIBLING_CAT = f"cat({SIBLING_LEFT}, {SIBLING_RIGHT})"


def _benchmark(
    name: str,
    formula,
    data: dict[str, np.ndarray],
    path: Path,
    *,
    rows: int,
):
    runtime = compile_formula(formula, data, n_instruments=INSTRUMENTS)
    for _ in range(WARMUPS):
        runtime.run(out_path=path, threads=1)
    timings = [runtime.run(out_path=path, threads=1).seconds for _ in range(RUNS)]
    med = median(timings)
    print(f"case={name}")
    print(f"formula={formula!r}")
    print(f"stages={[stage.kind for stage in runtime.plan.stages]}")
    print(f"scratch_slots={runtime.plan.scratch_slots}")
    print(f"matrix_scratch_slots={runtime.plan.matrix_scratch_slots}")
    print(f"median_seconds={med:.9f}")
    print(f"mean_seconds={mean(timings):.9f}")
    print(f"best_seconds={min(timings):.9f}")
    print("runs_seconds=" + ",".join(f"{value:.9f}" for value in timings))
    print(f"million_rows_per_second={rows / med / 1e6:.6f}")
    print(f"output_bytes={path.stat().st_size}")
    print(f"generated_cpp={runtime.generated_cpp}")
    print("---")
    return med, runtime


def _benchmark_equal_width(data: dict[str, np.ndarray]) -> None:
    bench_path = OUTPUT_DIR / "cpp_stream_multi_subgraph_bench.bin"
    top_seconds, top_runtime = _benchmark(
        "top_only", TOP, data, bench_path, rows=ROWS
    )
    cat_seconds, _ = _benchmark(
        "cat_subgraph_and_top", CAT_BOTH, data, bench_path, rows=ROWS
    )
    sub_top_seconds, sub_top_runtime = _benchmark(
        "subgraph_then_top", [SUBGRAPH, TOP], data, bench_path, rows=ROWS
    )
    top_sub_seconds, top_sub_runtime = _benchmark(
        "top_then_subgraph", [TOP, SUBGRAPH], data, bench_path, rows=ROWS
    )

    print(f"cat_over_top_only={cat_seconds / top_seconds - 1.0:.6%}")
    print(f"subgraph_then_top_overhead={sub_top_seconds / top_seconds - 1.0:.6%}")
    print(f"top_then_subgraph_overhead={top_sub_seconds / top_seconds - 1.0:.6%}")
    print(f"multi_over_equivalent_cat={sub_top_seconds / cat_seconds - 1.0:.6%}")
    print(
        "ewm_nodes="
        f"{sum(type(node.op).__name__ == 'EwmOp' for node in sub_top_runtime.program.nodes)}"
    )
    generated = sub_top_runtime.generated_cpp.read_text()
    entrypoints = generated.count('extern "C" int cpp_stream_run_arrays(')
    print(f"single_native_entrypoints={entrypoints}")

    top_result = top_runtime.run(
        out_path=OUTPUT_DIR / "cpp_stream_multi_top_check.bin", threads=1
    )
    multi_result = sub_top_runtime.run(
        out_path=OUTPUT_DIR / "cpp_stream_multi_check.bin", threads=1
    )
    top_values = top_result.load(mmap_mode="r")
    sub_values, multi_top_values = multi_result.load(mmap_mode="r")
    np.testing.assert_allclose(
        multi_top_values[-128:], top_values[-128:], rtol=0.0, atol=0.0, equal_nan=True
    )
    assert sub_values.shape == (ROWS, INSTRUMENTS)
    assert top_sub_runtime.return_multiple
    assert entrypoints == 1


def _benchmark_heterogeneous_width() -> None:
    rng = np.random.default_rng(20260820)
    x = rng.normal(size=(HET_ROWS, INSTRUMENTS)).astype(np.float64)
    data = {"x": x}
    bench_path = OUTPUT_DIR / "cpp_stream_multi_heterogeneous_bench.bin"

    cat_seconds, cat_runtime = _benchmark(
        "heterogeneous_equivalent_cat",
        HET_CAT,
        data,
        bench_path,
        rows=HET_ROWS,
    )
    multi_seconds, multi_runtime = _benchmark(
        "heterogeneous_list_n_and_nx8",
        [SUBGRAPH, HET_WIDE],
        data,
        bench_path,
        rows=HET_ROWS,
    )

    print(f"heterogeneous_multi_over_cat={multi_seconds / cat_seconds - 1.0:.6%}")
    print(f"heterogeneous_cat_stages={[stage.kind for stage in cat_runtime.plan.stages]}")
    print(f"heterogeneous_multi_stages={[stage.kind for stage in multi_runtime.plan.stages]}")
    print(f"heterogeneous_multi_shapes={multi_runtime.output_layout.outputs[0].shape,multi_runtime.output_layout.outputs[1].shape}")
    assert multi_runtime.output_layout.outputs[0].size == INSTRUMENTS
    assert multi_runtime.output_layout.outputs[1].size == INSTRUMENTS * 8
    assert cat_runtime.output_layout.row_width == multi_runtime.output_layout.row_width

    generated = multi_runtime.generated_cpp.read_text()
    assert f"        {INSTRUMENTS}," in generated
    assert f"        {INSTRUMENTS * 8}," in generated


def _benchmark_bundle_fusion() -> None:
    rng = np.random.default_rng(20260821)
    x = rng.normal(size=(FUSED_ROWS, INSTRUMENTS)).astype(np.float64)
    y = rng.normal(size=(FUSED_ROWS, INSTRUMENTS)).astype(np.float64)
    data = {"x": x, "y": y}
    bench_path = OUTPUT_DIR / "cpp_stream_multi_fused_ewm_bench.bin"

    cat_seconds, cat_runtime = _benchmark(
        "fused_ewm_equivalent_cat",
        FUSED_EQUIVALENT_CAT,
        data,
        bench_path,
        rows=FUSED_ROWS,
    )
    multi_seconds, multi_runtime = _benchmark(
        "fused_ewm_public_member",
        [FUSED_X, FUSED_TOP],
        data,
        bench_path,
        rows=FUSED_ROWS,
    )

    print(f"fused_ewm_multi_over_cat={multi_seconds / cat_seconds - 1.0:.6%}")
    print(f"fused_ewm_cat_stages={[stage.kind for stage in cat_runtime.plan.stages]}")
    print(f"fused_ewm_multi_stages={[stage.kind for stage in multi_runtime.plan.stages]}")
    print(f"fused_ewm_cat_scratch={cat_runtime.plan.scratch_slots.counts}")
    print(f"fused_ewm_multi_scratch={multi_runtime.plan.scratch_slots.counts}")
    assert cat_runtime.output_layout.row_width == multi_runtime.output_layout.row_width
    assert [stage.kind for stage in cat_runtime.plan.stages] == ["ewm_bundle"]
    assert [stage.kind for stage in multi_runtime.plan.stages] == ["ewm_bundle"]
    assert cat_runtime.plan.scratch_slots == 0
    assert multi_runtime.plan.scratch_slots == 0


def _benchmark_lazy_projection_cse() -> None:
    rng = np.random.default_rng(20260822)
    x = rng.normal(size=(DUP_ROWS, INSTRUMENTS)).astype(np.float64)
    y = rng.normal(size=(DUP_ROWS, INSTRUMENTS)).astype(np.float64)
    data = {"x": x, "y": y}
    bench_path = OUTPUT_DIR / "cpp_stream_multi_lazy_projection_bench.bin"

    input_single, _ = _benchmark(
        "duplicate_control_input_single", "x", data, bench_path, rows=DUP_ROWS
    )
    input_double, input_double_runtime = _benchmark(
        "duplicate_control_input_list", ["x", "x"], data, bench_path, rows=DUP_ROWS
    )
    expr_single, _ = _benchmark(
        "duplicate_lazy_expr_single", DUP_EXPR, data, bench_path, rows=DUP_ROWS
    )
    expr_double, expr_double_runtime = _benchmark(
        "duplicate_lazy_expr_list", [DUP_EXPR, DUP_EXPR], data, bench_path, rows=DUP_ROWS
    )
    expr_cat, expr_cat_runtime = _benchmark(
        "duplicate_lazy_expr_cat",
        f"cat({DUP_EXPR}, {DUP_EXPR})",
        data,
        bench_path,
        rows=DUP_ROWS,
    )
    sibling_list, sibling_runtime = _benchmark(
        "sibling_lazy_expr_list",
        [SIBLING_LEFT, SIBLING_RIGHT],
        data,
        bench_path,
        rows=DUP_ROWS,
    )
    sibling_cat, sibling_cat_runtime = _benchmark(
        "sibling_lazy_expr_cat",
        SIBLING_CAT,
        data,
        bench_path,
        rows=DUP_ROWS,
    )

    input_increment = input_double - input_single
    expr_increment = expr_double - expr_single
    print(f"duplicate_input_increment_seconds={input_increment:.9f}")
    print(f"duplicate_expr_increment_seconds={expr_increment:.9f}")
    print(f"duplicate_expr_increment_over_input={expr_increment / input_increment:.6f}x")
    print(f"duplicate_expr_list_over_cat={expr_double / expr_cat - 1.0:.6%}")
    print(f"sibling_expr_list_over_cat={sibling_list / sibling_cat - 1.0:.6%}")
    print(f"duplicate_input_list_stages={[stage.kind for stage in input_double_runtime.plan.stages]}")
    print(f"duplicate_expr_list_stages={[stage.kind for stage in expr_double_runtime.plan.stages]}")
    print(f"duplicate_expr_cat_stages={[stage.kind for stage in expr_cat_runtime.plan.stages]}")
    print(f"sibling_expr_list_stages={[stage.kind for stage in sibling_runtime.plan.stages]}")
    print(f"sibling_expr_cat_stages={[stage.kind for stage in sibling_cat_runtime.plan.stages]}")
    assert len(set(expr_double_runtime.program.outputs)) == 1
    assert expr_double_runtime.output_layout.row_width == expr_cat_runtime.output_layout.row_width
    assert [stage.kind for stage in sibling_runtime.plan.stages] == ["copy_bundle"]
    assert sibling_runtime.output_layout.row_width == sibling_cat_runtime.output_layout.row_width


def main() -> None:
    if (
        ROWS <= 0
        or HET_ROWS <= 0
        or FUSED_ROWS <= 0
        or DUP_ROWS <= 0
        or INSTRUMENTS <= 0
        or WARMUPS < 0
        or RUNS <= 0
    ):
        raise ValueError("invalid benchmark dimensions or run counts")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(20260819)
    x = rng.normal(size=(ROWS, INSTRUMENTS)).astype(np.float64)
    _benchmark_equal_width({"x": x})
    _benchmark_heterogeneous_width()
    _benchmark_bundle_fusion()
    _benchmark_lazy_projection_cse()


if __name__ == "__main__":
    main()
