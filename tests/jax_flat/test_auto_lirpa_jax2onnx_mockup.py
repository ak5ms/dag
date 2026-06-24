import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

from trading_dsl_engine.jax_flat import compile_formula


def _mockup_module():
    path = Path(__file__).resolve().parents[2] / "examples" / "auto_lirpa_jax2onnx_formula_mockup.py"
    spec = spec_from_file_location("auto_lirpa_jax2onnx_formula_mockup", path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_auto_lirpa_jax2onnx_mockup_formula_uses_custom_operators():
    module = _mockup_module()

    for token in ("xs_rank", "ewm", "groupby", "/", "-", "*"):
        assert token in module.EXAMPLE_FORMULA


def test_auto_lirpa_jax2onnx_mockup_uses_existing_runtime_operators():
    module = _mockup_module()
    runtime = module.build_existing_operator_runtime()

    lowered_names = {type(node.op).__name__ for node in runtime.program.nodes}

    assert "EwmOp" in lowered_names
    assert "GroupByOp" in lowered_names
    assert any(getattr(node.op, "cpp_name", None) == "xs_rank" for node in runtime.program.nodes)


def test_auto_lirpa_jax2onnx_mockup_jax_formula_matches_runtime():
    module = _mockup_module()
    close, open_, sector = module.example_inputs(time_steps=4, instruments=4)

    out = module.example_jax_formula(close, open_, sector)
    _, expected = compile_formula(module.EXAMPLE_FORMULA, cpp=False).run_batch(
        {"close": np.asarray(close), "open": np.asarray(open_), "sector": np.asarray(sector)},
        out_path=None,
    )

    np.testing.assert_allclose(np.asarray(out), np.asarray(expected), equal_nan=True)


def test_auto_lirpa_jax2onnx_mockup_overall_range_matches_materialized_output():
    module = _mockup_module()
    close, open_, sector = module.example_inputs(time_steps=6, instruments=4)

    materialized = module.example_jax_formula(close, open_, sector)
    overall = module.example_jax_formula_overall_range(close, open_, sector)

    finite = np.asarray(materialized)[np.isfinite(np.asarray(materialized))]
    np.testing.assert_allclose(np.asarray(overall), np.asarray([finite.min(), finite.max()]))


def test_auto_lirpa_jax2onnx_mockup_large_workload_estimate_avoids_full_output():
    module = _mockup_module()

    estimate = module.estimate_overall_range_workload(time_steps=3_000_000, instruments=9)

    assert estimate["input_bytes"] == 3_000_000 * 9 * 3 * 8
    assert estimate["per_timestep_output_bytes_avoided"] == 3_000_000 * 9 * 8
    assert estimate["aggregate_output_bytes"] == 16


def test_auto_lirpa_jax2onnx_mockup_shortcut_xs_rank_range_bound():
    module = _mockup_module()

    lower, upper = module.xs_rank_overall_range_bound(instruments=9)

    np.testing.assert_allclose([lower, upper], [-1.2815515655446004, 1.2815515655446004])


def test_auto_lirpa_jax2onnx_mockup_method_comparison_table():
    module = _mockup_module()

    rows = module.estimate_method_comparison(time_steps=3_000_000, instruments=9, representative_rows=1024)
    by_method = {row["method"]: row for row in rows}

    assert by_method["exact_streaming_overall_tick_scan"]["output_bytes"] == 16
    assert by_method["exact_streaming_overall_tick_scan"]["rows_evaluated"] == 3_000_000
    assert by_method["materialized_batch_scan_batch"]["output_bytes"] == 3_000_000 * 9 * 8
    assert by_method["representative_rows_empirical"]["relative_cell_work"] == 1024 / 3_000_000
    assert by_method["formula_shortcut_root_xs_rank"]["rows_evaluated"] == 0
    assert "auto_lirpa_ibp_aggregate_onnx" in by_method
    assert "auto_lirpa_crown_aggregate_onnx" in by_method
    table = module.format_method_comparison_markdown(rows)
    assert "| Method |" in table
    assert "Actual runtime (s)" in table


def test_auto_lirpa_jax2onnx_mockup_method_comparison_accepts_actual_runtimes():
    module = _mockup_module()

    rows = module.estimate_method_comparison(
        time_steps=8,
        instruments=4,
        actual_runtimes={"exact_streaming_overall_tick_scan": 0.123},
    )
    by_method = {row["method"]: row for row in rows}
    table = module.format_method_comparison_markdown(rows)

    assert by_method["exact_streaming_overall_tick_scan"]["actual_runtime_seconds"] == 0.123
    assert "0.123" in table


def test_auto_lirpa_jax2onnx_mockup_measure_actual_runtimes_smoke():
    module = _mockup_module()

    timings = module.measure_actual_runtimes(time_steps=4, instruments=4, representative_rows=2)

    assert timings["materialized_batch_scan_batch"] >= 0.0
    assert timings["exact_streaming_overall_tick_scan"] >= 0.0
    assert timings["formula_shortcut_root_xs_rank"] >= 0.0
    assert timings["representative_rows_empirical"] >= 0.0
    assert timings["auto_lirpa_ibp_aggregate_onnx"] is None
    assert timings["auto_lirpa_crown_aggregate_onnx"] is None


def test_auto_lirpa_jax2onnx_mockup_measure_lirpa_attempts_report_status():
    module = _mockup_module()

    timings = module.measure_actual_runtimes(time_steps=2, instruments=2, representative_rows=1, include_lirpa=True)

    assert timings["auto_lirpa_ibp_aggregate_onnx"] is not None
    assert timings["auto_lirpa_crown_aggregate_onnx"] is not None
    rows = module.estimate_method_comparison(time_steps=2, instruments=2, actual_runtimes=timings)
    table = module.format_method_comparison_markdown(rows)
    assert "auto_lirpa_ibp_aggregate_onnx" in table
    assert "auto_lirpa_crown_aggregate_onnx" in table
    assert ("failed after" in table) or isinstance(timings["auto_lirpa_ibp_aggregate_onnx"], float)
