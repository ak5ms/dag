from trading_dsl_engine.cpp_new.codegen import emit_source
from trading_dsl_engine.cpp_new.lowering import lower
from trading_dsl_engine.jax_flat.compile import compile_formula


def test_cpp_new_ewm_ir_is_static_and_straight_line():
    program = compile_formula("ewm(close, 4.0)", cpp=False).program
    ir = lower(program, n_instruments=150)
    assert [n.opcode for n in ir.nodes] == ["input", "ewm"]
    assert ir.state_bytes and ir.state_bytes % 64 == 0
    source = emit_source(ir)
    assert "ewm_tick" in source
    assert "switch" not in source
    assert "tick(state, input.row(row), output.row(row), workers)" in source
    assert "lines.append" not in source
    assert ir.canonical_json() == ir.canonical_json()


def test_cpp_new_cache_alias_diagnostics():
    program = compile_formula("ewm(close, 4.0) + ewm(close, 4.0)", cpp=False).program
    # The shared frontend already canonicalizes repeated expressions. The native
    # lowering retains the single transition and its source mapping.
    ir = lower(program, n_instruments=8)
    assert sum(node.opcode == "ewm" for node in ir.nodes) == 1


def test_cpp_new_scratch_coloring_reuses_nonoverlapping_intervals():
    from trading_dsl_engine.cpp_new.lowering import _color_scratch

    slots, size = _color_scratch([(1, 64, 1, 2), (2, 64, 3, 4), (3, 128, 2, 5)])
    assert slots[0].offset == slots[1].offset
    assert slots[0].color == slots[1].color
    assert slots[2].offset != slots[0].offset
    assert size == 192


def test_cpp_new_cat_lifts_sibling_ewm_parameter_lanes():
    from trading_dsl_engine.cpp_new.lanes import discover_lane_graph
    program = compile_formula(
        "cat(ewm(close, 4.0), ewm(close, 8.0), ewm(close, 16.0))", cpp=False
    ).program
    ir = lower(program, n_instruments=150)
    assert ir.nodes[-1].opcode == "cat"
    assert ir.nodes[-1].value_type.width == 3
    assert ir.diagnostics.lifted_lanes == ((1, 2, 3),)
    graph = discover_lane_graph(ir)
    assert graph is not None
    assert graph.source_inputs == (0,)
    source = emit_source(ir)
    assert "cat_tick(v1, v2, v3, output)" in source


def test_cpp_new_cat_ewm_native_lane_batch_and_state_isolation(tmp_path):
    import numpy as np

    from trading_dsl_engine.cpp_new import compile_formula as compile_cpp_new
    from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_generic

    formula = "cat(ewm(close, 4.0), ewm(close, 8.0), ewm(close, 16.0))"
    values = np.array(
        [[1.0, np.nan, -0.0], [2.0, 4.0, np.inf], [np.nan, 8.0, 3.0], [5.0, -2.0, 6.0]],
        dtype=np.float64,
    )
    expected_runtime = compile_generic(formula)
    actual_runtime = compile_cpp_new(formula, cache_dir=tmp_path, n_instruments=3)
    assert actual_runtime.execution_tier == "fused-ewm-lane-native"
    _, expected = expected_runtime.run_batch((values,))
    state_a, actual = actual_runtime.run_batch((values,))
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    mapped = np.memmap(tmp_path / "cat_ewm.bin", mode="w+", dtype=np.float64, shape=actual.shape)
    _, mapped_result = actual_runtime.run_batch((values,), states=actual_runtime.init_state(3), out=mapped)
    assert mapped_result is mapped
    np.testing.assert_allclose(mapped, expected, equal_nan=True)
    for variant in ("lane-major", "instrument-major", "materialized"):
        ablated = np.empty_like(actual)
        actual_runtime.run_batch_ablation(
            actual_runtime.init_state(3), ablated, (values,), variant
        )
        np.testing.assert_allclose(ablated, expected, equal_nan=True)

    # A second state sharing the compiled lane family must start from zero state.
    state_b = actual_runtime.init_state(3)
    first_a = np.empty((3, 3))
    first_b = np.empty((3, 3))
    actual_runtime.tick_into(state_a, first_a, values[0])
    actual_runtime.tick_into(state_b, first_b, values[0])
    assert not np.allclose(first_a, first_b, equal_nan=True)
    np.testing.assert_allclose(first_b[0], np.ones(3), equal_nan=True)


def test_cpp_new_automatically_lifts_cross_sectional_rank_barrier(tmp_path):
    import numpy as np

    from trading_dsl_engine.cpp_new import compile_formula as compile_cpp_new
    from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_generic

    formula = "cat(xs_rank(ewm(close, 4.0)), xs_rank(ewm(close, 8.0)))"
    values = np.array(
        [[1.0, 1.0, np.nan, -0.0], [4.0, 2.0, 3.0, np.inf], [2.0, np.nan, 2.0, 2.0]],
        dtype=np.float64,
    )
    expected_runtime = compile_generic(formula)
    actual_runtime = compile_cpp_new(formula, cache_dir=tmp_path, n_instruments=4)
    assert actual_runtime.execution_tier == "fused-ewm-rank-lane-native"
    _, expected = expected_runtime.run_batch((values,))
    _, actual = actual_runtime.run_batch((values,))
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_cpp_new_lifts_multiple_cross_sectional_transitions(tmp_path):
    import numpy as np
    from trading_dsl_engine.cpp_new import compile_formula as compile_cpp_new
    from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_generic

    formula = (
        "cat(ewm(xs_rank(ewm(close, 4.0)), 8.0), "
        "ewm(xs_rank(ewm(close, 6.0)), 12.0))"
    )
    values = np.array([[1., 3., 2., np.nan], [4., 1., 2., 5.], [np.nan, 2., 8., 3.]])
    expected_runtime = compile_generic(formula)
    actual_runtime = compile_cpp_new(formula, cache_dir=tmp_path, n_instruments=4)
    assert actual_runtime.execution_tier == "fused-ewm-cross-sectional-pipeline-native"
    _, expected = expected_runtime.run_batch((values,))
    _, actual = actual_runtime.run_batch((values,))
    np.testing.assert_allclose(actual, expected, equal_nan=True, rtol=1e-12, atol=1e-12)


def test_lane_discovery_rejects_adversarial_mismatched_topology():
    from trading_dsl_engine.cpp_new.lanes import discover_lane_graph

    program = compile_formula(
        "cat(ewm(close, 4.0), xs_rank(ewm(close, 8.0)))", cpp=False
    ).program
    assert discover_lane_graph(lower(program, n_instruments=8)) is None
