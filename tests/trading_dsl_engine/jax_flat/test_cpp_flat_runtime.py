import time

import jax
import pytest
import numpy as np
from trading_dsl_engine.jax_flat import compile_formula
from trading_dsl_engine.jax_flat.engine_cpp import (
    BroadcastMode,
    compile_formula as compile_formula_native,
    inspect_hybrid_partition,
    lower_native_plan,
)


def _assert_cpp_matches_jax(formula, data, *, rtol=1e-10, atol=1e-10):
    cpp_runtime = compile_formula_native(formula)
    jax_runtime = compile_formula(formula)
    _, cpp_out = cpp_runtime.run_batch(data)
    _, jax_out = jax_runtime.run_batch(data)
    np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=rtol, atol=atol, equal_nan=True)


def _assert_cpp_matches_pure_jax(formula, data, *, rtol=1e-10, atol=1e-10):
    cpp_runtime = compile_formula_native(formula)
    jax_runtime = compile_formula(formula, cpp=False)
    _, cpp_out = cpp_runtime.run_batch(data)
    _, jax_out = jax_runtime.run_batch(data)
    np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=rtol, atol=atol, equal_nan=True)


def test_cpp_flat_stateless_chain_matches_jax_flat():
    rows = 32
    cols = 5
    close = np.linspace(-2.0, 3.0, rows * cols, dtype=np.float64).reshape(rows, cols)
    open_ = np.linspace(1.0, 4.0, rows * cols, dtype=np.float64).reshape(rows, cols)
    close[3, 2] = np.nan
    open_[7, 1] = np.nan
    _assert_cpp_matches_jax(
        "xstd(add(abs(close), div(exp(fraction(open)), add(abs(close), 1.0))))",
        {"close": close, "open": open_},
    )


def test_cpp_flat_more_stateless_and_matrix_ops_match_jax_flat():
    rows = 24
    cols = 4
    close = np.linspace(-1.0, 1.0, rows * cols, dtype=np.float64).reshape(rows, cols)
    open_ = np.flip(close, axis=1) + 0.25
    close[4, 2] = np.nan
    open_[6, 1] = np.nan
    data = {"close": close, "open": open_}
    for formula in (
        "cat(close, open, bspline(fillna(close, 0.25), 4))",
        "col(cat(close, open, bspline(fillna(close, 0.25), 4)), 5)",
        "xs_rank(add(close, open))",
        "xs_sort(add(close, open))",
        "mean(add(close, open))",
        "where(gt(open, close), open, close)",
        "add(add(ln(abs(add(close, 2.0))), ceil(open)), add(floor(close), round(open)))",
        "add(add(sign(close), arctan(open)), add(isnan(close), purify(close)))",
        "add(add(mod(close, 0.7), pow(abs(open), 2.0)), floordiv(open, 0.5))",
        "add(add(and(gt(open, close), lt(close, open)), or(eq(open, close), ne(open, close))), xor(gt(open, close), lt(open, close)))",
    ):
        _assert_cpp_matches_jax(formula, data)


def test_cpp_flat_stateful_cumsum_ewm_shift_ffill_matches_jax_flat():
    rows = 40
    cols = 4
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    close = row * 0.25 + col
    lag = np.mod(row + col, 4.0)
    close[5, 1] = np.nan
    close[9, 3] = np.nan
    data = {"close": close, "lag": lag}
    _assert_cpp_matches_jax("add(cumsum(close), shift(ewm(ffill(close, 1), 3.0), lag, 5))", data)


def test_cpp_flat_default_lag_shift_matches_jax_flat():
    rows = 24
    cols = 3
    close = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) * 0.1
    close[4, 1] = np.nan
    data = {"close": close}
    for formula in ("shift(close)", "shift(cumsum(close))", "shift(isnan(close))"):
        _assert_cpp_matches_jax(formula, data)


def test_cpp_flat_default_lag_shift_with_runtime_cache_matches_jax_flat():
    x = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, np.nan], [7.0, 8.0]], dtype=np.float64)
    cache_runtime = compile_formula("cache(x)", cpp=False)
    cache_runtime.run_batch({"x": x})
    data = {"x": x}
    formula = "shift(isnan(x))"
    cpp_runtime = compile_formula(formula, runtimes=cache_runtime, cpp=True)
    jax_runtime = compile_formula(formula, runtimes=cache_runtime, cpp=False)
    _, cpp_out = cpp_runtime.run_batch(data)
    _, jax_out = jax_runtime.run_batch(data)
    np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=1e-10, atol=1e-10, equal_nan=True)


def test_cpp_flat_ridge_projections_match_jax_flat():
    rows = 16
    cols = 5
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    close = 0.2 * row + col
    open_ = 1.0 + 0.1 * row - 0.05 * col
    close[3, 2] = np.nan
    open_[7, 1] = np.nan
    data = {"close": close, "open": open_}
    _assert_cpp_matches_jax("get_preds(Ridge(cat(close, open), open, 1.0, 8.0, 0.01))", data)
    _assert_cpp_matches_jax("get_beta(Ridge(cat(close, open), open, 1.0, 8.0, 0.01))", data)



def test_cpp_flat_instant_ridge_nan_weights_match_jax_flat():
    x = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0]], dtype=np.float64)
    y = np.array([[2.0, 3.0, 4.0], [1.0, np.nan, 5.0]], dtype=np.float64)
    w_all_nan = np.full_like(x, np.nan)
    w_some_nan = np.array([[1.0, np.nan, 1.0], [0.5, 1.0, np.nan]], dtype=np.float64)

    for weights in (w_all_nan, w_some_nan):
        data = {"x": x, "y": y, "w": weights}
        _assert_cpp_matches_pure_jax("get_beta(Ridge(x, y, w, 0.0, 0.1))", data)
        _assert_cpp_matches_pure_jax("get_preds(Ridge(x, y, w, 0.0, 0.1))", data)


def test_cpp_flat_rbf_and_instrument_basis_mean_match_jax_flat():
    rows = 18
    cols = 4
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    ev_ts = row + 0.25 * col
    session_start = np.zeros((rows, cols), dtype=np.float64)
    session_end = np.full((rows, cols), 12.0, dtype=np.float64)
    volume = 10.0 + 0.5 * row + col
    ev_ts[1, 2] = -1.0
    ev_ts[14:, :] = 13.0
    volume[5, 1] = np.nan
    data = {
        "ev_ts": ev_ts,
        "session_start": session_start,
        "session_end": session_end,
        "volume": volume,
    }
    for formula in (
        "rbf_basis(ev_ts, session_start, session_end, 4)",
        "future_rbf_basis_sum(ev_ts, session_start, session_end, 4, 8)",
        "get_preds(InstrumentBasisMean(rbf_basis(ev_ts, session_start, session_end, 3), volume, 1.0, 4.0))",
        'einsum(get_beta(InstrumentBasisMean(rbf_basis(ev_ts, session_start, session_end, 3), volume, 1.0, 4.0)), future_rbf_basis_sum(ev_ts, session_start, session_end, 3, 8), "ij,ij->i")',
    ):
        _assert_cpp_matches_jax(formula, data, rtol=1e-9, atol=1e-9)

def test_cpp_flat_groupby_nested_rhs_matches_jax_flat():
    rows = 18
    cols = 5
    close = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) * 0.1
    key0 = np.mod(np.arange(rows, dtype=np.float64)[:, None] + np.arange(cols, dtype=np.float64)[None, :], 3.0)
    key1 = np.mod(np.arange(rows, dtype=np.float64)[:, None], 2.0) + np.zeros((rows, cols), dtype=np.float64)
    key0[4, 2] = np.nan
    close[7, 3] = np.nan
    data = {"close": close, "key0": key0, "key1": key1}
    _assert_cpp_matches_jax("groupby((key0, key1), close, cumsum(self_))", data)
    _assert_cpp_matches_jax("groupby((key0, key1), close, cumsum(cumsum(self_)))", data)
    _assert_cpp_matches_jax("groupby((key0, key1), close, add(cumsum(self_), 1.0))", data)
    _assert_cpp_matches_jax(
        "groupby((univ([0, 1], [2, 3, 4]), key0), close, cumsum(cumsum(self_)))",
        data,
    )


def test_cpp_flat_groupby_hash_index_preserves_nan_signed_zero_and_key_churn():
    rows, cols = 48, 6
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    close = row * 0.1 + col
    key0 = np.mod(row + 7.0 * col, 31.0)
    key1 = np.mod(3.0 * row + col, 11.0)
    key0[2, 0] = np.nan
    key0[3, 0] = np.nan
    key1[5, 1] = -0.0
    key1[6, 1] = 0.0
    _assert_cpp_matches_pure_jax(
        "groupby((key0, key1), close, add(cumsum(self_), 1.0))",
        {"close": close, "key0": key0, "key1": key1},
    )


def test_cpp_flat_tick_into_reuses_output_buffer():
    runtime = compile_formula_native("add(close, open)")
    state = runtime.init_state(3)
    out = np.empty(3, dtype=np.float64)
    close = np.array([1.0, 2.0, 3.0])
    open_ = np.array([10.0, 20.0, 30.0])
    runtime.tick_into(state, out, close, open_)
    np.testing.assert_allclose(out, [11.0, 22.0, 33.0])
    out_id = id(out)
    runtime.tick_into(state, out, close + 1.0, open_)
    assert id(out) == out_id
    np.testing.assert_allclose(out, [12.0, 23.0, 34.0])


def test_cpp_flat_tick_into_keeps_force_cast_input_alive_during_native_compute():
    runtime = compile_formula_native("add(close, open)")
    state = runtime.init_state(3)
    out = np.empty(3, dtype=np.float64)
    runtime.tick_into(
        state,
        out,
        np.array([1.0, 2.0, 3.0], dtype=np.float32),
        np.array([10, 20, 30], dtype=np.int32),
    )
    np.testing.assert_allclose(out, [11.0, 22.0, 33.0])


def test_cpp_flat_typed_plan_exposes_resolved_liveness_and_state():
    runtime = compile_formula_native("add(cumsum(close), mul(open, 2.0))")
    plan = runtime.native_plan
    diagnostic = runtime.inspect_native_plan()

    assert plan.output_id == len(plan.nodes) - 1
    assert diagnostic["version"] == 1
    assert diagnostic["dtype"] == "float64"
    assert diagnostic["nodes"][plan.output_id]["liveness"][1] == len(plan.nodes)
    cumsum_node = next(node for node in plan.nodes if node.opcode == "cumsum")
    assert cumsum_node.stateful and not cumsum_node.pure
    assert cumsum_node.state_index >= 0
    assert cumsum_node.value_type.broadcast is BroadcastMode.ELEMENTWISE
    assert all(node.live_until >= node.live_from for node in plan.nodes)


def test_cpp_flat_typed_plan_rejects_unknown_dtype_without_touching_runtime():
    runtime = compile_formula("add(close, 1.0)", cpp=False)
    with pytest.raises(ValueError, match="unsupported native dtype"):
        lower_native_plan(runtime.program, dtype="complex128")


def test_cpp_flat_plan_optimization_folds_literals_cses_and_preserves_ticks():
    formula = "add(add(close, mul(2.0, 3.0)), add(close, mul(2.0, 3.0)))"
    jax_runtime = compile_formula(formula, cpp=False)
    optimized, _ = lower_native_plan(jax_runtime.program)
    reference, _ = lower_native_plan(jax_runtime.program, optimize=False)

    assert len(optimized.nodes) < len(reference.nodes)
    assert dict(optimized.optimizations)["constant_folds"] >= 1
    assert dict(optimized.optimizations)["dead_nodes"] >= 1

    native = compile_formula_native(formula)
    state = native.init_state(4)
    jax_state = jax_runtime.init_state(4)
    rows = (
        np.array([1.0, np.nan, np.inf, -np.inf]),
        np.array([-2.0, 0.0, 5.0, np.nan]),
    )
    for row in rows:
        native_out = native.tick(state, row)
        jax_state, jax_out = jax_runtime.tick(jax_state, row)
        np.testing.assert_allclose(native_out, np.asarray(jax_out), equal_nan=True)


def test_cpp_flat_hybrid_partition_diagnostic_reports_cost_inputs():
    from trading_dsl_engine.base.dsl import cumsum, var
    from trading_dsl_engine.jax_flat import stateless

    unsupported = stateless(lambda x: x + 1, name="diagnostic_only")
    runtime = compile_formula(unsupported(cumsum(var("close"))) + 2.0, cpp=False)
    diagnostic = inspect_hybrid_partition(runtime.program, 1024, 150)
    assert diagnostic["version"] == 1
    candidate = diagnostic["candidates"][0]
    assert candidate["estimated_work"] > 0
    assert candidate["frontier_bytes"] == 1024 * 150 * 8
    assert candidate["conversion_copy"] is True
    assert isinstance(candidate["accelerate"], bool)


def test_cpp_flat_micro_runtime_comparison_smoke(capsys):
    rows = 256
    cols = 9
    rng = np.random.default_rng(123)
    data = {
        "close": rng.normal(size=(rows, cols)),
        "open": rng.normal(size=(rows, cols)),
    }
    formula = "xstd(add(abs(close), div(exp(fraction(open)), add(abs(close), 1.0))))"
    cpp_runtime = compile_formula_native(formula)
    jax_runtime = compile_formula(formula)

    # Warm both runtimes; benchmark timings below intentionally exclude compile/setup time.
    cpp_runtime.run_batch(data)
    jax_runtime.run_batch(data)
    jax.block_until_ready(jax_runtime.run_batch(data)[1])

    t0 = time.perf_counter()
    _, cpp_out = cpp_runtime.run_batch(data)
    cpp_elapsed = time.perf_counter() - t0

    t0 = time.perf_counter()
    _, jax_out = jax_runtime.run_batch(data)
    jax.block_until_ready(jax_out)
    jax_elapsed = time.perf_counter() - t0

    print(f"cpp_flat_smoke cpp_tick_s={cpp_elapsed:.6f} jax_flat_batch_s={jax_elapsed:.6f}")
    np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=1e-10, atol=1e-10, equal_nan=True)
    captured = capsys.readouterr()
    assert "cpp_flat_smoke" in captured.out


def test_compile_formula_cpp_flag_and_unsupported_groupby_fallback_warning():
    rows = 8
    cols = 3
    close = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    key = np.mod(close, 2.0)
    data = {"close": close, "key": key}

    assert compile_formula("add(close, 1.0)").cpp is True
    assert compile_formula("add(close, 1.0)", cpp=False).cpp is False

    formula = "groupby((key,), close, outer(self_))"
    runtime_cpp = compile_formula(formula)
    runtime_jax = compile_formula(formula, cpp=False)
    with pytest.warns(RuntimeWarning, match="unsupported.*outer"):
        _, fallback_out = runtime_cpp.run_batch(data)
    _, jax_out = runtime_jax.run_batch(data)
    np.testing.assert_allclose(np.asarray(fallback_out), np.asarray(jax_out), rtol=1e-10, atol=1e-10, equal_nan=True)


def test_cpp_flat_outer_and_einsum_subset_match_jax_flat():
    rows = 10
    cols = 4
    rng = np.random.default_rng(321)
    close = rng.normal(size=(rows, cols))
    open_ = rng.normal(size=(rows, cols))
    high = rng.normal(size=(rows, cols))
    close[2, 1] = np.nan
    open_[4, 2] = np.nan
    data = {"close": close, "open": open_, "high": high}
    for formula in (
        "outer(close)",
        'einsum(close, open, "i,i->i")',
        'einsum(close, open, high, "i,i,i->i")',
        'einsum(close, bspline(fillna(open, 0.25), 3), "i,ij->i")',
        'einsum(bspline(fillna(close, 0.25), 3), bspline(fillna(open, 0.5), 3), "ij,ij->ij")',
        'einsum(bspline(fillna(close, 0.25), 3), bspline(fillna(open, 0.5), 3), "ij,ij->")',
        'einsum(bspline(fillna(close, 0.25), 2), bspline(fillna(open, 0.5), 3), "ij,ik->jk")',
    ):
        _assert_cpp_matches_jax(formula, data, rtol=1e-9, atol=1e-9)


def test_cpp_flat_roll_mean_and_ewm_min_periods_match_jax_flat():
    rows = 48
    cols = 5
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    close = 0.5 * row - col
    close[0, 0] = 0.0
    close[1, 2] = np.nan
    close[4, 0] = np.nan
    close[9, 3] = np.nan
    data = {"close": close}
    _assert_cpp_matches_pure_jax("add(roll_mean(close, 5, 3), ewm(close, 4.0, 3.0))", data)
    _assert_cpp_matches_pure_jax("ewm(close, 4.0, 1.0)", data)


def test_cpp_flat_groupby_inner_ewm_min_periods_matches_jax_flat():
    rows = 36
    cols = 4
    row = np.arange(rows, dtype=np.float64)[:, None]
    col = np.arange(cols, dtype=np.float64)[None, :]
    close = row * 0.1 - col
    close[0, 0] = 0.0
    key = np.mod(row + col, 3.0)
    close[2, 1] = np.nan
    close[5, 3] = np.nan
    data = {"close": close, "key": key}
    _assert_cpp_matches_pure_jax("groupby((key,), close, ewm(self_, 3.0, 2.0))", data)


def test_cpp_hybrid_batch_runs_supported_subgraph_before_jax_lambda():
    from trading_dsl_engine.base.dsl import cumsum, groupby, self_, var
    from trading_dsl_engine.jax_flat import stateless

    rows = 20
    cols = 4
    close = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) * 0.1
    key = np.mod(np.arange(rows, dtype=np.float64)[:, None] + np.arange(cols, dtype=np.float64)[None, :], 3.0)
    close[3, 2] = np.nan
    data = {"close": close, "key": key}

    plus_one = stateless(lambda x: x + 1.0, name="plus_one")
    formula = plus_one(groupby((var("key"),), var("close"), cumsum(self_)))
    runtime_cpp = compile_formula(formula)
    runtime_jax = compile_formula(formula, cpp=False)

    hybrid_state, hybrid_out = runtime_cpp.run_batch(data)
    _, jax_out = runtime_jax.run_batch(data)

    np.testing.assert_allclose(np.asarray(hybrid_out), np.asarray(jax_out), rtol=1e-10, atol=1e-10, equal_nan=True)
    assert any(name.startswith("__cpp_subgraph_") for name in runtime_cpp.program.input_names) is False


def test_cpp_staged_subprogram_compacts_frontier_only_input_indices():
    from trading_dsl_engine.jax_flat.engine import DagNode, StateFieldRef, StateLayout, StreamingProgram
    from trading_dsl_engine.jax_flat.engine_cpp import _cpp_node_specs, _subprogram_for_node_with_frontier
    from trading_dsl_engine.jax_flat.ops import InputOp, LiteralOp, NaryOp

    jax_only = NaryOp(lambda x: x + 1.0)
    root = NaryOp(lambda x, y: x + y, cpp_name="add")
    program = StreamingProgram(
        nodes=(
            DagNode(InputOp(0), ()),
            DagNode(jax_only, (0,)),
            DagNode(LiteralOp(2.0), ()),
            DagNode(root, (1, 2)),
        ),
        outputs=(3,),
        input_names=("close",),
        state_layout=StateLayout((StateFieldRef(-1),) * 4, 0),
        metadata=None,
        cache_nodes=(),
    )

    subprogram, frontier_ids, input_sources = _subprogram_for_node_with_frontier(program, 3)
    node_specs, _, _ = _cpp_node_specs(subprogram)

    assert frontier_ids == (1,)
    assert input_sources == (("frontier", 1),)
    assert node_specs[0][0] == "input"
    assert node_specs[0][2] == 0


def test_cpp_hybrid_batch_can_stage_cpp_before_and_after_jax_lambda():
    from trading_dsl_engine.base.dsl import cumsum, ewm, groupby, self_, var
    from trading_dsl_engine.jax_flat import stateless

    rows = 24
    cols = 4
    close = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) * 0.05
    key = np.mod(np.arange(rows, dtype=np.float64)[:, None] + np.arange(cols, dtype=np.float64)[None, :], 3.0)
    close[4, 1] = np.nan
    data = {"close": close, "key": key}

    plus_one = stateless(lambda x: x + 1.0, name="plus_one")
    grouped = groupby((var("key"),), var("close"), cumsum(self_))
    formula = plus_one(grouped) * (ewm(var("close"), 5.0) + 3.0)
    runtime_cpp = compile_formula(formula)
    runtime_jax = compile_formula(formula, cpp=False)

    hybrid_state, hybrid_out = runtime_cpp.run_batch(data)
    _, jax_out = runtime_jax.run_batch(data)

    np.testing.assert_allclose(np.asarray(hybrid_out), np.asarray(jax_out), rtol=1e-10, atol=1e-10, equal_nan=True)
    assert type(hybrid_state).__name__ == "CppFlatState"


def test_cpp_hybrid_batch_handles_multiple_jax_frontiers_and_native_islands():
    from trading_dsl_engine.base.dsl import cumsum, ewm, groupby, self_, var
    from trading_dsl_engine.jax_flat import stateless

    rows = 22
    cols = 4
    close = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols) * 0.07
    key = np.mod(np.arange(rows, dtype=np.float64)[:, None] + np.arange(cols, dtype=np.float64)[None, :], 4.0)
    close[5, 2] = np.nan
    data = {"close": close, "key": key}

    plus_one = stateless(lambda x: x + 1.0, name="plus_one")
    minus_one = stateless(lambda x: x - 1.0, name="minus_one")
    grouped = groupby((var("key"),), var("close"), cumsum(self_))
    smoothed = ewm(var("close"), 5.0)
    formula = (plus_one(grouped) + minus_one(smoothed)) * (ewm(var("close"), 3.0) + 3.0)
    runtime_cpp = compile_formula(formula)
    runtime_jax = compile_formula(formula, cpp=False)

    hybrid_state, hybrid_out = runtime_cpp.run_batch(data)
    _, jax_out = runtime_jax.run_batch(data)

    np.testing.assert_allclose(np.asarray(hybrid_out), np.asarray(jax_out), rtol=1e-10, atol=1e-10, equal_nan=True)
    assert type(hybrid_state).__name__ == "CppFlatState"


def test_cpp_hybrid_batch_accepts_memmap_inputs(tmp_path):
    rows = 18
    cols = 4
    close_path = tmp_path / "close.dat"
    key_path = tmp_path / "key.dat"
    close = np.memmap(close_path, mode="w+", dtype=np.float64, shape=(rows, cols))
    key = np.memmap(key_path, mode="w+", dtype=np.float64, shape=(rows, cols))
    base = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    close[:] = base * 0.1
    key[:] = np.mod(base, 3.0)
    close.flush()
    key.flush()

    data = {"close": close, "key": key}
    formula = "groupby((key,), close, cumsum(self_))"
    runtime_cpp = compile_formula(formula)
    runtime_jax = compile_formula(formula, cpp=False)

    state, cpp_out = runtime_cpp.run_batch(data)
    _, jax_out = runtime_jax.run_batch({"close": np.asarray(close), "key": np.asarray(key)})

    np.testing.assert_allclose(np.asarray(cpp_out), np.asarray(jax_out), rtol=1e-10, atol=1e-10, equal_nan=True)
    assert type(state).__name__ == "CppFlatState"
