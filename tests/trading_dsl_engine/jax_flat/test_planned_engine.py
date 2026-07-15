import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import add, cumsum, ewm, shift, var
from trading_dsl_engine.jax_flat import compile_features, compile_formula
from trading_dsl_engine.jax_flat import engine
from trading_dsl_engine.jax_flat import engine_legacy

jax.config.update("jax_enable_x64", True)


def _legacy_batch(expr, values):
    runtime = engine_legacy.compile_formula(expr, cpp=False)
    inputs = (jnp.asarray(values, dtype=jnp.float64),)
    state, output, _ = engine_legacy._jit_batch_from_initial_state(runtime, inputs)
    jax.block_until_ready((state, output))
    return state, output


def test_compile_formula_preserves_single_output_behavior(monkeypatch):
    monkeypatch.setenv("TRADING_DSL_JAX_FLAT_BATCH_CHUNK_SIZE", "64")
    rng = np.random.default_rng(0)
    values = rng.normal(size=(257, 4))
    values[rng.random(values.shape) < 0.12] = np.nan
    expr = ewm(ewm(var("x"), 7.0, ignore_na=True, adjust=False), 19.0, ignore_na=True, adjust=False)

    expected_state, expected = _legacy_batch(expr, values)
    runtime = compile_formula(expr, cpp=False)
    actual_state, actual = runtime.run_batch({"x": jnp.asarray(values)})

    assert isinstance(actual, jax.Array)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=True)
    jax.tree_util.tree_map(
        lambda left, right: np.testing.assert_allclose(left, right, rtol=1e-11, atol=1e-11, equal_nan=True),
        actual_state,
        expected_state,
    )


def test_named_outputs_share_one_dag_and_return_dict(monkeypatch):
    monkeypatch.setenv("TRADING_DSL_JAX_FLAT_BATCH_CHUNK_SIZE", "96")
    rng = np.random.default_rng(1)
    values = rng.normal(size=(300, 5))
    values[rng.random(values.shape) < 0.08] = np.nan
    base = ewm(var("x"), 11.0, ignore_na=True, adjust=False)
    formulas = {
        "fast": ewm(base, 17.0, ignore_na=True, adjust=False),
        "slow": ewm(base, 43.0, ignore_na=True, adjust=False),
        "cum": cumsum(add(var("x"), 1.0)),
    }

    runtime = compile_features(formulas)
    _, outputs = runtime.run_batch({"x": jnp.asarray(values)})

    assert list(outputs) == list(formulas)
    separate_node_count = sum(
        len(engine_legacy.compile_formula(expr, cpp=False).program.nodes)
        for expr in formulas.values()
    )
    assert len(runtime.program.nodes) < separate_node_count
    for name, expr in formulas.items():
        _, expected = _legacy_batch(expr, values)
        np.testing.assert_allclose(outputs[name], expected, rtol=1e-11, atol=1e-11, equal_nan=True)


def test_masked_fixed_tail_preserves_shift_state_across_calls(monkeypatch):
    monkeypatch.setenv("TRADING_DSL_JAX_FLAT_BATCH_CHUNK_SIZE", "64")
    values = np.arange(3 * 173, dtype=np.float64).reshape(173, 3)
    expr = shift(var("x"), 5)

    runtime = compile_formula(expr, cpp=False)
    state = runtime.init_state(3)
    state, first = runtime.run_batch({"x": jnp.asarray(values[:91])}, states=state)
    state, second = runtime.run_batch({"x": jnp.asarray(values[91:])}, states=state)

    _, expected = _legacy_batch(expr, values)
    actual = jnp.concatenate((first, second), axis=0)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0, equal_nan=True)


def test_execution_plan_classifies_and_limits_regions():
    x = var("x")
    runtime = compile_features(
        {
            "a": ewm(ewm(x, 5.0, ignore_na=True, adjust=False), 17.0, ignore_na=True, adjust=False),
            "b": ewm(ewm(x, 7.0, ignore_na=True, adjust=False), 29.0, ignore_na=True, adjust=False),
        }
    )
    plan = engine.build_execution_plan(runtime.program)
    assert plan.strategy == "ewm_branch_batch"
    assert plan.chunk_size in {4_096, 8_192}
    assert any(region.kind is engine.ExecutionKind.AFFINE for region in plan.regions)
    assert all(len(region.node_ids) >= 1 for region in plan.regions)


def test_associative_affine_ewm_matches_scan(monkeypatch):
    monkeypatch.setenv("TRADING_DSL_JAX_FLAT_ASSOCIATIVE_EWM_MIN_WIDTH", "1")
    rng = np.random.default_rng(4)
    values = rng.normal(size=(129, 4))
    values[rng.random(values.shape) < 0.15] = np.nan
    expr = ewm(var("x"), 13.0, min_periods=3, ignore_na=True, adjust=False)

    _, expected = _legacy_batch(expr, values)
    runtime = compile_formula(expr, cpp=False)
    _, actual = runtime.run_batch({"x": jnp.asarray(values)})
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-10, equal_nan=True)


def test_planned_stateful_runtime_does_not_regress_materially(monkeypatch):
    monkeypatch.setenv("TRADING_DSL_JAX_FLAT_BATCH_CHUNK_SIZE", "4096")
    rows, assets = 262_144, 9
    values = jax.random.normal(jax.random.PRNGKey(9), (rows, assets), dtype=jnp.float64)
    expr = var("x")
    for span in (5.0, 11.0, 23.0, 47.0, 83.0):
        expr = ewm(expr, span, ignore_na=True, adjust=False)

    legacy_runtime = engine_legacy.compile_formula(expr, cpp=False)
    planned_runtime = compile_formula(expr, cpp=False)
    inputs = (values,)

    def legacy_call():
        result = engine_legacy._jit_batch_from_initial_state(legacy_runtime, inputs)
        jax.block_until_ready(result)

    def planned_call():
        result = planned_runtime.run_batch({"x": values})
        jax.block_until_ready(result)

    legacy_call()
    planned_call()
    legacy_times = []
    planned_times = []
    for _ in range(3):
        start = time.perf_counter()
        legacy_call()
        legacy_times.append(time.perf_counter() - start)
        start = time.perf_counter()
        planned_call()
        planned_times.append(time.perf_counter() - start)

    legacy_median = statistics.median(legacy_times)
    planned_median = statistics.median(planned_times)
    assert planned_median <= legacy_median * 1.20, (legacy_times, planned_times)
