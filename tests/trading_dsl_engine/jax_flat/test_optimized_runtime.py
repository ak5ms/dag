import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import add, cumsum, ewm, var
from trading_dsl_engine.jax_flat import compile_formula as compile_current
from trading_dsl_engine.jax_flat.optimized import compile_features, compile_formula

jax.config.update("jax_enable_x64", True)


def _current(expr, data):
    runtime = compile_current(expr, cpp=False)
    _, out = runtime.run_batch(data)
    return np.asarray(jax.block_until_ready(out))


def test_nested_stateful_compound_scan_matches_current_with_tail_padding():
    rng = np.random.default_rng(42)
    x = rng.normal(size=(257, 4))
    x[rng.random(x.shape) < 0.15] = np.nan
    expr = ewm(ewm(var("x"), 7.0, ignore_na=True, adjust=False), 13.0, ignore_na=True, adjust=False)

    expected = _current(expr, {"x": jnp.asarray(x)})
    runtime = compile_formula(expr, chunk_size=64, max_in_flight=3)
    assert runtime.execution_strategy() == "compound"
    _, actual = runtime.run_batch({"x": x}, out_path=None)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=True)


def test_associative_ewm_matches_current_and_uses_node_batch():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(513, 5))
    x[rng.random(x.shape) < 0.2] = np.nan
    expr = ewm(var("x"), 21.0, min_periods=3, ignore_na=True, adjust=False)

    expected = _current(expr, {"x": jnp.asarray(x)})
    runtime = compile_formula(expr, chunk_size=128)
    assert runtime.execution_strategy() == "node_batch"
    _, actual = runtime.run_batch({"x": x}, out_path=None)

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=True)


def test_multi_root_compilation_shares_dag_and_matches_independent_runs():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 6))
    x[rng.random(x.shape) < 0.1] = np.nan
    base = ewm(var("x"), 11.0, ignore_na=True, adjust=False)
    formulas = {
        "fast": base,
        "nested": ewm(base, 31.0, ignore_na=True, adjust=False),
        "cum": cumsum(add(var("x"), 1.0)),
    }

    runtime = compile_features(formulas, chunk_size=96, max_in_flight=2)
    _, actual = runtime.run_batch({"x": x}, out_path=None)
    assert set(actual) == set(formulas)

    for name, expr in formulas.items():
        expected = _current(expr, {"x": jnp.asarray(x)})
        np.testing.assert_allclose(actual[name], expected, rtol=1e-11, atol=1e-11, equal_nan=True)

    assert len(runtime.program.nodes) < sum(len(compile_current(expr, cpp=False).program.nodes) for expr in formulas.values())


def test_state_is_consumed_and_returned_under_donation():
    x0 = np.arange(60.0).reshape(20, 3)
    x1 = np.arange(60.0, 120.0).reshape(20, 3)
    expr = ewm(var("x"), 9.0, ignore_na=True, adjust=False)
    runtime = compile_formula(expr, chunk_size=16)

    state = runtime.init_state(3)
    state, out0 = runtime.run_batch({"x": x0}, states=state, out_path=None)
    state, out1 = runtime.run_batch({"x": x1}, states=state, out_path=None)

    full_expected = _current(expr, {"x": jnp.asarray(np.concatenate([x0, x1], axis=0))})
    np.testing.assert_allclose(out0, full_expected[:20], rtol=1e-11, atol=1e-11, equal_nan=True)
    np.testing.assert_allclose(out1, full_expected[20:], rtol=1e-11, atol=1e-11, equal_nan=True)
