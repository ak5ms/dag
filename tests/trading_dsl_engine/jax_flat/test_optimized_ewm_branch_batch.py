import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import ewm, var
from trading_dsl_engine.jax_flat import compile_features, compile_formula
from trading_dsl_engine.jax_flat import optimized_planner

jax.config.update("jax_enable_x64", True)


def _formulas():
    x = var("x")
    shared = ewm(x, 7.0, min_periods=2, ignore_na=True, adjust=False)
    return {
        "fast": ewm(ewm(shared, 11.0, ignore_na=True, adjust=False), 19.0, ignore_na=True, adjust=False),
        "medium": ewm(ewm(shared, 13.0, ignore_na=True, adjust=False), 23.0, ignore_na=True, adjust=False),
        "slow": ewm(ewm(shared, 17.0, ignore_na=True, adjust=False), 31.0, ignore_na=True, adjust=False),
        "slower": ewm(ewm(shared, 21.0, ignore_na=True, adjust=False), 43.0, ignore_na=True, adjust=False),
    }


def _input(rows=513, assets=5):
    rng = np.random.default_rng(123)
    values = rng.normal(size=(rows, assets))
    values[rng.random(values.shape) < 0.17] = np.nan
    return values


def test_ewm_branches_are_detected_and_match_independent_runtimes():
    formulas = _formulas()
    values = _input()
    runtime = compile_features(formulas, max_in_flight=2)

    assert runtime.execution_strategy() == "ewm_branch_batch"
    plan = optimized_planner._detect_ewm_branch_plan(runtime.program)
    assert plan is not None
    assert plan.breadth == 4
    assert plan.depth == 2

    _, actual = runtime.run_batch({"x": values}, out_path=None)
    for name, formula in formulas.items():
        reference = compile_formula(formula, cpp=False)
        _, expected = reference.run_batch({"x": jnp.asarray(values)})
        np.testing.assert_allclose(actual[name], np.asarray(expected), rtol=1e-11, atol=1e-11, equal_nan=True)


def test_ewm_branch_state_continuation_matches_one_shot():
    formulas = _formulas()
    values = _input(rows=701, assets=4)
    runtime = compile_features(formulas, max_in_flight=2)

    split = 317
    state = runtime.init_state(values.shape[1])
    state, first = runtime.run_batch({"x": values[:split]}, states=state, out_path=None)
    state, second = runtime.run_batch({"x": values[split:]}, states=state, out_path=None)

    full_runtime = compile_features(formulas, max_in_flight=2)
    _, full = full_runtime.run_batch({"x": values}, out_path=None)
    for name in formulas:
        combined = np.concatenate([first[name], second[name]], axis=0)
        np.testing.assert_allclose(combined, full[name], rtol=1e-11, atol=1e-11, equal_nan=True)


def test_cpu_chunk_planner_uses_cache_sized_tiles():
    branch_runtime = compile_features(_formulas())
    assert optimized_planner._cpu_chunk_size(branch_runtime.program) == 8_192

    single = compile_features({"one": ewm(var("x"), 11.0, ignore_na=True, adjust=False)})
    assert optimized_planner._cpu_chunk_size(single.program) == 4_096
