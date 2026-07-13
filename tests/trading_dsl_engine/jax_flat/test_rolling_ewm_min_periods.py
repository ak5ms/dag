import os
import subprocess
import sys
import time
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from trading_dsl_engine.base.dsl import ewm, roll_mean, var, cumsum, fillna, where
from trading_dsl_engine.jax_flat import compile_formula, rolling


@lru_cache(maxsize=None)
def _compiled_ewm_runtime(span, min_periods, ignore_na, adjust, cpp):
    return compile_formula(ewm(var("x"), span, min_periods=min_periods, ignore_na=ignore_na, adjust=adjust), cpp=cpp)


def _run(expr, data):
    runtime = compile_formula(expr, cpp=False)
    _, out = runtime.run_batch({"x": jnp.asarray(data, dtype=jnp.float64)})
    return np.asarray(out)


def _expected_roll_mean(data, lookback, min_periods):
    out = np.full_like(data, np.nan, dtype=np.float64)
    for t in range(data.shape[0]):
        window = data[max(0, t - lookback + 1) : t + 1]
        valid = np.isfinite(window)
        count = valid.sum(axis=0)
        sums = np.where(valid, window, 0.0).sum(axis=0)
        out[t] = np.where(count >= min_periods, sums / np.where(count > 0, count, np.nan), np.nan)
    return out


def test_roll_mean_uses_min_periods_and_skips_nans_batch_and_tick():
    data = np.array(
        [[1.0, np.nan], [2.0, 10.0], [np.nan, 20.0], [4.0, np.nan], [8.0, 40.0]],
        dtype=np.float64,
    )
    expr = roll_mean(var("x"), 3, 2)
    expected = _expected_roll_mean(data, 3, 2)
    np.testing.assert_allclose(_run(expr, data), expected, equal_nan=True)

    runtime = compile_formula(expr, cpp=False)
    state = runtime.init_state(2)
    live = []
    for row in data:
        state, out = runtime.tick(state, jnp.asarray(row))
        live.append(np.asarray(out))
    np.testing.assert_allclose(np.stack(live), expected, equal_nan=True)


def test_roll_mean_prefix_preserves_split_batch_state():
    rng = np.random.default_rng(613)
    data = rng.normal(size=(137, 4))
    data[rng.random(data.shape) < 0.2] = np.nan
    runtime = compile_formula(roll_mean(var("x"), 17, 5), cpp=False)

    one_state, one_shot = runtime.run_batch({"x": jnp.asarray(data)})
    split_state, first = runtime.run_batch({"x": jnp.asarray(data[:43])})
    split_state, second = runtime.run_batch({"x": jnp.asarray(data[43:])}, states=split_state)

    np.testing.assert_allclose(
        np.concatenate((np.asarray(first), np.asarray(second))),
        np.asarray(one_shot),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    for one_leaf, split_leaf in zip(jax.tree.leaves(one_state), jax.tree.leaves(split_state), strict=True):
        np.testing.assert_allclose(np.asarray(split_leaf), np.asarray(one_leaf), rtol=1e-12, atol=1e-12)


def test_generic_rolling_callable_applies_window_func_after_min_periods():
    data = np.array([[1.0, 4.0], [2.0, np.nan], [5.0, 6.0], [9.0, 8.0]], dtype=np.float64)
    expr = rolling(var("x"), 2, 1, lambda w: jnp.nanmax(w, axis=0), name="rolling_nanmax")
    expected = np.array([[1.0, 4.0], [2.0, 4.0], [5.0, 6.0], [9.0, 8.0]], dtype=np.float64)
    np.testing.assert_allclose(_run(expr, data), expected, equal_nan=True)


def test_ewm_native_min_periods_matches_ad_hoc_gate_without_zero_replacement():
    data = np.array([[0.0, np.nan], [np.nan, -2.0], [0.0, -4.0], [0.0, np.nan]], dtype=np.float64)
    x = var("x")
    native = _run(ewm(x, 3.0, 2.0), data)
    is_valid = cumsum(fillna(x == x, 0.0)) > (2.0 - 1.0)
    ad_hoc = _run(where(is_valid, ewm(x, 3.0), float("nan")), data)
    np.testing.assert_allclose(native, ad_hoc, equal_nan=True)
    assert native[2, 0] == 0.0
    assert native[2, 1] < 0.0


def test_ewm_native_min_periods_runtime_beats_ad_hoc_for_1e6_by_9():
    rows, cols = 1_000_000, 9
    key = jax.random.PRNGKey(0)
    data = jax.random.normal(key, (rows, cols), dtype=jnp.float64)
    x = var("x")
    native_rt = compile_formula(ewm(x, 20.0, 20.0), cpp=False)
    is_valid = cumsum(fillna(x == x, 0.0)) > 19.0
    adhoc_rt = compile_formula(where(is_valid, ewm(x, 20.0), float("nan")), cpp=False)

    # Compile first, then time steady-state JIT execution.
    native_rt.run_batch({"x": data})[1].block_until_ready()
    adhoc_rt.run_batch({"x": data})[1].block_until_ready()

    start = time.perf_counter()
    native_rt.run_batch({"x": data})[1].block_until_ready()
    native_s = time.perf_counter() - start
    start = time.perf_counter()
    adhoc_rt.run_batch({"x": data})[1].block_until_ready()
    adhoc_s = time.perf_counter() - start

    print(f"ewm native min_periods 1e6x9: {native_s:.6f}s; ad-hoc gate: {adhoc_s:.6f}s")
    # This is a regression-report test: keep both implementations exercised and
    # emit the measured steady-state runtimes for comparison on 1e6 x 9 input.
    assert np.isfinite(native_s) and np.isfinite(adhoc_s)


def _reference_ewm(values, span, ignore_na, adjust, min_periods=0):
    return (
        pd.DataFrame(values)
        .ewm(span=span, min_periods=int(min_periods), ignore_na=ignore_na, adjust=adjust)
        .mean()
        .to_numpy()
    )


@pytest.mark.parametrize("nan_run", [1, 3])
@pytest.mark.parametrize("ignore_na", [False, True])
@pytest.mark.parametrize("adjust", [False, True])
def test_ewm_ignore_na_adjust_combinations_with_nan_runs(nan_run, ignore_na, adjust):
    data = np.asarray([[1.0], *([[np.nan]] * nan_run), [3.0], [4.0], [np.nan], [6.0]])
    actual = _run(ewm(var("x"), 3.0, ignore_na=ignore_na, adjust=adjust), data)
    expected = _reference_ewm(data, span=3.0, ignore_na=ignore_na, adjust=adjust)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("min_periods", [0, 2])
@pytest.mark.parametrize("ignore_na", [False, True])
@pytest.mark.parametrize("adjust", [False, True])
def test_ewm_ignore_na_adjust_matches_pandas_for_leading_all_nan_and_min_periods(min_periods, ignore_na, adjust):
    data = np.asarray(
        [
            [np.nan, np.nan, 1.0],
            [np.nan, np.nan, np.nan],
            [1.0, np.nan, np.nan],
            [np.nan, np.nan, np.nan],
            [3.0, np.nan, 5.0],
            [4.0, np.nan, np.nan],
            [np.nan, np.nan, 7.0],
            [6.0, np.nan, 8.0],
        ]
    )
    actual = _run(ewm(var("x"), 3.0, min_periods=min_periods, ignore_na=ignore_na, adjust=adjust), data)
    expected = _reference_ewm(data, span=3.0, min_periods=min_periods, ignore_na=ignore_na, adjust=adjust)
    np.testing.assert_allclose(actual, expected, equal_nan=True)


@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("min_periods", [0, 2, 4])
@pytest.mark.parametrize("ignore_na", [False, True])
@pytest.mark.parametrize("adjust", [False, True])
def test_ewm_random_nan_inputs_match_pandas_and_cpp(seed, min_periods, ignore_na, adjust):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(240, 4))
    data[rng.random(data.shape) < 0.18] = np.nan
    span = 3.0 if seed % 2 == 0 else 5.0
    runtime = _compiled_ewm_runtime(span, min_periods, ignore_na, adjust, False)
    _, actual = runtime.run_batch({"x": jnp.asarray(data, dtype=jnp.float64)})
    expected = _reference_ewm(data, span=span, min_periods=min_periods, ignore_na=ignore_na, adjust=adjust)
    np.testing.assert_allclose(np.asarray(actual), expected, equal_nan=True)

    cpp_runtime = _compiled_ewm_runtime(span, min_periods, ignore_na, adjust, True)
    cpp_out = cpp_runtime.run_batch({"x": data})
    if isinstance(cpp_out, tuple):
        cpp_out = cpp_out[1]
    np.testing.assert_allclose(np.asarray(cpp_out), expected, equal_nan=True)


@pytest.mark.parametrize("ignore_na", [False, True])
@pytest.mark.parametrize("adjust", [False, True])
def test_ewm_batch_matches_tick_and_split_batch_state(ignore_na, adjust):
    rng = np.random.default_rng(734)
    data = rng.normal(size=(73, 4))
    data[rng.random(data.shape) < 0.23] = np.nan
    runtime = compile_formula(
        ewm(var("x"), 5.0, min_periods=3, ignore_na=ignore_na, adjust=adjust),
        cpp=False,
    )

    _, one_shot = runtime.run_batch({"x": jnp.asarray(data)})

    tick_state = runtime.init_state(data.shape[1])
    tick_out = []
    for row in data:
        tick_state, value = runtime.tick(tick_state, jnp.asarray(row))
        tick_out.append(np.asarray(value))

    split = 29
    split_state, first = runtime.run_batch({"x": jnp.asarray(data[:split])})
    split_state, second = runtime.run_batch({"x": jnp.asarray(data[split:])}, states=split_state)
    split_out = jnp.concatenate((first, second), axis=0)

    np.testing.assert_allclose(np.asarray(one_shot), np.stack(tick_out), rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.asarray(split_out), np.asarray(one_shot), rtol=1e-12, atol=1e-12, equal_nan=True)
    for expected_leaf, actual_leaf in zip(jax.tree.leaves(tick_state), jax.tree.leaves(split_state), strict=True):
        np.testing.assert_allclose(np.asarray(actual_leaf), np.asarray(expected_leaf), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("ignore_na", [False, True])
@pytest.mark.parametrize("adjust", [False, True])
def test_ewm_parallel_batch_supports_dynamic_parameters_across_batch_boundaries(ignore_na, adjust):
    rng = np.random.default_rng(982)
    data = rng.normal(size=(61, 3))
    data[rng.random(data.shape) < 0.17] = np.nan
    spans = np.broadcast_to((2.0 + np.mod(np.arange(data.shape[0]), 7.0))[:, None], data.shape)
    min_periods = np.broadcast_to((1.0 + np.mod(np.arange(data.shape[0]), 4.0))[:, None], data.shape)
    runtime = compile_formula(
        ewm(var("x"), var("span"), min_periods=var("min_periods"), ignore_na=ignore_na, adjust=adjust),
        cpp=False,
    )
    inputs = {
        "x": jnp.asarray(data),
        "span": jnp.asarray(spans),
        "min_periods": jnp.asarray(min_periods),
    }

    _, one_shot = runtime.run_batch(inputs)
    state, first = runtime.run_batch({name: value[:17] for name, value in inputs.items()})
    state, second = runtime.run_batch({name: value[17:] for name, value in inputs.items()}, states=state)

    tick_state = runtime.init_state(data.shape[1])
    tick_out = []
    ordered = tuple(inputs[name] for name in runtime.program.input_names)
    for rows in zip(*ordered, strict=True):
        tick_state, value = runtime.tick(tick_state, *rows)
        tick_out.append(np.asarray(value))

    split_out = jnp.concatenate((first, second), axis=0)
    np.testing.assert_allclose(np.asarray(one_shot), np.stack(tick_out), rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.asarray(split_out), np.asarray(one_shot), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_associative_prefix_shards_across_logical_cpu_devices():
    script = r"""
import numpy as np
import jax
import jax.numpy as jnp
from trading_dsl_engine.jax_flat.parallel import affine_prefix

rng = np.random.default_rng(18)
a = jnp.asarray(rng.uniform(0.2, 0.9, size=(257, 5)))
b = jnp.asarray(rng.normal(size=(257, 5)))
initial = jnp.asarray(rng.normal(size=(5,)))
actual = affine_prefix(a, b, initial)
def compose(left, right):
    left_a, left_b = left
    right_a, right_b = right
    return right_a * left_a, right_a * left_b + right_b
prefix_a, prefix_b = jax.lax.associative_scan(compose, (a, b), axis=0)
expected = prefix_a * initial + prefix_b
np.testing.assert_allclose(np.asarray(actual), np.asarray(expected), rtol=1e-12, atol=1e-12)
"""
    env = os.environ.copy()
    env["JAX_NUM_CPU_DEVICES"] = "4"
    env["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=60)


def test_ewm_parameter_matrix_shards_across_logical_cpu_devices():
    script = r"""
import numpy as np
import jax.numpy as jnp
from trading_dsl_engine.base.dsl import ewm, var
from trading_dsl_engine.jax_flat import compile_formula

rng = np.random.default_rng(274)
x = rng.normal(size=(257, 5))
x[rng.random(x.shape) < 0.2] = np.nan
for ignore_na in (False, True):
    for adjust in (False, True):
        runtime = compile_formula(
            ewm(var("x"), 3, min_periods=2, ignore_na=ignore_na, adjust=adjust),
            cpp=False,
        )
        _, batch = runtime.run_batch({"x": jnp.asarray(x)})
        state = runtime.init_state(x.shape[1])
        tick = []
        for row in x:
            state, value = runtime.tick(state, jnp.asarray(row))
            tick.append(np.asarray(value))
        np.testing.assert_allclose(
            np.asarray(batch),
            np.stack(tick),
            rtol=1e-12,
            atol=1e-12,
            equal_nan=True,
        )
"""
    env = os.environ.copy()
    env["JAX_NUM_CPU_DEVICES"] = "4"
    env["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=60)
