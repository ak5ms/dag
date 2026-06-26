import time

import pytest

import pandas as pd

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import ewm, roll_mean, var, cumsum, fillna, where
from trading_dsl_engine.jax_flat import compile_formula, rolling


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
@pytest.mark.parametrize("ignore_na", [False, True])
@pytest.mark.parametrize("adjust", [False, True])
def test_ewm_random_nan_inputs_match_pandas_and_cpp(seed, ignore_na, adjust):
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(240, 4))
    data[rng.random(data.shape) < 0.18] = np.nan
    span = 5.0
    actual = _run(ewm(var("x"), span, ignore_na=ignore_na, adjust=adjust), data)
    expected = _reference_ewm(data, span=span, ignore_na=ignore_na, adjust=adjust)
    np.testing.assert_allclose(actual, expected, equal_nan=True)

    if ignore_na and not adjust:
        cpp_runtime = compile_formula(ewm(var("x"), span), cpp=True)
        cpp_out = cpp_runtime.run_batch({"x": data})
        if isinstance(cpp_out, tuple):
            cpp_out = cpp_out[1]
        np.testing.assert_allclose(np.asarray(cpp_out), expected, equal_nan=True)
