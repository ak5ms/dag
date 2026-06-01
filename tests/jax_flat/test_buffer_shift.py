import jax.numpy as jnp
import numpy as np
import pytest

from trading_dsl_engine.base.dsl import buffer, shift, var
from trading_dsl_engine.jax_flat.engine import compile_formula


def test_buffer_shift_batch_returns_ordered_lag_cube_with_dynamic_bounds():
    runtime = compile_formula("buffer(shift(close, lag=upper_lag, max_lag=4), min=min_lag, max=4)")
    close = jnp.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ],
        dtype=jnp.float64,
    )
    upper_lag = jnp.array(
        [
            [1.0, 3.0],
            [1.0, 3.0],
            [3.0, 2.0],
            [4.0, 4.0],
            [4.0, 4.0],
        ],
        dtype=jnp.float64,
    )
    min_lag = jnp.array(
        [
            [1.0, 2.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 3.0],
            [3.0, 1.0],
        ],
        dtype=jnp.float64,
    )

    _, out = runtime.run_batch({"close": close, "upper_lag": upper_lag, "min_lag": min_lag})

    expected = np.array(
        [
            [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            [[1.0, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            [[np.nan, 1.0, np.nan, np.nan], [20.0, 10.0, np.nan, np.nan]],
            [[3.0, 2.0, 1.0, np.nan], [np.nan, np.nan, 10.0, np.nan]],
            [[np.nan, np.nan, 2.0, 1.0], [40.0, 30.0, 20.0, 10.0]],
        ],
        dtype=np.float64,
    )
    assert out.shape == (5, 2, 4)
    np.testing.assert_allclose(np.asarray(out), expected, equal_nan=True)


def test_buffer_shift_live_tick_preserves_ring_order_after_wraparound():
    runtime = compile_formula("buffer(shift(close, upper_lag, 4), min_lag, 4)")
    state = runtime.init_state(2)
    out = None
    for t in range(6):
        state, out = runtime.tick(
            state,
            jnp.array([10.0 + t, 100.0 + t], dtype=jnp.float64),
            jnp.array([4.0, 3.0], dtype=jnp.float64),
            jnp.array([2.0, 1.0], dtype=jnp.float64),
        )

    expected = np.array(
        [
            [np.nan, 13.0, 12.0, 11.0],
            [104.0, 103.0, 102.0, np.nan],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(out), expected, equal_nan=True)


def test_buffer_shift_python_helper_lowers_to_same_jax_flat_ast():
    close = var("close")
    upper_lag = var("upper_lag")
    min_lag = var("min_lag")
    runtime = compile_formula(buffer(shift(close, upper_lag, 3), min_lag, 3))
    _, out = runtime.run_batch(
        {
            "close": jnp.array([[1.0, 10.0], [2.0, 20.0]], dtype=jnp.float64),
            "upper_lag": jnp.ones((2, 2), dtype=jnp.float64),
            "min_lag": jnp.ones((2, 2), dtype=jnp.float64),
        }
    )
    expected = np.array(
        [
            [[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]],
            [[1.0, np.nan, np.nan], [10.0, np.nan, np.nan]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(out), expected, equal_nan=True)


def test_buffer_shift_validates_direct_shift_and_static_capacity():
    with pytest.raises(ValueError, match="buffer first arg must be a direct shift"):
        compile_formula("buffer(close, 1, 2)")
    with pytest.raises(ValueError, match="buffer max_lag must be <= shift max_size"):
        compile_formula("buffer(shift(close, upper_lag, 2), 1, 3)")
