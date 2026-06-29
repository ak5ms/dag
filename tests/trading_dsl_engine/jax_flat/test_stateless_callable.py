import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import buffer, shift, var
from trading_dsl_engine.jax_flat import compile_formula, stateless


def test_stateless_callable_flips_buffer_lag_axis_in_batch_and_tick():
    rev = stateless(lambda x: jnp.flip(x, axis=-1), name="rev")
    close = var("close")
    upper_lag = var("upper_lag")
    min_lag = var("min_lag")
    runtime = compile_formula(rev(buffer(shift(close, lag=upper_lag, max_lag=4), min=min_lag, max=4)))

    close_data = jnp.array(
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

    _, batch_out = runtime.run_batch({"close": close_data, "upper_lag": upper_lag, "min_lag": min_lag})

    unflipped = np.array(
        [
            [[np.nan, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            [[1.0, np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan, np.nan]],
            [[np.nan, 1.0, np.nan, np.nan], [20.0, 10.0, np.nan, np.nan]],
            [[3.0, 2.0, 1.0, np.nan], [np.nan, np.nan, 10.0, np.nan]],
            [[np.nan, np.nan, 2.0, 1.0], [40.0, 30.0, 20.0, 10.0]],
        ],
        dtype=np.float64,
    )
    expected = np.flip(unflipped, axis=-1)
    assert batch_out.shape == (5, 2, 4)
    np.testing.assert_allclose(np.asarray(batch_out), expected, equal_nan=True)

    state = runtime.init_state(2)
    live = []
    for row in range(close_data.shape[0]):
        state, tick_out = runtime.tick(state, close_data[row], upper_lag[row], min_lag[row])
        live.append(np.asarray(tick_out))
    np.testing.assert_allclose(np.stack(live), expected, equal_nan=True)


def test_stateless_callable_supports_variadic_functions_and_output_metadata():
    weighted_mean = stateless(
        lambda x, y: jnp.nanmean(jnp.stack((x, y), axis=0), axis=0),
        output_kind="vector",
        output_width=1,
        name="weighted_mean",
    )
    runtime = compile_formula(weighted_mean(var("open"), var("close")))
    open_data = jnp.array([[1.0, 2.0], [jnp.nan, 6.0]], dtype=jnp.float64)
    close_data = jnp.array([[3.0, 4.0], [5.0, 8.0]], dtype=jnp.float64)

    _, out = runtime.run_batch({"open": open_data, "close": close_data})

    expected = np.nanmean(np.stack((np.asarray(open_data), np.asarray(close_data)), axis=0), axis=0)
    np.testing.assert_allclose(np.asarray(out), expected, equal_nan=True)
