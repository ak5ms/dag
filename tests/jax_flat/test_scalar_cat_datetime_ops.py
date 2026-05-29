from datetime import datetime, timezone

import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import cat, ceil, dayofyear, floor, hour, round, timeofday, to_dt, var
from trading_dsl_engine.jax_flat.engine import compile_formula


def _epoch_us(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def test_cumsum_broadcasts_scalar_literal_to_instrument_array_in_batch():
    runtime = compile_formula("cumsum(1)")
    dummy = jnp.zeros((4, 3), dtype=jnp.float64)

    _, out = runtime.run_batch((dummy,))

    expected = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ]
    )
    np.testing.assert_allclose(np.asarray(out), expected)


def test_cat_stacks_multiple_alpha_vectors_on_last_axis():
    runtime = compile_formula("cat(alpha1, alpha2, alpha3)")
    alpha1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    alpha2 = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    alpha3 = jnp.array([[100.0, 200.0], [300.0, 400.0]])

    _, out = runtime.run_batch((alpha1, alpha2, alpha3))

    expected = np.stack([np.asarray(alpha1), np.asarray(alpha2), np.asarray(alpha3)], axis=-1)
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(np.asarray(out), expected)


def test_datetime_calendar_and_rounding_ops_from_microsecond_timestamps():
    ev_dt = to_dt(var("ev_ts"), unit="us")
    runtime = compile_formula(
        cat(
            dayofyear(ev_dt),
            timeofday(ev_dt),
            hour(ev_dt),
            floor(ev_dt, freq="H"),
            ceil(ev_dt, freq="H"),
            round(ev_dt, freq="H"),
        )
    )
    ev_ts = jnp.array(
        [
            [_epoch_us(datetime(2024, 2, 29, 23, 58, 30)), _epoch_us(datetime(2023, 1, 1, 0, 29, 30))],
            [_epoch_us(datetime(2024, 3, 1, 0, 2, 0)), _epoch_us(datetime(2023, 12, 31, 23, 31, 0))],
        ],
        dtype=jnp.float64,
    )

    _, out = runtime.run_batch((ev_ts,))

    hour_s = 3600.0
    ev_seconds = np.asarray(ev_ts) * 1e-6
    expected = np.stack(
        [
            np.array([[60.0, 1.0], [61.0, 365.0]]),
            np.array([[86310.0, 1770.0], [120.0, 84660.0]]),
            np.array([[23.0, 0.0], [0.0, 23.0]]),
            np.floor(ev_seconds / hour_s) * hour_s,
            np.ceil(ev_seconds / hour_s) * hour_s,
            np.floor(ev_seconds / hour_s + 0.5) * hour_s,
        ],
        axis=-1,
    )
    assert out.shape == (2, 2, 6)
    np.testing.assert_allclose(np.asarray(out), expected)
