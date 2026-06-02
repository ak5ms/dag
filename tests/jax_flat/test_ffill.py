import numpy as np
import pytest
import jax.numpy as jnp

from trading_dsl_engine.base.dsl import ffill, var
from trading_dsl_engine.jax_flat.engine import compile_formula


def test_ffill_batch_matches_reference_and_streaming_ticks():
    runtime = compile_formula("ffill(close, 2)")
    close = np.array(
        [
            [1.0, np.nan, 5.0],
            [np.nan, 2.0, np.nan],
            [np.nan, np.nan, np.nan],
            [4.0, np.nan, np.nan],
            [np.nan, np.nan, 9.0],
            [np.nan, 8.0, np.nan],
        ],
        dtype=np.float64,
    )
    expected = np.array(
        [
            [1.0, np.nan, 5.0],
            [1.0, 2.0, 5.0],
            [1.0, 2.0, 5.0],
            [4.0, 2.0, np.nan],
            [4.0, np.nan, 9.0],
            [4.0, 8.0, 9.0],
        ],
        dtype=np.float64,
    )

    _, out_batch = runtime.run_batch((jnp.asarray(close),))
    np.testing.assert_allclose(np.asarray(out_batch), expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    stream_runtime = compile_formula("ffill(close, 2)")
    state = stream_runtime.init_state(close.shape[1])
    rows = []
    for row in close:
        state, out = stream_runtime.tick(state, jnp.asarray(row))
        rows.append(np.asarray(out))
    np.testing.assert_allclose(np.vstack(rows), expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_ffill_dynamic_nan_limit_suppresses_output_without_updating_state():
    runtime = compile_formula("ffill(close, limit)")
    close = np.array(
        [
            [1.0, 10.0],
            [np.nan, np.nan],
            [np.nan, 12.0],
            [np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    limit = np.array(
        [
            [2.0, 2.0],
            [np.nan, np.nan],
            [2.0, 2.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )

    _, out = runtime.run_batch((jnp.asarray(close), jnp.asarray(limit)))
    expected = np.array(
        [
            [1.0, 10.0],
            [np.nan, np.nan],
            [1.0, 12.0],
            [1.0, 12.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_ffill_python_helper_and_literal_validation():
    runtime = compile_formula(ffill(var("close"), 1))
    _, out = runtime.run_batch((jnp.asarray([[1.0], [np.nan], [np.nan]], dtype=jnp.float64),))
    expected = np.array([[1.0], [1.0], [np.nan]], dtype=np.float64)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-12, equal_nan=True)

    with pytest.raises(ValueError, match="ffill limit must be >= 0"):
        compile_formula("ffill(close, -1)")
