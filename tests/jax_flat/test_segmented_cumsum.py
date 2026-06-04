import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import segmented_cumsum, var
from trading_dsl_engine.jax_flat.engine import compile_formula


def test_segmented_cumsum_resets_when_key_changes_and_matches_tick():
    x = jnp.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, jnp.nan],
            [5.0, 50.0],
        ],
        dtype=jnp.float64,
    )
    key = jnp.array(
        [
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 2.0],
            [2.0, 2.0],
            [jnp.nan, 2.0],
        ],
        dtype=jnp.float64,
    )
    runtime = compile_formula(segmented_cumsum(var("x"), var("key")))

    state, batch_out = runtime.run_batch({"x": x, "key": key})

    expected = np.array(
        [
            [1.0, 10.0],
            [3.0, 20.0],
            [3.0, 50.0],
            [7.0, np.nan],
            [5.0, 100.0],
        ]
    )
    np.testing.assert_allclose(np.asarray(batch_out), expected)

    tick_state = runtime.init_state(2)
    tick_rows = []
    for row in range(x.shape[0]):
        tick_state, tick_out = runtime.tick(tick_state, x[row], key[row])
        tick_rows.append(np.asarray(tick_out))
    np.testing.assert_allclose(np.stack(tick_rows), np.asarray(batch_out))
    assert len(state) == len(tick_state)
