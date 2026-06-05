import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import InstrumentBasisMean, get_beta, rbf_basis, var
from trading_dsl_engine.jax_flat.engine import compile_formula


def test_instrument_basis_mean_batch_matches_tick_and_tracks_per_instrument_betas():
    x = jnp.array(
        [
            [0.0, 0.5],
            [0.25, 0.75],
            [0.5, 1.0],
            [jnp.nan, 0.25],
        ],
        dtype=jnp.float64,
    )
    y = jnp.array(
        [
            [10.0, 20.0],
            [12.0, 22.0],
            [14.0, jnp.nan],
            [16.0, 24.0],
        ],
        dtype=jnp.float64,
    )
    runtime = compile_formula(get_beta(InstrumentBasisMean(rbf_basis(var("x"), 3), var("y"), 1.0, 100.0)))

    state, batch_out = runtime.run_batch({"x": x, "y": y})

    tick_state = runtime.init_state(2)
    tick_rows = []
    for row in range(x.shape[0]):
        tick_state, tick_out = runtime.tick(tick_state, x[row], y[row])
        tick_rows.append(np.asarray(tick_out))

    np.testing.assert_allclose(np.asarray(batch_out), np.stack(tick_rows), rtol=1e-12, atol=1e-12)
    assert batch_out.shape == (4, 2, 3)
    assert np.all(np.isfinite(np.asarray(batch_out[-1, 0])))
    assert np.all(np.isfinite(np.asarray(batch_out[-1, 1])))
    assert len(state) == len(tick_state)
