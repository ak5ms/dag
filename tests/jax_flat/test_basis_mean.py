import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import InstrumentBasisMean, einsum, get_beta, rbf_basis, var
from trading_dsl_engine.jax_flat.engine import compile_formula


def test_instrument_basis_mean_batch_matches_tick_and_tracks_per_instrument_betas():
    ev_ts = jnp.array(
        [
            [0.0, 0.5],
            [0.25, 0.75],
            [0.5, 1.0],
            [jnp.nan, 0.25],
        ],
        dtype=jnp.float64,
    )
    start = jnp.zeros_like(ev_ts)
    end = jnp.ones_like(ev_ts)
    y = jnp.array(
        [
            [10.0, 20.0],
            [12.0, 22.0],
            [14.0, jnp.nan],
            [16.0, 24.0],
        ],
        dtype=jnp.float64,
    )
    runtime = compile_formula(
        get_beta(InstrumentBasisMean(rbf_basis(var("ev_ts"), var("start"), var("end"), 3), var("y"), 1.0, 100.0))
    )

    state, batch_out = runtime.run_batch({"ev_ts": ev_ts, "start": start, "end": end, "y": y})

    tick_state = runtime.init_state(2)
    tick_rows = []
    for row in range(ev_ts.shape[0]):
        tick_state, tick_out = runtime.tick(tick_state, ev_ts[row], start[row], end[row], y[row])
        tick_rows.append(np.asarray(tick_out))

    np.testing.assert_allclose(np.asarray(batch_out), np.stack(tick_rows), rtol=1e-12, atol=1e-12)
    assert batch_out.shape == (4, 2, 3)
    assert np.all(np.isfinite(np.asarray(batch_out[-1, 0])))
    assert np.all(np.isfinite(np.asarray(batch_out[-1, 1])))
    assert len(state) == len(tick_state)


def test_instrument_basis_mean_beta_can_feed_matrix_einsum_forecast():
    ev_ts = jnp.array(
        [
            [0.0, 0.1, 0.2],
            [0.3, 0.4, 0.5],
            [0.6, 0.7, 0.8],
        ],
        dtype=jnp.float64,
    )
    start = jnp.zeros_like(ev_ts)
    end = jnp.ones_like(ev_ts)
    y = jnp.arange(1.0, 10.0, dtype=jnp.float64).reshape(3, 3)
    beta = get_beta(InstrumentBasisMean(rbf_basis(var("ev_ts"), var("start"), var("end"), 3), var("y"), 1.0, 100.0))
    formula = einsum(beta, rbf_basis(var("ev_ts"), var("start"), var("end"), 3), "nf,nf->n")

    runtime = compile_formula(formula, cpp=False)
    _state, out = runtime.run_batch({"ev_ts": ev_ts, "start": start, "end": end, "y": y})

    assert out.shape == (3, 3)
    assert np.isfinite(np.asarray(out[-1])).all()
