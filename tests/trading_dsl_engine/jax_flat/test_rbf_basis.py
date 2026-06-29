import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import rbf_basis, einsum, future_rbf_basis_sum, var
from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import FutureRbfBasisSumOp, RbfBasisOp


def test_rbf_basis_uses_epoch_session_clock_and_masks_out_of_session_rows():
    ev_ts = jnp.array([[110.0, 150.0, 210.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)

    formula = rbf_basis(var("ev_ts"), var("start"), var("end"), 4)
    _, out = compile_formula(formula).run_batch({"ev_ts": ev_ts, "start": start, "end": end})

    expected = np.asarray(RbfBasisOp._normalized_basis(jnp.array([0.1, 0.5], dtype=jnp.float64), 4))
    out_np = np.asarray(out[0])
    np.testing.assert_allclose(out_np[:2], expected, rtol=1e-12, atol=1e-12)
    assert np.all(np.isnan(out_np[2]))  # outside [start, end)


def test_future_rbf_basis_sum_uses_current_session_without_phase_precompute():
    ev_ts = jnp.array([[100.0, 125.0, 199.0, 200.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)

    formula = future_rbf_basis_sum(var("ev_ts"), var("start"), var("end"), 3, 8)
    _, out = compile_formula(formula).run_batch({"ev_ts": ev_ts, "start": start, "end": end})

    expected = np.asarray(FutureRbfBasisSumOp._basis_suffix_table(3, 8))[[1, 3, 8, 8]]
    np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-12, atol=1e-12)


def test_rbf_projection_matches_explicit_session_phase_projection_inside_session():
    ev_ts = jnp.array([[125.0, 175.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)

    formula = einsum(
        rbf_basis(var("ev_ts"), var("start"), var("end"), 3),
        future_rbf_basis_sum(var("ev_ts"), var("start"), var("end"), 3, 8),
        "nf,nf->n",
    )

    _, out = compile_formula(formula).run_batch({"ev_ts": ev_ts, "start": start, "end": end})

    phase = np.asarray(((ev_ts - start) / (end - start))[0])
    basis = np.asarray(RbfBasisOp._normalized_basis(jnp.asarray(phase), 3))
    suffix = np.asarray(FutureRbfBasisSumOp._basis_suffix_table(3, 8))[[3, 7]]
    expected = np.einsum("nf,nf->n", basis, suffix)
    np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-12, atol=1e-12)
