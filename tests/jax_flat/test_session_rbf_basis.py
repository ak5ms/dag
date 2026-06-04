import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import (
    einsum,
    future_rbf_basis_sum,
    future_session_rbf_basis_sum,
    rbf_basis,
    session_rbf_basis,
    var,
)
from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import _future_rbf_basis_sum, _rbf_basis


def test_session_rbf_basis_uses_epoch_session_clock_and_masks_non_tradable_rows():
    ev_ts = jnp.array([[110.0, 150.0, 210.0, 150.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)
    is_tradable = jnp.array([[1.0, 1.0, 1.0, 0.0]], dtype=jnp.float64)

    formula = session_rbf_basis(var("ev_ts"), var("start"), var("end"), var("is_tradable"), 4)
    _, out = compile_formula(formula).run_batch(
        {"ev_ts": ev_ts, "start": start, "end": end, "is_tradable": is_tradable}
    )

    expected = np.asarray(_rbf_basis(jnp.array([0.1, 0.5], dtype=jnp.float64), 4))
    out_np = np.asarray(out[0])
    np.testing.assert_allclose(out_np[:2], expected, rtol=1e-12, atol=1e-12)
    assert np.all(np.isnan(out_np[2]))  # outside [start, end)
    assert np.all(np.isnan(out_np[3]))  # masked by is_tradable == 0


def test_future_session_rbf_basis_sum_uses_current_session_without_phase_precompute():
    ev_ts = jnp.array([[100.0, 125.0, 199.0, 200.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)

    formula = future_session_rbf_basis_sum(var("ev_ts"), var("start"), var("end"), 3, 8)
    _, out = compile_formula(formula).run_batch({"ev_ts": ev_ts, "start": start, "end": end})

    phases = jnp.array([0.0, 0.25, 0.99, 1.0], dtype=jnp.float64)
    expected = np.asarray(_future_rbf_basis_sum(phases, 3, 8))
    np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-12, atol=1e-12)


def test_future_session_rbf_basis_sum_can_roll_to_next_session_after_close():
    ev_ts = jnp.array([[250.0, 350.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)
    next_start = jnp.full_like(ev_ts, 300.0)
    next_end = jnp.full_like(ev_ts, 500.0)

    formula = future_session_rbf_basis_sum(
        var("ev_ts"),
        var("start"),
        var("end"),
        var("next_start"),
        var("next_end"),
        3,
        8,
    )
    _, out = compile_formula(formula).run_batch(
        {"ev_ts": ev_ts, "start": start, "end": end, "next_start": next_start, "next_end": next_end}
    )

    # 250 is before next session -> full next-session mass. 350 is 25% into next session.
    full_session = np.asarray(_rbf_basis(jnp.arange(8, dtype=jnp.float64) / 8.0, 3)).sum(axis=0)
    partial_session = np.asarray(_future_rbf_basis_sum(jnp.array([0.25], dtype=jnp.float64), 3, 8))[0]
    expected = np.stack([full_session, partial_session])
    np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-12, atol=1e-12)


def test_future_session_rbf_basis_sum_uses_next_session_when_current_bounds_are_missing():
    ev_ts = jnp.array([[250.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, jnp.nan)
    end = jnp.full_like(ev_ts, jnp.nan)
    next_start = jnp.full_like(ev_ts, 300.0)
    next_end = jnp.full_like(ev_ts, 500.0)

    formula = future_session_rbf_basis_sum(
        var("ev_ts"), var("start"), var("end"), var("next_start"), var("next_end"), 3, 8
    )
    _, out = compile_formula(formula).run_batch(
        {"ev_ts": ev_ts, "start": start, "end": end, "next_start": next_start, "next_end": next_end}
    )

    expected = np.asarray(_rbf_basis(jnp.arange(8, dtype=jnp.float64) / 8.0, 3)).sum(axis=0)
    np.testing.assert_allclose(np.asarray(out[0, 0]), expected, rtol=1e-12, atol=1e-12)


def test_session_basis_projection_matches_phase_based_projection_inside_session():
    ev_ts = jnp.array([[125.0, 175.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)
    phase = (ev_ts - start) / (end - start)

    session_formula = einsum(
        session_rbf_basis(var("ev_ts"), var("start"), var("end"), 3),
        future_session_rbf_basis_sum(var("ev_ts"), var("start"), var("end"), 3, 8),
        "nf,nf->n",
    )
    phase_formula = einsum(rbf_basis(var("phase"), 3), future_rbf_basis_sum(var("phase"), 3, 8), "nf,nf->n")

    _, session_out = compile_formula(session_formula).run_batch({"ev_ts": ev_ts, "start": start, "end": end})
    _, phase_out = compile_formula(phase_formula).run_batch({"phase": phase})

    np.testing.assert_allclose(np.asarray(session_out), np.asarray(phase_out), rtol=1e-12, atol=1e-12)
