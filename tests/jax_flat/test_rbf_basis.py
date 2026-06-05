import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import einsum, future_RBF_basis_sum, RBF_basis, var
from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import _basis_suffix_table, _normalized_RBF_basis


def test_RBF_basis_supports_direct_phase_form():
    phase = jnp.array([[0.0, 0.5, 1.0], [jnp.nan, 0.25, 0.75]], dtype=jnp.float64)

    _, out = compile_formula(RBF_basis(var("phase"), 4)).run_batch({"phase": phase})

    out_np = np.asarray(out)
    np.testing.assert_allclose(np.nansum(out_np, axis=2), np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))
    assert out_np[0, 0, 0] > out_np[0, 0, -1]
    assert out_np[0, 2, -1] > out_np[0, 2, 0]
    assert np.all(np.isnan(out_np[1, 0]))


def test_future_RBF_basis_sum_supports_direct_phase_form():
    phase = jnp.array([[0.0, 0.25, 0.99, jnp.nan]], dtype=jnp.float64)

    _, out = compile_formula(future_RBF_basis_sum(var("phase"), 3, 8)).run_batch({"phase": phase})

    expected = np.asarray(_basis_suffix_table(3, 8))[[1, 3, 8, 1]]
    expected[-1] = np.nan
    np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-12, atol=1e-12)


def test_RBF_basis_uses_epoch_session_clock_and_masks_non_tradable_rows():
    ev_ts = jnp.array([[110.0, 150.0, 210.0, 150.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)
    is_tradable = jnp.array([[1.0, 1.0, 1.0, 0.0]], dtype=jnp.float64)

    formula = RBF_basis(var("ev_ts"), var("start"), var("end"), var("is_tradable"), 4)
    _, out = compile_formula(formula).run_batch(
        {"ev_ts": ev_ts, "start": start, "end": end, "is_tradable": is_tradable}
    )

    expected = np.asarray(_normalized_RBF_basis(jnp.array([0.1, 0.5], dtype=jnp.float64), 4))
    out_np = np.asarray(out[0])
    np.testing.assert_allclose(out_np[:2], expected, rtol=1e-12, atol=1e-12)
    assert np.all(np.isnan(out_np[2]))  # outside [start, end)
    assert np.all(np.isnan(out_np[3]))  # masked by is_tradable == 0


def test_future_RBF_basis_sum_uses_current_session_without_phase_precompute():
    ev_ts = jnp.array([[100.0, 125.0, 199.0, 200.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)

    formula = future_RBF_basis_sum(var("ev_ts"), var("start"), var("end"), 3, 8)
    _, out = compile_formula(formula).run_batch({"ev_ts": ev_ts, "start": start, "end": end})

    expected = np.asarray(_basis_suffix_table(3, 8))[[1, 3, 8, 8]]
    np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-12, atol=1e-12)


def test_future_RBF_basis_sum_can_roll_to_next_session_after_close():
    ev_ts = jnp.array([[250.0, 350.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)
    next_start = jnp.full_like(ev_ts, 300.0)
    next_end = jnp.full_like(ev_ts, 500.0)

    formula = future_RBF_basis_sum(
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
    expected = np.asarray(_basis_suffix_table(3, 8))[[0, 3]]
    np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-12, atol=1e-12)


def test_future_RBF_basis_sum_uses_next_session_when_current_bounds_are_missing():
    ev_ts = jnp.array([[250.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, jnp.nan)
    end = jnp.full_like(ev_ts, jnp.nan)
    next_start = jnp.full_like(ev_ts, 300.0)
    next_end = jnp.full_like(ev_ts, 500.0)

    formula = future_RBF_basis_sum(
        var("ev_ts"), var("start"), var("end"), var("next_start"), var("next_end"), 3, 8
    )
    _, out = compile_formula(formula).run_batch(
        {"ev_ts": ev_ts, "start": start, "end": end, "next_start": next_start, "next_end": next_end}
    )

    expected = np.asarray(_basis_suffix_table(3, 8))[0]
    np.testing.assert_allclose(np.asarray(out[0, 0]), expected, rtol=1e-12, atol=1e-12)


def test_session_basis_projection_matches_explicit_session_phase_projection_inside_session():
    ev_ts = jnp.array([[125.0, 175.0]], dtype=jnp.float64)
    start = jnp.full_like(ev_ts, 100.0)
    end = jnp.full_like(ev_ts, 200.0)

    formula = einsum(
        RBF_basis(var("ev_ts"), var("start"), var("end"), 3),
        future_RBF_basis_sum(var("ev_ts"), var("start"), var("end"), 3, 8),
        "nf,nf->n",
    )

    _, out = compile_formula(formula).run_batch({"ev_ts": ev_ts, "start": start, "end": end})

    phase = np.asarray(((ev_ts - start) / (end - start))[0])
    basis = np.asarray(_normalized_RBF_basis(jnp.asarray(phase), 3))
    suffix = np.asarray(_basis_suffix_table(3, 8))[[3, 7]]
    expected = np.einsum("nf,nf->n", basis, suffix)
    np.testing.assert_allclose(np.asarray(out[0]), expected, rtol=1e-12, atol=1e-12)
