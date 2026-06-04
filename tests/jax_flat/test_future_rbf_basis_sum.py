import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import einsum, future_rbf_basis_sum, rbf_basis, var
from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import _future_rbf_basis_sum, _rbf_basis


def _explicit_future_sum(phase, n_basis: int, n_steps: int):
    grid = jnp.arange(n_steps, dtype=jnp.float64) / n_steps
    basis = _rbf_basis(grid, n_basis)
    suffix = jnp.flip(jnp.cumsum(jnp.flip(basis, axis=0), axis=0), axis=0)
    suffix = jnp.concatenate((suffix, jnp.zeros((1, n_basis), dtype=basis.dtype)), axis=0)
    idx = jnp.clip(jnp.floor(phase * n_steps).astype(jnp.int32) + 1, 0, n_steps)
    return suffix[idx]


def test_rbf_basis_is_non_circular_and_normalized():
    x = jnp.array([[0.0, 0.5, 1.0], [jnp.nan, 0.25, 0.75]], dtype=jnp.float64)

    _, out = compile_formula(rbf_basis(var("x"), 4)).run_batch({"x": x})

    out_np = np.asarray(out)
    np.testing.assert_allclose(np.nansum(out_np, axis=2), np.array([[1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]))
    assert out_np[0, 0, 0] > out_np[0, 0, -1]
    assert out_np[0, 2, -1] > out_np[0, 2, 0]
    assert np.all(np.isnan(out_np[1, 0]))


def test_future_rbf_basis_sum_matches_explicit_future_grid_sum():
    phase = jnp.array(
        [
            [0.0, 1.0 / 8.0, 6.9 / 8.0, 7.0 / 8.0],
            [jnp.nan, 0.25, 0.5, 0.99],
        ],
        dtype=jnp.float64,
    )
    n_basis = 3
    n_steps = 8

    _, out = compile_formula(future_rbf_basis_sum(var("phase"), n_basis, n_steps)).run_batch({"phase": phase})

    expected = jax.vmap(lambda row: _explicit_future_sum(row, n_basis, n_steps))(phase)
    expected = np.array(expected)
    expected[1, 0, :] = np.nan
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-12)


def test_future_rbf_basis_sum_projects_linear_model_without_lag_cube():
    phase = jnp.array([[0.0, 0.5], [0.75, 0.99]], dtype=jnp.float64)
    beta = rbf_basis(var("phase"), 3)
    mass = future_rbf_basis_sum(var("phase"), 3, 8)

    _, out = compile_formula(einsum(beta, mass, "nf,nf->n")).run_batch({"phase": phase})

    beta_np = np.asarray(_rbf_basis(phase.reshape(-1), 3)).reshape(phase.shape + (3,))
    mass_np = np.asarray(_future_rbf_basis_sum(phase.reshape(-1), 3, 8)).reshape(phase.shape + (3,))
    expected = np.einsum("tnf,tnf->tn", beta_np, mass_np)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-12, atol=1e-12)
