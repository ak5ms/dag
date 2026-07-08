import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import bspline, prod, sum, var
import trading_dsl_engine.jax_flat.engine as engine
from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import _bspline


def _run(formula, **inputs):
    return compile_formula(formula, cpp=False).run_batch({k: jnp.asarray(v, dtype=jnp.float64) for k, v in inputs.items()})[1]


def test_sum_reduces_matrix_instrument_axis_to_feature_series():
    x = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.2]], dtype=np.float64)

    out = np.asarray(_run("sum(bspline(x, 4), axis=1)", x=x))

    basis = np.asarray(jax.vmap(lambda row: _bspline(row, 4))(jnp.asarray(x)))
    expected = np.sum(basis, axis=1)
    assert out.shape == (2, 4)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_sum_keepdims_broadcasts_reduced_values_to_original_tick_shape():
    x = np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.2]], dtype=np.float64)

    out = np.asarray(_run("sum(bspline(x, 3), axis=(1, 2), keepdims=True)", x=x))

    basis = np.asarray(jax.vmap(lambda row: _bspline(row, 3))(jnp.asarray(x)))
    expected = np.broadcast_to(np.sum(basis, axis=(1, 2), keepdims=True), basis.shape)
    assert out.shape == (2, 3, 3)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_prod_reduces_with_python_composed_formula():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
    formula = prod(bspline(var("x"), 2), axis=2)

    out = np.asarray(_run(formula, x=x))

    basis = np.asarray(jax.vmap(lambda row: _bspline(row, 2))(jnp.asarray(x)))
    expected = np.prod(basis, axis=2)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12)


def test_time_feature_reduction_outputs_instruments_across_batch_chunks(monkeypatch):
    x = (np.arange(70, dtype=np.float64).reshape(10, 7) % 11.0) / 10.0
    formula = "sum(bspline(x, 5), axis=[0, 2])"

    monkeypatch.setattr(engine, "_BATCH_CHUNK_SIZE", 64)
    full_chunk = np.asarray(_run(formula, x=x))

    monkeypatch.setattr(engine, "_BATCH_CHUNK_SIZE", 3)
    small_chunks = np.asarray(_run(formula, x=x))

    basis = np.asarray(jax.vmap(lambda row: _bspline(row, 5))(jnp.asarray(x)))
    expected = np.sum(basis, axis=(0, 2))
    assert small_chunks.shape == (7,)
    np.testing.assert_allclose(full_chunk, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(small_chunks, expected, rtol=1e-12, atol=1e-12)
