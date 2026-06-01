import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trading_dsl_engine.base.dsl import bspline, einsum, var
from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import _bspline


def _run(formula, *arrays):
    return compile_formula(formula).run_batch(tuple(jnp.asarray(a, dtype=jnp.float64) for a in arrays))[1]


def test_einsum_accepts_arbitrary_number_of_vector_inputs():
    x1 = np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0]], dtype=np.float64)
    x2 = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)
    x3 = np.array([[0.5, 1.5, 2.5], [3.5, 4.5, 5.5]], dtype=np.float64)

    out = np.asarray(_run('einsum(x1, x2, x3, "i,i,i->i")', x1, x2, x3))

    expected = np.einsum("ti,ti,ti->ti", x1, x2, x3)
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_einsum_vector_output_from_vector_and_matrix_inputs():
    x = np.array([[1.0, 2.0, np.nan], [4.0, 5.0, 6.0]], dtype=np.float64)
    y = np.array([[0.0, 0.25, 0.5], [0.75, 1.0, np.nan]], dtype=np.float64)

    out = np.asarray(_run('einsum(x, bspline(y, 4), "i,ij->i")', x, y))

    expected = np.asarray(
        jax.vmap(lambda row_x, row_y: jnp.einsum("i,ij->i", row_x, _bspline(row_y, 4)))(
            jnp.asarray(x),
            jnp.asarray(y),
        )
    )
    assert out.shape == (2, 3)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_einsum_matrix_output_from_matrix_inputs():
    x = np.array([[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]], dtype=np.float64)
    y = np.array([[0.1, 0.3, 0.9], [0.8, 0.6, 0.4]], dtype=np.float64)

    out = np.asarray(_run('einsum(bspline(x, 2), bspline(y, 3), "ij,ik->jk")', x, y))

    expected = np.asarray(
        jax.vmap(lambda row_x, row_y: jnp.einsum("ij,ik->jk", _bspline(row_x, 2), _bspline(row_y, 3)))(
            jnp.asarray(x),
            jnp.asarray(y),
        )
    )
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_einsum_python_composed_formula_matches_string_formula_for_matrix_output():
    x = np.array([[0.0, 0.25], [0.5, 0.75]], dtype=np.float64)
    y = np.array([[1.0, 0.5], [0.25, 0.0]], dtype=np.float64)
    formula = einsum(bspline(var("x"), 3), bspline(var("y"), 3), "ij,ij->ij")

    composed = np.asarray(_run(formula, x, y))
    string = np.asarray(_run('einsum(bspline(x, 3), bspline(y, 3), "ij,ij->ij")', x, y))

    assert composed.shape == (2, 2, 3)
    np.testing.assert_allclose(composed, string, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_einsum_rejects_repeated_output_subscripts_at_compile_time():
    with pytest.raises(ValueError, match="output subscripts must be unique"):
        compile_formula('einsum(bspline(x, 2), bspline(y, 2), "ij,ij->jj")')
