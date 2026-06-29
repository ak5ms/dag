import os
import time

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


def test_einsum_scalar_output_from_matrix_inputs():
    x = np.array([[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]], dtype=np.float64)
    y = np.array([[0.1, 0.3, 0.9], [0.8, 0.6, 0.4]], dtype=np.float64)

    out = np.asarray(_run('einsum(bspline(x, 3), bspline(y, 3), "ij,ij->")', x, y))

    expected = np.asarray(
        jax.vmap(lambda row_x, row_y: jnp.einsum("ij,ij->", _bspline(row_x, 3), _bspline(row_y, 3)))(
            jnp.asarray(x),
            jnp.asarray(y),
        )
    )
    assert out.shape == (2,)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_cpp_einsum_generic_reduces_to_non_instrument_label():
    from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_formula_native

    x = np.array([[0.0, 0.5, 1.0], [0.2, 0.4, 0.6]], dtype=np.float64)
    y = np.array([[0.1, 0.3, 0.9], [0.8, 0.6, 0.4]], dtype=np.float64)

    jax_runtime = compile_formula('einsum(bspline(x, 3), bspline(y, 3), "ij,ij->j")', cpp=False)
    cpp_runtime = compile_formula_native('einsum(bspline(x, 3), bspline(y, 3), "ij,ij->j")')
    _, jax_out = jax_runtime.run_batch({"x": x, "y": y})
    _, cpp_out = cpp_runtime.run_batch({"x": x, "y": y})

    expected = np.asarray(
        jax.vmap(lambda row_x, row_y: jnp.einsum("ij,ij->j", _bspline(row_x, 3), _bspline(row_y, 3)))(
            jnp.asarray(x),
            jnp.asarray(y),
        )
    )
    assert cpp_out.shape == (2, 3)
    np.testing.assert_allclose(cpp_out, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_cpp_einsum_accepts_feature_vector_label_names_like_jax():
    from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_formula_native

    x1 = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5], [2.0, 3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[0.5, 1.5, 2.5], [0.25, 1.25, 2.25], [0.75, 1.75, 2.75]], dtype=np.float64)
    y = np.array([[2.0, 3.0, 4.0], [2.5, 3.5, 4.5], [3.0, 4.0, 5.0]], dtype=np.float64)
    formula = 'einsum(get_beta(Ridge(cat(x1, x2), y, 0.0, 0.1)), cat(x1, x2), "f,nf->n")'

    jax_runtime = compile_formula(formula, cpp=False)
    cpp_runtime = compile_formula_native(formula)
    inputs = {"x1": x1, "x2": x2, "y": y}
    _, jax_out = jax_runtime.run_batch(inputs)
    _, cpp_out = cpp_runtime.run_batch(inputs)

    assert cpp_out.shape == (3, 3)
    np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=1e-12, atol=1e-12, equal_nan=True)


@pytest.mark.perf
@pytest.mark.skipif(os.getenv("RUN_PERF_TESTS", "0") != "1", reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_cpp_einsum_3m_x_9_x_20_features_vs_jitted_jax():
    from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_formula_native

    t_rows = int(os.getenv("EINSUM_PERF_ROWS", "3000000"))
    n_assets = 9
    n_features = 20
    row = np.arange(t_rows, dtype=np.float64)[:, None]
    col = np.arange(n_assets, dtype=np.float64)[None, :]
    x_np = np.sin(row * 0.00001 + col * 0.1)
    y_np = np.mod(row * 0.000001 + col / max(n_assets - 1, 1), 1.0)

    cpp_runtime = compile_formula_native(f'einsum(x, bspline(y, {n_features}), "i,ij->i")')

    @jax.jit
    def jax_einsum(x, y):
        return jax.vmap(lambda row_x, row_y: jnp.einsum("i,ij->i", row_x, _bspline(row_y, n_features)))(x, y)

    x_jax = jnp.asarray(x_np)
    y_jax = jnp.asarray(y_np)
    jax.block_until_ready(jax_einsum(x_jax, y_jax))
    _, cpp_warm = cpp_runtime.run_batch({"x": x_np, "y": y_np})
    assert cpp_warm.shape == (t_rows, n_assets)

    start = time.perf_counter()
    jax_out = jax_einsum(x_jax, y_jax)
    jax.block_until_ready(jax_out)
    jax_elapsed = time.perf_counter() - start

    start = time.perf_counter()
    _, cpp_out = cpp_runtime.run_batch({"x": x_np, "y": y_np})
    cpp_elapsed = time.perf_counter() - start

    print(
        "einsum_i_ij_to_i_3m_x_9_x_20 "
        f"cpp_elapsed_s={cpp_elapsed:.6f} jax_jit_elapsed_s={jax_elapsed:.6f} "
        f"cpp_rows_per_s={t_rows / cpp_elapsed:.3f} jax_rows_per_s={t_rows / jax_elapsed:.3f} "
        f"shape={cpp_out.shape}"
    )
    np.testing.assert_allclose(cpp_out[:1024], np.asarray(jax_out[:1024]), rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(cpp_out[-1024:], np.asarray(jax_out[-1024:]), rtol=1e-10, atol=1e-10, equal_nan=True)
