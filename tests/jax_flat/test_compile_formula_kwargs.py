import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.jax_flat.engine import compile_formula


def test_compile_formula_accepts_kwargs_for_stateless_ops():
    runtime = compile_formula(
        "where(cond=mask, true=clip(x=value, min=0, max=10), false=fillna(x=fallback, y=0))",
        cpp=False,
    )

    value = jnp.array([[-1.0, 5.0, 11.0], [2.0, 8.0, 20.0]], dtype=jnp.float64)
    mask = jnp.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=jnp.float64)
    fallback = jnp.array([[np.nan, 7.0, np.nan], [3.0, np.nan, 9.0]], dtype=jnp.float64)

    _, out = runtime.run_batch({"value": value, "mask": mask, "fallback": fallback})

    expected = np.where(
        np.asarray(mask) != 0.0,
        np.clip(np.asarray(value), 0.0, 10.0),
        np.nan_to_num(np.asarray(fallback), nan=0.0),
    )
    np.testing.assert_allclose(np.asarray(out), expected)


def test_compile_formula_accepts_kwargs_for_stateful_and_static_arg_ops():
    runtime = compile_formula(
        "cat("
        "arg0=ewm(x=close, span=2, min_periods=1), "
        "arg1=shift(x=close, nlag=lag, max_size=3), "
        "arg2=round(x=open, decimals=0)"
        ")",
        cpp=False,
    )

    close = jnp.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]], dtype=jnp.float64)
    open_ = jnp.array([[1.2, 10.6], [2.4, 20.2], [3.8, 30.9]], dtype=jnp.float64)
    lag = jnp.ones_like(close)

    _, out = runtime.run_batch({"close": close, "open": open_, "lag": lag})

    expected_shift = np.array([[np.nan, np.nan], [1.0, 10.0], [2.0, 20.0]])
    expected_round = np.round(np.asarray(open_), decimals=0)
    assert out.shape == (3, 2, 3)
    np.testing.assert_allclose(np.asarray(out)[:, :, 1], expected_shift, equal_nan=True)
    np.testing.assert_allclose(np.asarray(out)[:, :, 2], expected_round)


def test_compile_formula_accepts_kwargs_for_variadic_and_object_ops():
    runtime = compile_formula("get_beta(x=Ridge(x, y=target, hl=0, lam=0.1))", cpp=False)

    x = jnp.array([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]], dtype=jnp.float64)
    target = 2.0 * x

    _, out = runtime.run_batch({"x": x, "target": target})

    assert out.shape == (3, 1)
    assert np.all(np.isfinite(np.asarray(out)[-1]))
