import numpy as np

from trading_dsl_engine.jax_flat import compile_formula
from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_formula_native


def _run(formula, **data):
    runtime = compile_formula(formula, cpp=False)
    _, out = runtime.run_batch(data)
    return np.asarray(out)


def test_cache_clip_norm_and_invert_magic():
    x = np.array([[1.0, 2.0, 4.0], [3.0, np.nan, -1.0]], dtype=np.float64)

    runtime = compile_formula("add(cache(x), 1.0)", cpp=False)
    _, out = runtime.run_batch({"x": x})
    np.testing.assert_allclose(out, x + 1.0, equal_nan=True)
    cached = runtime.get_cached_values()
    assert tuple(cached) == runtime.program.cache_nodes
    np.testing.assert_allclose(next(iter(cached.values())), x, equal_nan=True)

    disk_runtime = compile_formula('cache(x, "disk")', cpp=False)
    _, disk_out = disk_runtime.run_batch({"x": x})
    disk_cached = next(iter(disk_runtime.get_cached_values().values()))
    assert isinstance(disk_cached, np.memmap)
    np.testing.assert_allclose(disk_out, x, equal_nan=True)
    np.testing.assert_allclose(disk_cached, x, equal_nan=True)
    np.testing.assert_allclose(_run("clip(x, 0.0, 2.5)", x=x), np.clip(x, 0.0, 2.5), equal_nan=True)
    np.testing.assert_allclose(_run("xs_norm(x)", x=x)[0], [1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0])
    np.testing.assert_allclose(_run("~x", x=x), 1.0 - x, equal_nan=True)


def test_xs_rank_uses_n_plus_one_and_normal_scores():
    x = np.array([[1.0, 2.0, 3.0, np.nan]], dtype=np.float64)
    out = _run("xs_rank(x)", x=x)[0]
    expected = _run("norm_inv(cache(r))", r=np.array([[0.25, 0.5, 0.75, np.nan]], dtype=np.float64))[0]
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_hl_zero_ridge_is_stateless_and_get_hat_projects_current_row():
    x = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], dtype=np.float64)
    y = x + 1.0
    beta = _run("get_beta(Ridge(x, y, 0, 0.1))", x=x, y=y)
    hat = _run("get_hat(Ridge(x, y, 0, 0.1))", x=x, y=y)

    for t in range(x.shape[0]):
        xmat = x[t, :, None]
        xx = xmat.T @ xmat
        system = xx + 0.1 * np.diag(np.diag(xx))
        np.testing.assert_allclose(beta[t], np.linalg.solve(system, xmat.T @ y[t]))
        np.testing.assert_allclose(hat[t], xmat @ np.linalg.solve(system, xmat.T))

    runtime = compile_formula("Ridge(x, y, 0, 0.1)", cpp=False)
    ridge_op = runtime.program.nodes[runtime.program.outputs[0]].op
    assert not ridge_op.is_stateful


def test_cpp_flat_new_stateless_ops_match_jax_flat():
    x = np.array([[0.2, 0.5, 0.8], [0.1, np.nan, 0.9]], dtype=np.float64)
    for formula in ("norm_inv(x)", "xs_norm(x)", "clip(x, 0.25, 0.75)", "cache(xs_rank(x))"):
        jax_runtime = compile_formula(formula, cpp=False)
        cpp_runtime = compile_formula_native(formula)
        _, jax_out = jax_runtime.run_batch({"x": x})
        _, cpp_out = cpp_runtime.run_batch({"x": x})
        np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=1e-7, atol=1e-7, equal_nan=True)
