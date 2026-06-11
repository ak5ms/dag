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


def test_hl_zero_ridge_is_stateless_and_projects_current_row_beta():
    x = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], dtype=np.float64)
    y = x + 1.0
    beta = _run("get_beta(Ridge(x, y, 0, 0.1))", x=x, y=y)

    for t in range(x.shape[0]):
        xmat = x[t, :, None]
        xx = xmat.T @ xmat
        system = xx + 0.1 * np.diag(np.diag(xx))
        b = xmat.T
        z = np.linalg.solve(system, b)
        np.testing.assert_allclose(beta[t], z @ y[t])

    runtime = compile_formula("Ridge(x, y, 0, 0.1)", cpp=False)
    ridge_op = runtime.program.nodes[runtime.program.outputs[0]].op
    assert not ridge_op.is_stateful


def test_hl_zero_ridge_solves_weighted_design_for_beta_with_nan_consistency():
    x1 = np.array([[1.0, np.nan, 3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[2.0, 4.0, 1.0, np.nan]], dtype=np.float64)
    y = np.array([[5.0, np.nan, 7.0, 11.0]], dtype=np.float64)
    weights = np.array([[1.0, 2.0, 0.5, 1.0]], dtype=np.float64)

    beta = _run("get_beta(Ridge(cat(x1, x2), y, weights, 0, 0.1))", x1=x1, x2=x2, y=y, weights=weights)[0]

    xmat = np.nan_to_num(np.stack((x1[0], x2[0]), axis=1), nan=0.0)
    y0 = np.nan_to_num(y[0], nan=0.0)
    w0 = np.nan_to_num(weights[0], nan=0.0)
    xx = xmat.T @ (xmat * w0[:, None])
    system = xx + 0.1 * np.diag(np.diag(xx))
    b = xmat.T * w0[None, :]
    z = np.linalg.solve(system, b)

    np.testing.assert_allclose(beta, z @ y0, rtol=1e-12, atol=1e-12)


def test_cpp_flat_new_stateless_ops_match_jax_flat():
    x = np.array([[0.2, 0.5, 0.8], [0.1, np.nan, 0.9]], dtype=np.float64)
    for formula in ("norm_inv(x)", "xs_norm(x)", "clip(x, 0.25, 0.75)", "cache(xs_rank(x))"):
        jax_runtime = compile_formula(formula, cpp=False)
        cpp_runtime = compile_formula_native(formula)
        _, jax_out = jax_runtime.run_batch({"x": x})
        _, cpp_out = cpp_runtime.run_batch({"x": x})
        np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=1e-7, atol=1e-7, equal_nan=True)
