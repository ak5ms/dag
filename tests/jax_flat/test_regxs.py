import numpy as np

from trading_dsl_engine.jax_flat import compile_formula


def _run(formula, **data):
    runtime = compile_formula(formula, cpp=False)
    _, out = runtime.run_batch(data)
    return np.asarray(out)


def _expected_fp(xmat, lam):
    valid = np.isfinite(xmat)
    x0 = np.where(valid, xmat, 0.0)
    counts = valid.astype(float).T @ valid.astype(float)
    sums = x0.T @ x0
    xx = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    system = xx + lam * np.diag(np.diag(xx))
    inv = np.linalg.pinv(system)
    col_counts = np.maximum(valid.sum(axis=0).astype(float), 1.0)
    fp = (x0 / col_counts) @ inv.T
    return np.where(valid, fp, np.nan)


def test_regxs_get_fp_uses_pairwise_intersection_counts_for_nan_xx():
    x1 = np.array([[1.0, np.nan, 3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[2.0, 4.0, 1.0, np.nan]], dtype=np.float64)

    fp = _run("get_fp(RegXS(cat(x1, x2), 0.1))", x1=x1, x2=x2)
    expected = _expected_fp(np.column_stack([x1[0], x2[0]]), 0.1)[None, :, :]
    np.testing.assert_allclose(fp, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_regxs_factor_portfolio_reconstructs_pairwise_mean_beta_with_predictor_nans():
    rng = np.random.default_rng(123)
    x1 = rng.normal(size=(48, 9))
    x2 = rng.normal(size=(48, 9))
    y = 0.7 * x1 - 0.2 * x2 + rng.normal(scale=0.05, size=(48, 9))
    x1[rng.random(x1.shape) < 0.15] = np.nan
    x2[rng.random(x2.shape) < 0.17] = np.nan

    fp = _run("get_fp(RegXS(cat(x1, x2), 0.05))", x1=x1, x2=x2)
    y0 = np.where(np.isfinite(y), y, 0.0)
    beta_from_fp = np.einsum("tnk,tn->tk", np.nan_to_num(fp), y0)
    expected = []
    for x1_row, x2_row, y_row in zip(x1, x2, y, strict=True):
        xmat = np.column_stack([x1_row, x2_row])
        fp_row = _expected_fp(xmat, 0.05)
        expected.append(np.nan_to_num(fp_row).T @ np.nan_to_num(y_row))
    np.testing.assert_allclose(beta_from_fp, np.asarray(expected), rtol=1e-10, atol=1e-10)
