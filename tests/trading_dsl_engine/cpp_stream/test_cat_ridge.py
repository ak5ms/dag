from __future__ import annotations

from pathlib import Path

import numpy as np

from trading_dsl_engine.cpp_stream import compile_npy_formula


def _save(path: Path, value: np.ndarray) -> Path:
    np.save(path, np.asarray(value))
    return path


def _solve_row(
    features: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    ridge_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    n, k = features.shape
    xx = np.zeros((k, k), dtype=np.float64)
    xy = np.zeros(k, dtype=np.float64)
    for j in range(k):
        for ell in range(k):
            valid = np.isfinite(features[:, j]) & np.isfinite(features[:, ell]) & np.isfinite(weights)
            if np.any(valid):
                xx[j, ell] = np.sum(
                    features[valid, j] * weights[valid] * features[valid, ell]
                )
        valid = (
            np.isfinite(features[:, j])
            & np.isfinite(y)
            & np.isfinite(weights)
        )
        if np.any(valid):
            xy[j] = np.sum(features[valid, j] * weights[valid] * y[valid])
    system = xx + ridge_lambda * np.diag(np.diag(xx))
    try:
        beta = np.linalg.solve(system, xy)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(system) @ xy
    if not np.all(np.isfinite(beta)):
        beta = np.zeros(k, dtype=np.float64)
    valid_prediction = np.isfinite(y) & np.all(np.isfinite(features), axis=1)
    preds = np.where(valid_prediction, features @ beta, np.nan)
    return beta, preds


def test_cat_root_writes_row_major_matrix(tmp_path: Path) -> None:
    rows, n = 7, 5
    x1 = np.arange(rows * n, dtype=np.float64).reshape(rows, n)
    x2 = x1 + 100.0
    x3 = -x1
    paths = {
        "x1": _save(tmp_path / "x1.npy", x1),
        "x2": _save(tmp_path / "x2.npy", x2),
        "x3": _save(tmp_path / "x3.npy", x3),
    }
    runtime = compile_npy_formula("cat(x1, x2, x3)", paths, n_instruments=n)
    output = tmp_path / "cat.bin"
    runtime.run_npy_files(paths, out_path=output)
    actual = np.memmap(output, mode="r", dtype=np.float64, shape=(rows, n, 3))
    expected = np.stack((x1, x2, x3), axis=-1)
    np.testing.assert_array_equal(actual, expected)
    assert runtime.plan.output_row_width == n * 3


def test_stateless_ridge_beta_matches_pairwise_reference(tmp_path: Path) -> None:
    rng = np.random.default_rng(20260801)
    rows, n = 12, 6
    x1 = rng.normal(size=(rows, n))
    x2 = rng.normal(size=(rows, n))
    x3 = rng.normal(size=(rows, n))
    y = 0.7 * x1 - 0.3 * x2 + 0.15 * x3 + rng.normal(scale=0.02, size=(rows, n))
    weights = rng.uniform(0.25, 2.0, size=(rows, n))
    x1[1, 2] = np.nan
    x2[3, 4] = np.nan
    x3[5, 1] = np.nan
    y[7, 3] = np.nan
    weights[9, 0] = np.nan
    paths = {
        "x1": _save(tmp_path / "x1.npy", x1),
        "x2": _save(tmp_path / "x2.npy", x2),
        "x3": _save(tmp_path / "x3.npy", x3),
        "y": _save(tmp_path / "y.npy", y),
        "weights": _save(tmp_path / "weights.npy", weights),
    }
    formula = (
        "get_beta(Ridge(cat(x1, x2, x3), "
        "y=y, weights=weights, hl=0, lambda_=0.1))"
    )
    runtime = compile_npy_formula(formula, paths, n_instruments=n)
    output = tmp_path / "beta.bin"
    runtime.run_npy_files(paths, out_path=output)
    actual = np.memmap(output, mode="r", dtype=np.float64, shape=(rows, 3))
    expected = np.stack(
        [
            _solve_row(
                np.stack((x1[t], x2[t], x3[t]), axis=1),
                y[t],
                weights[t],
                0.1,
            )[0]
            for t in range(rows)
        ]
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::RidgeNode" in generated
    assert "stackdsl::FeatureList" in generated
    assert "GroupedRidge" not in generated


def test_stateful_ridge_preds_use_prior_beta_and_streaming_moments(tmp_path: Path) -> None:
    rng = np.random.default_rng(44)
    rows, n = 30, 9
    x1 = rng.normal(size=(rows, n))
    x2 = rng.normal(size=(rows, n))
    x3 = rng.normal(size=(rows, n))
    y = 0.4 * x1 - 0.2 * x2 + 0.1 * x3 + rng.normal(scale=0.05, size=(rows, n))
    paths = {
        "x1": _save(tmp_path / "x1.npy", x1),
        "x2": _save(tmp_path / "x2.npy", x2),
        "x3": _save(tmp_path / "x3.npy", x3),
        "y": _save(tmp_path / "y.npy", y),
    }
    formula = "get_preds(Ridge(cat(x1, x2, x3), y=y, hl=64, lambda_=0.1))"
    runtime = compile_npy_formula(formula, paths, n_instruments=n)
    output = tmp_path / "preds.bin"
    runtime.run_npy_files(paths, out_path=output)
    actual = np.memmap(output, mode="r", dtype=np.float64, shape=(rows, n))

    alpha = 1.0 - np.exp(np.log(0.5) / 64.0)
    beta = np.zeros(3, dtype=np.float64)
    xx = None
    xy = None
    expected = np.empty((rows, n), dtype=np.float64)
    for t in range(rows):
        features = np.stack((x1[t], x2[t], x3[t]), axis=1)
        expected[t] = features @ beta
        xx_new = features.T @ features
        xy_new = features.T @ y[t]
        if xx is None:
            xx = xx_new
            xy = xy_new
        else:
            xx = xx + alpha * (xx_new - xx)
            xy = xy + alpha * (xy_new - xy)
        system = xx + 0.1 * np.diag(np.diag(xx))
        beta = np.linalg.solve(system, xy)
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)


def test_grouped_ridge_uses_same_generic_node(tmp_path: Path) -> None:
    rng = np.random.default_rng(901)
    rows, n = 9, 5
    x1 = rng.normal(size=(rows, n))
    x2 = rng.normal(size=(rows, n))
    y = 0.6 * x1 - 0.25 * x2 + rng.normal(scale=0.01, size=(rows, n))
    paths = {
        "x1": _save(tmp_path / "x1.npy", x1),
        "x2": _save(tmp_path / "x2.npy", x2),
        "y": _save(tmp_path / "y.npy", y),
    }
    formula = (
        "groupby(univ([0, 1], [2, 3, 4]), x1, "
        "get_preds(Ridge(cat(self_, x2), y=y, hl=0, lambda_=0.1)))"
    )
    runtime = compile_npy_formula(formula, paths, n_instruments=n)
    output = tmp_path / "grouped_preds.bin"
    runtime.run_npy_files(paths, out_path=output)
    actual = np.memmap(output, mode="r", dtype=np.float64, shape=(rows, n))
    expected = np.empty_like(actual)
    groups = ((0, 1), (2, 3, 4))
    for t in range(rows):
        for group in groups:
            lanes = np.asarray(group)
            features = np.stack((x1[t, lanes], x2[t, lanes]), axis=1)
            _, preds = _solve_row(features, y[t, lanes], np.ones(len(lanes)), 0.1)
            expected[t, lanes] = preds
    np.testing.assert_allclose(actual, expected, rtol=2e-9, atol=2e-9)
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::RidgeNode" in generated
    assert "stackdsl::GroupedExecution" in generated
    assert "GroupedRidge" not in generated
