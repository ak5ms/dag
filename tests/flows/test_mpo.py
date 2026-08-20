import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

from flows.mpo import build_mpo_problem, solve_mpo
from flows.riskmodel import covariance_diagnostics, risk_covariance, sanitize_covariance
from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.jax_flat import compile_formula


def test_risk_covariance_preserves_existing_pairwise_missing_semantics():
    # The production flow treats NaN and exact-zero returns as missing observations.
    returns = np.array(
        [
            [1.0, 2.0],
            [0.0, 3.0],
            [np.nan, 4.0],
            [5.0, 0.0],
            [6.0, 7.0],
        ],
        dtype=np.float64,
    )
    expr = risk_covariance(var("returns"), span=3)
    runtime = compile_formula(expr, cpp=False)
    _, actual = runtime.run_batch({"returns": jnp.asarray(returns)})
    actual = np.asarray(actual)

    observed = np.nan_to_num(returns, nan=0.0)
    products = np.einsum("ti,tj->tij", observed, observed)
    products[products == 0.0] = np.nan
    expected = (
        pd.DataFrame(products.reshape(len(returns), -1))
        .ewm(span=3, ignore_na=True, adjust=False)
        .mean()
        .to_numpy()
        .reshape(products.shape)
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12, equal_nan=True)
    # Intermittent missingness/zeros must not poison the final snapshot.
    assert np.all(np.isfinite(actual[-1]))


def test_sanitize_covariance_handles_missing_asymmetry_and_zero_variance():
    raw = np.array(
        [
            [0.04, 0.018, np.nan, 0.0],
            [0.020, 0.09, np.nan, 0.01],
            [0.03, np.nan, 0.00, np.nan],
            [np.nan, 0.012, np.nan, 0.16],
        ]
    )
    # Default optimizer policy is strict: do not invent a missing asset variance.
    with pytest.raises(ValueError, match="coverage"):
        sanitize_covariance(raw, max_condition_number=1e6)
    # The sanitizer can still repair a deliberately relaxed snapshot when requested.
    clean = sanitize_covariance(
        raw,
        max_condition_number=1e6,
        min_diagonal_coverage=0.5,
        min_finite_fraction=0.5,
    )
    np.testing.assert_allclose(clean, clean.T, rtol=0.0, atol=1e-12)
    assert np.all(np.isfinite(clean))
    diagnostics = covariance_diagnostics(clean)
    assert diagnostics["finite"]
    assert diagnostics["min_eigenvalue"] > 0.0
    assert diagnostics["condition_number"] <= 1.00001e6


def test_sanitize_covariance_rejects_insufficient_or_degenerate_risk_coverage():
    with pytest.raises(ValueError, match="degenerate"):
        sanitize_covariance(np.full((3, 3), np.nan))
    with pytest.raises(ValueError, match="degenerate"):
        sanitize_covariance(np.zeros((3, 3)))

    mostly_missing = np.full((4, 4), np.nan)
    mostly_missing[0, 0] = 0.01
    with pytest.raises(ValueError, match="coverage"):
        sanitize_covariance(mostly_missing)


def test_build_problem_encodes_turnover_and_soc_risk():
    expected_returns = np.array([[0.01, -0.02], [0.005, 0.01]])
    covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
    current = np.array([0.10, -0.20])
    problem = build_mpo_problem(
        expected_returns,
        covariance,
        half_spread_bps=np.array([1.0, 2.0]),
        risk_constraint=np.array([0.02, 0.03]),
        current_weights=current,
    )
    assert problem.num_nonneg_cones == 8
    assert problem.so_cone_dims == (3, 3)
    assert problem.A.shape == (14, 8)
    np.testing.assert_allclose(problem.half_spread[0], np.array([1e-4, 2e-4]))

    # A feasible hand-picked point with exact auxiliary turnover.
    weights = np.array([[0.05, -0.04], [0.02, 0.03]])
    delta = np.vstack([weights[0] - current, weights[1] - weights[0]])
    u = np.abs(delta)
    x = np.concatenate([weights.reshape(-1), u.reshape(-1)])
    slack = problem.b - problem.A @ x
    assert np.min(slack[:problem.num_nonneg_cones]) >= -1e-12
    offset = problem.num_nonneg_cones
    for t, dim in enumerate(problem.so_cone_dims):
        cone = slack[offset:offset + dim]
        assert np.linalg.norm(cone[1:]) <= cone[0] + 1e-12
        # SOC norm is exactly sqrt(w' S w) for the encoded factor.
        assert np.isclose(np.linalg.norm(cone[1:]) ** 2, weights[t] @ problem.covariance[t] @ weights[t])
        offset += dim


def test_soc_is_exact_for_more_than_two_assets():
    n = 4
    covariance = np.array(
        [
            [0.08, 0.01, 0.02, 0.00],
            [0.01, 0.12, 0.01, 0.02],
            [0.02, 0.01, 0.10, 0.01],
            [0.00, 0.02, 0.01, 0.06],
        ]
    )
    weights = np.array([[0.10, -0.07, 0.04, 0.03]])
    problem = build_mpo_problem(np.zeros((1, n)), covariance, 1.0, np.array([0.01]))

    u = np.abs(weights[0])
    x = np.concatenate([weights.reshape(-1), u])
    slack = problem.b - problem.A @ x
    assert np.min(slack[:problem.num_nonneg_cones]) >= -1e-12
    cone = slack[problem.num_nonneg_cones:]
    assert cone.shape == (n + 1,)
    assert np.linalg.norm(cone[1:]) <= cone[0] + 1e-12
    assert np.isclose(
        np.linalg.norm(cone[1:]) ** 2,
        weights[0] @ problem.covariance[0] @ weights[0],
        rtol=1e-10,
        atol=1e-12,
    )


def test_build_problem_accepts_horizon_specific_covariances_and_spreads():
    h, n = 3, 4
    er = np.arange(h * n, dtype=np.float64).reshape(h, n) * 1e-4
    cov = np.stack([np.diag(np.linspace(1e-4, 4e-4, n)) * (1.0 + 0.1 * t) for t in range(h)])
    hs = np.arange(h * n, dtype=np.float64).reshape(h, n) * 0.1 + 0.5
    risk = np.array([1e-3, 2e-3, 3e-3])
    problem = build_mpo_problem(er, cov, hs, risk)
    assert problem.covariance.shape == (h, n, n)
    assert problem.half_spread.shape == (h, n)
    assert problem.so_cone_dims == (n + 1,) * h


def test_moreau_solution_respects_risk_and_auxiliary_cost():
    covariance = np.array(
        [
            [0.00016, 0.00004, 0.00002],
            [0.00004, 0.00025, 0.00003],
            [0.00002, 0.00003, 0.00009],
        ]
    )
    expected_returns = np.array(
        [
            [8e-4, 2e-4, -1e-4],
            [5e-4, 4e-4, -1e-4],
            [2e-4, 3e-4, 1e-4],
        ]
    )
    risk = np.full(3, 0.10**2 / 252.0)
    result = solve_mpo(
        expected_returns,
        covariance,
        half_spread_bps=np.array([0.5, 1.0, 1.5]),
        risk_constraint=risk,
    )
    assert result.weights.shape == expected_returns.shape
    assert np.all(result.risk_variance <= risk * (1.0 + 2e-5) + 1e-12)
    assert np.isfinite(result.objective)
    assert result.iterations > 0
