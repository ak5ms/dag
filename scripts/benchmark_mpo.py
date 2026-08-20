"""Synthetic risk-model + Moreau MPO benchmark.

The simulated signal's exact frictionless annualized ex-ante Sharpe decays linearly
from 3.0 at the shortest horizon to 1.0 at the slowest horizon.
"""

import time

import jax.numpy as jnp
import numpy as np

from flows.mpo import solve_mpo
from flows.riskmodel import covariance_diagnostics, risk_covariance, sanitize_covariance
from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.jax_flat import compile_formula


SEED = 42
N_ASSETS = 24
N_HORIZONS = 8
HISTORY_ROWS = 1_500
RISKMODEL_SPAN = 250
PERIODS_PER_YEAR = 252.0
ANNUAL_RISK = 0.10
N_RUNS = 10


def _true_daily_covariance(rng):
    k = 5
    loadings = rng.normal(size=(N_ASSETS, k))
    raw_factor = loadings @ loadings.T / k
    raw_factor /= np.sqrt(np.outer(np.diag(raw_factor), np.diag(raw_factor)))
    corr = 0.65 * raw_factor + 0.35 * np.eye(N_ASSETS)
    ann_vol = rng.uniform(0.12, 0.35, size=N_ASSETS)
    daily_vol = ann_vol / np.sqrt(PERIODS_PER_YEAR)
    return corr * np.outer(daily_vol, daily_vol)


def make_simulation():
    rng = np.random.default_rng(SEED)
    true_cov = _true_daily_covariance(rng)

    returns = rng.multivariate_normal(np.zeros(N_ASSETS), true_cov, size=HISTORY_ROWS)
    returns[rng.random(returns.shape) < 0.025] = np.nan
    # Match the production risk-model convention that exact zero is also a missing return.
    returns[rng.random(returns.shape) < 0.01] = 0.0

    # Run the same covariance DSL used by flows.riskmodel.cov, just with a shorter span
    # appropriate for this compact simulation.
    risk_runtime = compile_formula(risk_covariance(var("returns"), span=RISKMODEL_SPAN), cpp=False)
    risk_runtime.run_batch({"returns": jnp.asarray(returns)})[1].block_until_ready()
    t0 = time.perf_counter()
    _, risk_path = risk_runtime.run_batch({"returns": jnp.asarray(returns)})
    risk_path.block_until_ready()
    riskmodel_seconds = time.perf_counter() - t0
    raw_risk_cov = np.asarray(risk_path[-1])
    raw_diagnostics = covariance_diagnostics(raw_risk_cov)
    risk_cov = sanitize_covariance(raw_risk_cov)
    clean_diagnostics = covariance_diagnostics(risk_cov)

    # Correlated alpha directions across forecast horizons. Scale each one so the exact
    # frictionless tangency Sharpe under the estimated covariance is 3 -> 1 annualized.
    target_sharpes = np.linspace(3.0, 1.0, N_HORIZONS)
    inv_cov = np.linalg.inv(risk_cov)
    directions = np.empty((N_HORIZONS, N_ASSETS))
    directions[0] = rng.normal(size=N_ASSETS)
    for t in range(1, N_HORIZONS):
        directions[t] = 0.80 * directions[t - 1] + np.sqrt(1.0 - 0.80**2) * rng.normal(size=N_ASSETS)

    expected_returns = np.empty_like(directions)
    realized_sharpes = np.empty(N_HORIZONS)
    for t, target in enumerate(target_sharpes):
        direction = directions[t]
        unit_sr = np.sqrt(direction @ inv_cov @ direction) * np.sqrt(PERIODS_PER_YEAR)
        expected_returns[t] = direction * (target / unit_sr)
        realized_sharpes[t] = (
            np.sqrt(expected_returns[t] @ inv_cov @ expected_returns[t]) * np.sqrt(PERIODS_PER_YEAR)
        )

    half_spread_bps = rng.uniform(0.5, 2.5, size=N_ASSETS)
    current_weights = rng.normal(scale=0.01, size=N_ASSETS)
    risk_constraint = np.full(N_HORIZONS, ANNUAL_RISK**2 / PERIODS_PER_YEAR)
    return {
        "expected_returns": expected_returns,
        "covariance": risk_cov,
        "half_spread_bps": half_spread_bps,
        "current_weights": current_weights,
        "risk_constraint": risk_constraint,
        "target_sharpes": target_sharpes,
        "realized_sharpes": realized_sharpes,
        "raw_diagnostics": raw_diagnostics,
        "clean_diagnostics": clean_diagnostics,
        "riskmodel_seconds": riskmodel_seconds,
    }


def _print_simulation(sim):
    print(f"shape={N_HORIZONS} horizons x {N_ASSETS} assets")
    print(f"target annualized signal Sharpe={np.round(sim['target_sharpes'], 6)}")
    print(f"realized annualized signal Sharpe={np.round(sim['realized_sharpes'], 6)}")
    raw = sim["raw_diagnostics"]
    clean = sim["clean_diagnostics"]
    print(
        "riskmodel raw="
        f"finite_fraction={raw['finite_fraction']:.6f}, "
        f"diagonal_coverage={raw['diagonal_coverage']:.6f}, "
        f"min_eig={raw['min_eigenvalue']:.6e}, condition={raw['condition_number']:.3f}"
    )
    print(
        "riskmodel cleaned="
        f"finite={clean['finite']}, min_eig={clean['min_eigenvalue']:.6e}, "
        f"condition={clean['condition_number']:.3f}, "
        f"steady_state_seconds={sim['riskmodel_seconds']:.6f}"
    )


def main():
    sim = make_simulation()
    _print_simulation(sim)

    args = (
        sim["expected_returns"],
        sim["covariance"],
        sim["half_spread_bps"],
        sim["risk_constraint"],
    )
    kwargs = {"current_weights": sim["current_weights"]}
    first = solve_mpo(*args, **kwargs)
    samples = []
    result = first
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        result = solve_mpo(*args, **kwargs)
        samples.append(time.perf_counter() - t0)

    samples = np.asarray(samples)
    print(
        "end-to-end solve_mpo seconds="
        f"mean={samples.mean():.6f}, median={np.median(samples):.6f}, "
        f"min={samples.min():.6f}, max={samples.max():.6f}"
    )
    print(
        "Moreau IPM last solve="
        f"iterations={result.iterations}, solve_time={result.solve_time:.6f}, "
        f"setup_time={result.setup_time:.6f}, construction_time={result.construction_time:.6f}"
    )
    print(f"objective={result.objective:.8f}")
    print(f"risk usage={np.round(result.risk_variance / sim['risk_constraint'], 6)}")
    print(f"turnover={np.round(result.turnover, 6)}")
    print(f"expected return={np.round(result.expected_return, 8)}")
    print(f"transaction cost={np.round(result.transaction_cost, 8)}")


if __name__ == "__main__":
    main()
