"""One-pass Ridge -> risk model -> CVXPYgen MPO -> downstream PnL example.

The generated runner has one temporal loop. Ridge forecasts, the matrix EWM risk
model, PSD factorization, CVXPYgen canonicalization, persistent Clarabel solve,
and downstream ``shift(weights[0]) * returns`` all run in that loop. No optimizer
input is materialized as a historical array and there is no second pass over time.
"""

from __future__ import annotations

import os
from pathlib import Path

import cvxpy as cp
import numpy as np

from flows.riskmodel import risk_covariance
from trading_dsl_engine.base.dsl import (
    Ridge,
    cat,
    ewm,
    get_preds,
    psd_factor,
    shift,
    var,
)
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.optimizer import (
    ClarabelNativePaths,
    build_current_clarabel,
    clarabel_program,
    get_field,
)

N_ASSETS = 6
N_HORIZONS = 3
ROWS = 500
CACHE = Path(".generated/cpp_stream_mpo_one_pass")


def _clarabel() -> ClarabelNativePaths:
    include = os.environ.get("CLARABEL_INCLUDE_DIR")
    library = os.environ.get("CLARABEL_STATIC_LIBRARY")
    if include and library:
        return ClarabelNativePaths(Path(include), Path(library))
    return build_current_clarabel()


@clarabel_program(
    cache_dir=CACHE / "cvxpygen",
    clarabel=_clarabel,
    parameter_options={
        "half_spread_bps": {"nonneg": True},
        "risk_radius": {"nonneg": True},
    },
)
def MPO(
    expected_returns,
    half_spread_bps,
    current_weights,
    risk_factor,
    risk_radius=0.08,
) -> cp.Problem:
    """Define the optimizer once; the decorator supplies CVXPY Parameters."""

    n_horizons, n_assets = expected_returns.shape
    weights = cp.Variable((n_horizons, n_assets), name="weights")
    turnover = cp.Variable((n_horizons, n_assets), name="turnover")
    previous = cp.vstack([current_weights, weights[:-1]])
    delta = weights - previous
    turnover_up = turnover >= delta
    turnover_up.set_label("turnover_up")
    turnover_down = turnover >= -delta
    turnover_down.set_label("turnover_down")
    constraints = [turnover_up, turnover_down]
    for horizon in range(n_horizons):
        risk = cp.SOC(risk_radius, risk_factor @ weights[horizon])
        risk.set_label(f"risk_{horizon}")
        constraints.append(risk)
    return cp.Problem(
        cp.Minimize(
            -cp.sum(cp.multiply(expected_returns, weights))
            + cp.sum(cp.multiply(half_spread_bps * 1e-4, turnover))
        ),
        constraints,
    )


def _formula():
    returns = var("returns")
    lagged = shift(returns, 1, 1)
    fast_level = ewm(returns, 8, min_periods=2)

    # Three streaming Ridge models produce one forecast vector per horizon.
    horizon_forecasts = tuple(
        get_preds(
            Ridge(
                lagged,
                fast_level,
                y=returns,
                hl=half_life,
                lambda_=0.1,
            )
        )
        for half_life in (8, 32, 128)
    )
    expected_returns = cat(*horizon_forecasts)  # logical shape (assets, horizons)

    covariance = risk_covariance(
        returns,
        span=64,
        min_periods=8,
        ignore_na=True,
        adjust=False,
    )
    # psd_factor emits row-major L. CVXPY's column-major parameter ABI sees L.T,
    # so C.T @ C equals L @ L.T, the repaired covariance matrix.
    risk_factor = psd_factor(covariance, eigenvalue_floor=1e-8)

    mpo = MPO(
        expected_returns=expected_returns,
        half_spread_bps=var("half_spread_bps"),
        current_weights=var("current_weights"),
        risk_factor=risk_factor,
        risk_radius=0.08,
    )
    next_weights = get_field(mpo, "weights[0]")
    first_horizon_turnover = get_field(mpo, "turnover[0]")
    turnover_lagrangian = get_field(mpo, "turnover_up.lagrangian[0]")
    first_risk_dual = get_field(mpo, "risk_0.dual")
    first_risk_value = get_field(mpo, "risk_0.value")
    objective = get_field(mpo, "objective")
    iterations = get_field(mpo, "iterations")

    # This remains downstream of the native optimizer in the same row transition.
    pnl = shift(next_weights, 1, 1) * returns
    return (
        pnl,
        next_weights,
        first_horizon_turnover,
        turnover_lagrangian,
        first_risk_dual,
        first_risk_value,
        objective,
        iterations,
    )


def _simulation() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    loadings = rng.normal(size=(N_ASSETS, 2))
    covariance = loadings @ loadings.T
    covariance /= np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    covariance = 2e-5 * covariance + 8e-5 * np.eye(N_ASSETS)
    returns = rng.multivariate_normal(
        np.zeros(N_ASSETS), covariance, size=ROWS
    )
    returns[rng.random(returns.shape) < 0.01] = np.nan
    half_spread = np.broadcast_to(
        np.linspace(0.5, 1.5, N_ASSETS), (ROWS, N_ASSETS)
    ).copy()
    current_weights = np.zeros((ROWS, N_ASSETS), dtype=np.float64)
    return {
        "returns": returns,
        "half_spread_bps": half_spread,
        "current_weights": current_weights,
    }


def main() -> None:
    data = _simulation()
    runtime = compile_formula(list(_formula()), data)

    generated = runtime.generated_cpp.read_text()
    row_loop = "for (std::size_t t = row_begin; t < row_end; ++t)"
    assert generated.count(row_loop) == 1
    assert generated.count("stackdsl::CvxpygenNode<") == 1
    assert "stackdsl::PsdFactorNode<" in generated
    assert "stackdsl::RidgeNode<" in generated or "stackdsl::RidgeBundleNode<" in generated

    result = runtime.run(out_path=CACHE / "result.npy")
    (
        pnl,
        weights,
        turnover,
        turnover_lagrangian,
        risk_dual,
        risk_value,
        objective,
        iterations,
    ) = result.load()
    print(runtime.explain())
    print(f"single temporal loop: {generated.count(row_loop)}")
    print(f"rows={result.rows}, seconds={result.seconds:.6f}")
    print(f"pnl shape={pnl.shape}")
    print(f"weights shape={weights.shape}")
    print(f"turnover shape={turnover.shape}")
    print(f"turnover Lagrangian shape={turnover_lagrangian.shape}")
    print(f"risk dual/value shapes={risk_dual.shape}/{risk_value.shape}")
    print(f"objective/iterations shapes={objective.shape}/{iterations.shape}")
    print(f"last weights={np.asarray(weights[-1])}")


if __name__ == "__main__":
    main()
