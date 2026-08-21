from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import cvxpy as cp

from trading_dsl_engine.cpp_stream.optimizer import (
    CvxpyNodeBuild,
    CvxpyOutput,
    constraint_dual,
    constraint_slack,
    cvxpy_node,
    expression_value,
    solver_metric,
    variable_value,
)


def default_mpo_outputs() -> dict[str, CvxpyOutput]:
    """Default projections; callers may replace or extend this mapping."""
    return {
        "weights": variable_value("weights"),
        "next_weights": expression_value("next_weights"),
        "turnover": variable_value("turnover"),
        "expected_return": expression_value("expected_return"),
        "transaction_cost": expression_value("transaction_cost"),
        "risk_0_dual": constraint_dual("risk_0"),
        "risk_0_cone": constraint_slack("risk_0"),
        "status": solver_metric("status"),
        "iterations": solver_metric("iterations"),
        "objective": solver_metric("objective"),
    }


def make_mpo_cvxpy_node(
    n_assets: int,
    n_horizons: int,
    *,
    outputs: Mapping[str, CvxpyOutput | str] | None = None,
    solver_settings: Mapping[str, Any] | None = None,
):
    """Return a static-shape changing-covariance MPO optimizer definition.

    Runtime parameters
    ------------------
    expected_returns:
        ``(horizons, assets)`` expected return forecasts.
    half_spread_bps:
        ``(horizons, assets)`` or upstream-broadcast half spreads, in basis points.
    current_weights:
        Current ``(assets,)`` holdings used as ``w[-1]``.
    risk_radius:
        ``sqrt(risk_variance_constraint)`` for every horizon.
    risk_factor:
        Shared dense ``(assets, assets)`` factor C satisfying ``C.T @ C == S``.
        It may change on every problem, so the repeated solver path updates A/q/b.
    """
    if n_assets <= 0 or n_horizons <= 0:
        raise ValueError("n_assets and n_horizons must be positive")
    selected_outputs = dict(outputs or default_mpo_outputs())
    settings = {
        "max_iter": 200,
        "tol_gap_abs": 1e-8,
        "tol_gap_rel": 1e-8,
        "tol_feas": 1e-8,
        **dict(solver_settings or {}),
    }

    @cvxpy_node(
        outputs=selected_outputs,
        solver_settings=settings,
        name=f"mpo_{n_horizons}x{n_assets}",
    )
    def mpo_node():
        expected_returns = cp.Parameter(
            (n_horizons, n_assets), name="expected_returns"
        )
        half_spread_bps = cp.Parameter(
            (n_horizons, n_assets), nonneg=True, name="half_spread_bps"
        )
        current_weights = cp.Parameter(n_assets, name="current_weights")
        risk_radius = cp.Parameter(
            n_horizons, nonneg=True, name="risk_radius"
        )
        # Reuse one Parameter across horizons when the same S applies throughout.
        risk_factor = cp.Parameter(
            (n_assets, n_assets), name="risk_factor"
        )

        weights = cp.Variable((n_horizons, n_assets), name="weights")
        turnover = cp.Variable((n_horizons, n_assets), name="turnover")
        previous = cp.vstack([current_weights, weights[:-1]])
        delta = weights - previous

        constraint_names: list[str] = ["turnover_pos", "turnover_neg"]
        constraints: list[cp.Constraint] = [
            turnover >= delta,
            turnover >= -delta,
        ]
        for horizon in range(n_horizons):
            constraint_names.append(f"risk_{horizon}")
            constraints.append(
                cp.SOC(risk_radius[horizon], risk_factor @ weights[horizon])
            )
        named_constraints = dict(zip(constraint_names, constraints, strict=True))

        expected_return = cp.sum(cp.multiply(expected_returns, weights))
        transaction_cost = cp.sum(
            cp.multiply(half_spread_bps * 1e-4, turnover)
        )
        problem = cp.Problem(
            cp.Maximize(expected_return - transaction_cost), constraints
        )
        return CvxpyNodeBuild(
            problem=problem,
            parameters={
                "expected_returns": expected_returns,
                "half_spread_bps": half_spread_bps,
                "current_weights": current_weights,
                "risk_radius": risk_radius,
                "risk_factor": risk_factor,
            },
            variables={"weights": weights, "turnover": turnover},
            constraints=named_constraints,
            expressions={
                "next_weights": weights[0],
                "expected_return": expected_return,
                "transaction_cost": transaction_cost,
            },
        )

    return mpo_node
