"""Benchmark CVXPY+Clarabel and CVXPYgen+Clarabel on the MPO formulation.

The problem is the same 8-horizon x 24-asset turnover-aware SOCP used by
``scripts/benchmark_mpo.py``. The script reports three update regimes because
they matter for a native hot loop:

* ``same_data``: no parameter is changed between solves.
* ``q_b``: alpha/current holdings/risk radius change, but covariance factors do not.
* ``all``: covariance factors change too, forcing the canonical A matrix to update.

CVXPYgen emits the generated C sources under ``--code-dir``. The workflow
uploads that directory so the direct C entry point can be inspected and linked
from the DAG independently of the Python wrapper.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import cvxpy as cp
import numpy as np
from cvxpygen import cpg


PERIODS_PER_YEAR = 252.0
ANNUAL_RISK = 0.10


@dataclass(frozen=True)
class TimingSummary:
    runs: int
    mean_ms: float
    median_ms: float
    min_ms: float
    p90_ms: float
    max_ms: float


def summarize(samples_s: list[float]) -> TimingSummary:
    values = np.asarray(samples_s, dtype=np.float64) * 1e3
    return TimingSummary(
        runs=int(values.size),
        mean_ms=float(values.mean()),
        median_ms=float(np.median(values)),
        min_ms=float(values.min()),
        p90_ms=float(np.percentile(values, 90)),
        max_ms=float(values.max()),
    )


def benchmark(fn: Callable[[], object], *, warmups: int, runs: int) -> TimingSummary:
    for _ in range(warmups):
        fn()
    samples: list[float] = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) * 1e-9)
    return summarize(samples)


def directory_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def covariance_factor(covariance: np.ndarray) -> np.ndarray:
    evals, evecs = np.linalg.eigh(0.5 * (covariance + covariance.T))
    if float(evals.min()) <= 0:
        raise ValueError("covariance must be positive definite")
    return np.sqrt(evals)[:, None] * evecs.T


def make_problem_data(
    *, n_assets: int = 24, n_horizons: int = 8, seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate deterministic data with exact frictionless Sharpe 3 -> 1."""
    rng = np.random.default_rng(seed)
    n_factors = min(5, n_assets)
    loadings = rng.normal(size=(n_assets, n_factors))
    raw_factor = loadings @ loadings.T / n_factors
    raw_factor /= np.sqrt(np.outer(np.diag(raw_factor), np.diag(raw_factor)))
    corr = 0.65 * raw_factor + 0.35 * np.eye(n_assets)
    annual_vol = rng.uniform(0.12, 0.35, size=n_assets)
    daily_vol = annual_vol / np.sqrt(PERIODS_PER_YEAR)
    covariance = corr * np.outer(daily_vol, daily_vol)
    covariance = 0.5 * (covariance + covariance.T)
    factor = covariance_factor(covariance)

    target_sharpes = np.linspace(3.0, 1.0, n_horizons)
    inv_cov = np.linalg.inv(covariance)
    directions = np.empty((n_horizons, n_assets), dtype=np.float64)
    directions[0] = rng.normal(size=n_assets)
    for t in range(1, n_horizons):
        directions[t] = (
            0.80 * directions[t - 1]
            + np.sqrt(1.0 - 0.80**2) * rng.normal(size=n_assets)
        )

    expected_returns = np.empty_like(directions)
    realized_sharpes = np.empty(n_horizons, dtype=np.float64)
    for t, target in enumerate(target_sharpes):
        direction = directions[t]
        unit_sr = np.sqrt(direction @ inv_cov @ direction) * np.sqrt(PERIODS_PER_YEAR)
        expected_returns[t] = direction * (target / unit_sr)
        realized_sharpes[t] = (
            np.sqrt(expected_returns[t] @ inv_cov @ expected_returns[t])
            * np.sqrt(PERIODS_PER_YEAR)
        )

    half_spread_bps = rng.uniform(0.5, 2.5, size=n_assets)
    half_spread = np.broadcast_to(
        half_spread_bps[None, :] * 1e-4, (n_horizons, n_assets)
    ).copy()
    current_weights = rng.normal(scale=0.01, size=n_assets)
    risk_variance = np.full(
        n_horizons, ANNUAL_RISK**2 / PERIODS_PER_YEAR, dtype=np.float64
    )
    risk_radius = np.sqrt(risk_variance)
    risk_factors = np.broadcast_to(
        factor[None, :, :], (n_horizons, n_assets, n_assets)
    ).copy()
    return {
        "expected_returns": expected_returns,
        "half_spread": half_spread,
        "half_spread_bps": half_spread_bps,
        "current_weights": current_weights,
        "risk_variance": risk_variance,
        "risk_radius": risk_radius,
        "risk_factors": risk_factors,
        "covariance": covariance,
        "target_sharpes": target_sharpes,
        "realized_sharpes": realized_sharpes,
    }


@dataclass
class CvxpyMPO:
    problem: cp.Problem
    weights: cp.Variable
    turnover_aux: cp.Variable
    expected_returns: cp.Parameter
    half_spread: cp.Parameter
    current_weights: cp.Parameter
    risk_radius: cp.Parameter
    risk_factors: list[cp.Parameter]

    @property
    def q_b_parameter_names(self) -> list[str]:
        return ["expected_returns", "current_weights", "risk_radius"]

    @property
    def all_parameter_names(self) -> list[str]:
        return [
            "expected_returns",
            "half_spread",
            "current_weights",
            "risk_radius",
            *[p.name() for p in self.risk_factors],
        ]


def build_problem(n_horizons: int, n_assets: int) -> CvxpyMPO:
    weights = cp.Variable((n_horizons, n_assets), name="weights")
    turnover_aux = cp.Variable((n_horizons, n_assets), name="turnover_aux")
    expected_returns = cp.Parameter(
        (n_horizons, n_assets), name="expected_returns"
    )
    half_spread = cp.Parameter(
        (n_horizons, n_assets), nonneg=True, name="half_spread"
    )
    current_weights = cp.Parameter(n_assets, name="current_weights")
    risk_radius = cp.Parameter(n_horizons, nonneg=True, name="risk_radius")
    risk_factors = [
        cp.Parameter((n_assets, n_assets), name=f"risk_factor_{t}")
        for t in range(n_horizons)
    ]

    constraints: list[cp.Constraint] = [turnover_aux >= 0]
    for t in range(n_horizons):
        previous = current_weights if t == 0 else weights[t - 1]
        delta = weights[t] - previous
        constraints.extend(
            [
                delta <= turnover_aux[t],
                -delta <= turnover_aux[t],
                cp.SOC(risk_radius[t], risk_factors[t] @ weights[t]),
            ]
        )

    objective = cp.Minimize(
        -cp.sum(cp.multiply(expected_returns, weights))
        + cp.sum(cp.multiply(half_spread, turnover_aux))
    )
    problem = cp.Problem(objective, constraints)
    if not problem.is_dcp(dpp=True):
        raise RuntimeError("MPO formulation is not DPP-compliant")
    return CvxpyMPO(
        problem=problem,
        weights=weights,
        turnover_aux=turnover_aux,
        expected_returns=expected_returns,
        half_spread=half_spread,
        current_weights=current_weights,
        risk_radius=risk_radius,
        risk_factors=risk_factors,
    )


def assign_data(model: CvxpyMPO, data: dict[str, np.ndarray]) -> None:
    model.expected_returns.value = data["expected_returns"]
    model.half_spread.value = data["half_spread"]
    model.current_weights.value = data["current_weights"]
    model.risk_radius.value = data["risk_radius"]
    for parameter, value in zip(model.risk_factors, data["risk_factors"], strict=True):
        parameter.value = value


def update_data(
    model: CvxpyMPO,
    base: dict[str, np.ndarray],
    iteration: int,
    *,
    update_factors: bool,
) -> None:
    """Apply small deterministic changes while preserving feasibility."""
    phase = float(iteration + 1)
    model.expected_returns.value = base["expected_returns"] * (
        1.0 + 2e-4 * np.sin(phase)
    )
    model.current_weights.value = base["current_weights"] + 1e-5 * np.cos(
        phase + np.arange(base["current_weights"].size)
    )
    model.risk_radius.value = base["risk_radius"] * (
        1.0 + 1e-4 * np.cos(phase)
    )
    if update_factors:
        scales = 1.0 + 1e-4 * np.sin(
            phase + np.arange(base["risk_factors"].shape[0])
        )
        for parameter, factor, scale in zip(
            model.risk_factors, base["risk_factors"], scales, strict=True
        ):
            parameter.value = factor * scale


def evaluate_solution(model: CvxpyMPO) -> dict[str, object]:
    weights = np.asarray(model.weights.value, dtype=np.float64)
    previous = np.vstack([model.current_weights.value, weights[:-1]])
    delta = weights - previous
    factors = np.asarray([p.value for p in model.risk_factors])
    covariance = np.einsum("hji,hjk->hik", factors, factors)
    risk_variance = np.einsum("hi,hij,hj->h", weights, covariance, weights)
    expected_return = np.einsum("hi,hi->h", model.expected_returns.value, weights)
    transaction_cost = np.sum(np.abs(delta) * model.half_spread.value, axis=1)
    objective = float(np.sum(expected_return - transaction_cost))
    return {
        "objective": objective,
        "risk_variance": risk_variance.tolist(),
        "risk_limit": np.square(model.risk_radius.value).tolist(),
        "max_risk_ratio": float(
            np.max(risk_variance / np.square(model.risk_radius.value))
        ),
        "turnover": float(np.sum(np.abs(delta))),
        "expected_return": float(np.sum(expected_return)),
        "transaction_cost": float(np.sum(transaction_cost)),
    }


def solve_cvxpy(model: CvxpyMPO) -> float:
    value = model.problem.solve(
        solver=cp.CLARABEL,
        verbose=False,
        warm_start=False,
        max_iter=200,
        tol_gap_abs=1e-8,
        tol_gap_rel=1e-8,
        tol_feas=1e-8,
    )
    if model.problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"Clarabel failed with status {model.problem.status}")
    return float(value)


def solve_cpg(model: CvxpyMPO, updated_params: list[str]) -> float:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        value = model.problem.solve(
            method="CPG",
            updated_params=updated_params,
            verbose=False,
            max_iters=200,
            tol_gap_abs=1e-8,
            tol_gap_rel=1e-8,
            tol_feas=1e-8,
        )
    if model.problem.status not in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}:
        raise RuntimeError(f"CVXPYgen Clarabel failed with status {model.problem.status}")
    return float(value)


def benchmark_changing(
    model: CvxpyMPO,
    base: dict[str, np.ndarray],
    solve: Callable[[], float],
    *,
    update_factors: bool,
    warmups: int,
    runs: int,
) -> TimingSummary:
    counter = 0

    def one() -> float:
        nonlocal counter
        update_data(model, base, counter, update_factors=update_factors)
        counter += 1
        return solve()

    return benchmark(one, warmups=warmups, runs=runs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assets", type=int, default=24)
    parser.add_argument("--horizons", type=int, default=8)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--cvxpy-runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--code-dir", type=Path, default=Path("mpo_cvxpygen_code"))
    parser.add_argument("--json-out", type=Path, default=Path("mpo_cvxpygen_results.json"))
    args = parser.parse_args()

    data = make_problem_data(n_assets=args.assets, n_horizons=args.horizons)
    model = build_problem(args.horizons, args.assets)
    assign_data(model, data)

    cvxpy_initial_value = solve_cvxpy(model)
    cvxpy_initial_solution = evaluate_solution(model)
    cvxpy_same = benchmark(
        lambda: solve_cvxpy(model), warmups=args.warmups, runs=args.cvxpy_runs
    )
    assign_data(model, data)
    cvxpy_q_b = benchmark_changing(
        model,
        data,
        lambda: solve_cvxpy(model),
        update_factors=False,
        warmups=args.warmups,
        runs=args.cvxpy_runs,
    )
    assign_data(model, data)
    cvxpy_all = benchmark_changing(
        model,
        data,
        lambda: solve_cvxpy(model),
        update_factors=True,
        warmups=args.warmups,
        runs=args.cvxpy_runs,
    )

    if args.code_dir.exists():
        shutil.rmtree(args.code_dir)
    assign_data(model, data)
    generation_start = time.perf_counter()
    cpg.generate_code(
        model.problem,
        code_dir=str(args.code_dir),
        solver="CLARABEL",
        enable_settings=[
            "verbose",
            "max_iter",
            "tol_gap_abs",
            "tol_gap_rel",
            "tol_feas",
        ],
        wrapper=True,
    )
    generation_seconds = time.perf_counter() - generation_start

    assign_data(model, data)
    cpg_initial_value = solve_cpg(model, model.all_parameter_names)
    cpg_initial_solution = evaluate_solution(model)
    cpg_same = benchmark(
        lambda: solve_cpg(model, []), warmups=args.warmups, runs=args.runs
    )
    assign_data(model, data)
    cpg_q_b = benchmark_changing(
        model,
        data,
        lambda: solve_cpg(model, model.q_b_parameter_names),
        update_factors=False,
        warmups=args.warmups,
        runs=args.runs,
    )
    assign_data(model, data)
    cpg_all = benchmark_changing(
        model,
        data,
        lambda: solve_cpg(model, model.all_parameter_names),
        update_factors=True,
        warmups=args.warmups,
        runs=args.runs,
    )

    assign_data(model, data)
    cvxpy_reference = solve_cvxpy(model)
    cvxpy_weights = np.asarray(model.weights.value).copy()
    cpg_reference = solve_cpg(model, model.all_parameter_names)
    cpg_weights = np.asarray(model.weights.value).copy()

    generated_files = sorted(
        str(path.relative_to(args.code_dir))
        for path in args.code_dir.rglob("*")
        if path.is_file()
    )
    cvxpygen_module = sys.modules.get("cvxpygen")
    results = {
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "numpy": np.__version__,
            "cvxpy": cp.__version__,
            "cvxpygen": getattr(cvxpygen_module, "__version__", "1.0.0"),
            "clarabel": __import__("clarabel").__version__,
        },
        "problem": {
            "assets": args.assets,
            "horizons": args.horizons,
            "weight_variables": args.assets * args.horizons,
            "total_user_variables": 2 * args.assets * args.horizons,
            "target_sharpes": data["target_sharpes"].tolist(),
            "realized_sharpes": data["realized_sharpes"].tolist(),
            "dpp": model.problem.is_dcp(dpp=True),
        },
        "codegen": {
            "seconds": generation_seconds,
            "directory_bytes": directory_bytes(args.code_dir),
            "file_count": len(generated_files),
            "files": generated_files,
        },
        "cvxpy_clarabel": {
            "initial_objective": cvxpy_initial_value,
            "initial_solution": cvxpy_initial_solution,
            "same_data": asdict(cvxpy_same),
            "q_b_updates": asdict(cvxpy_q_b),
            "all_updates": asdict(cvxpy_all),
        },
        "cvxpygen_clarabel_python": {
            "initial_objective": cpg_initial_value,
            "initial_solution": cpg_initial_solution,
            "same_data": asdict(cpg_same),
            "q_b_updates": asdict(cpg_q_b),
            "all_updates": asdict(cpg_all),
        },
        "agreement": {
            "objective_abs_diff": float(abs(cvxpy_reference - cpg_reference)),
            "weights_max_abs_diff": float(np.max(np.abs(cvxpy_weights - cpg_weights))),
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
