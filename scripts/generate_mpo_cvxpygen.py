from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

import cvxpy as cp
import numpy as np
from cvxpygen import cpg


def parse_size(value: str) -> tuple[int, int]:
    try:
        assets_text, horizons_text = value.lower().split("x", 1)
        n_assets = int(assets_text)
        n_horizons = int(horizons_text)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("sizes must use ASSETSxHORIZONS, e.g. 24x8") from exc
    if n_assets <= 0 or n_horizons <= 0:
        raise argparse.ArgumentTypeError("assets and horizons must be positive")
    return n_assets, n_horizons


def build_problem(n_assets: int, n_horizons: int):
    """Build a DPP CVXPY formulation of the multi-period optimizer.

    The absolute turnover cost is represented with an auxiliary nonnegative
    variable. Besides matching the Moreau conic formulation exactly, this avoids
    multiplying a parameter by a nonlinear expression that itself contains the
    current-weight parameter, which would violate CVXPY's DPP rules.
    """
    expected_returns = cp.Parameter((n_horizons, n_assets), name="expected_returns")
    half_spread = cp.Parameter((n_horizons, n_assets), nonneg=True, name="half_spread")
    current_weights = cp.Parameter(n_assets, name="current_weights")
    risk_factor = cp.Parameter((n_horizons * n_assets, n_assets), name="risk_factor")
    risk_radius = cp.Parameter(n_horizons, nonneg=True, name="risk_radius")

    weights = cp.Variable((n_horizons, n_assets), name="weights")
    turnover = cp.Variable((n_horizons, n_assets), nonneg=True, name="turnover")
    previous = cp.vstack(
        [cp.reshape(current_weights, (1, n_assets), order="C"), weights[:-1, :]]
    )
    delta = weights - previous

    objective = cp.Maximize(
        cp.sum(cp.multiply(expected_returns, weights))
        - cp.sum(cp.multiply(half_spread, turnover))
    )
    constraints = [turnover >= delta, turnover >= -delta]
    constraints.extend(
        cp.norm(
            risk_factor[t * n_assets : (t + 1) * n_assets, :] @ weights[t, :],
            2,
        )
        <= risk_radius[t]
        for t in range(n_horizons)
    )
    problem = cp.Problem(objective, constraints)
    if not problem.is_dcp(dpp=True):
        raise RuntimeError("MPO formulation is not DPP-compatible")
    return problem, {
        "expected_returns": expected_returns,
        "half_spread": half_spread,
        "current_weights": current_weights,
        "risk_factor": risk_factor,
        "risk_radius": risk_radius,
        "weights": weights,
        "turnover": turnover,
    }


def deterministic_parameters(n_assets: int, n_horizons: int, seed: int = 42) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed + 1000 * n_assets + n_horizons)
    loadings = rng.normal(size=(n_assets, min(5, n_assets)))
    covariance = loadings @ loadings.T / max(loadings.shape[1], 1)
    covariance += np.diag(rng.uniform(0.2, 0.8, size=n_assets))
    covariance *= 1e-4 / float(np.mean(np.diag(covariance)))
    evals, evecs = np.linalg.eigh(covariance)
    factor = np.sqrt(np.maximum(evals, 1e-12))[:, None] * evecs.T

    directions = rng.normal(size=(n_horizons, n_assets))
    expected_returns = directions * 2e-4 / np.maximum(
        np.linalg.norm(directions, axis=1, keepdims=True), 1e-12
    )
    half_spread = np.broadcast_to(
        rng.uniform(0.5, 2.5, size=n_assets) * 1e-4,
        (n_horizons, n_assets),
    ).copy()
    current_weights = rng.normal(scale=0.01, size=n_assets)
    risk_factor = np.vstack([factor for _ in range(n_horizons)])
    risk_radius = np.full(n_horizons, 0.10 / np.sqrt(252.0))
    return {
        "expected_returns": expected_returns,
        "half_spread": half_spread,
        "current_weights": current_weights,
        "risk_factor": risk_factor,
        "risk_radius": risk_radius,
    }


def smoke_test(code_dir: Path, problem, symbols, parameters: dict[str, np.ndarray]) -> dict[str, object]:
    parent = str(code_dir.parent.resolve())
    if parent not in sys.path:
        sys.path.insert(0, parent)
    module_name = f"{code_dir.name}.cpg_solver"
    importlib.invalidate_caches()
    cpg_solver = importlib.import_module(module_name)
    problem.register_solve("CPG", cpg_solver.cpg_solve)
    for name, value in parameters.items():
        symbols[name].value = value
    t0 = time.perf_counter()
    value = problem.solve(method="CPG", updated_params=list(parameters))
    elapsed = time.perf_counter() - t0
    weights = np.asarray(symbols["weights"].value)
    turnover = np.asarray(symbols["turnover"].value)
    if weights.shape != symbols["weights"].shape or not np.all(np.isfinite(weights)):
        raise RuntimeError("generated solver returned invalid weights")
    if turnover.shape != symbols["turnover"].shape or not np.all(np.isfinite(turnover)):
        raise RuntimeError("generated solver returned invalid turnover")
    return {
        "objective": float(value),
        "elapsed_seconds": elapsed,
        "weight_norm": float(np.linalg.norm(weights)),
        "turnover_sum": float(np.sum(turnover)),
        "status": str(problem.status),
    }


def generate_one(output_root: Path, n_assets: int, n_horizons: int) -> dict[str, object]:
    name = f"mpo_n{n_assets}_h{n_horizons}"
    code_dir = output_root / name
    problem, symbols = build_problem(n_assets, n_horizons)
    t0 = time.perf_counter()
    cpg.generate_code(
        problem,
        code_dir=str(code_dir),
        solver="CLARABEL",
        wrapper=True,
        prefix=f"n{n_assets}_h{n_horizons}_",
    )
    generation_seconds = time.perf_counter() - t0
    parameters = deterministic_parameters(n_assets, n_horizons)
    np.savez(output_root / f"{name}_parameters.npz", **parameters)
    smoke = smoke_test(code_dir, problem, symbols, parameters)
    return {
        "name": name,
        "n_assets": n_assets,
        "n_horizons": n_horizons,
        "generation_seconds": generation_seconds,
        "smoke_test": smoke,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=parse_size, default=[(24, 8)])
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [generate_one(args.output_dir, n, h) for n, h in args.sizes]
    metadata = {
        "cvxpy_version": cp.__version__,
        "sizes": results,
    }
    (args.output_dir / "generation_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
