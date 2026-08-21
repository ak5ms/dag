"""Changing-A/q/b benchmark for the cpp_stream CVXPY optimizer node."""
from __future__ import annotations

import argparse
import json
import os
import platform
import time

import cvxpy as cp
import numpy as np

from trading_dsl_engine.cpp_stream.optimizer import (
    CvxpyNodeBuild,
    cvxpy_node,
    solver_metric,
    variable_value,
)


def make_node(n_assets: int, n_horizons: int, *, include_diagnostics: bool):
    outputs = {"weights": variable_value("weights")}
    if include_diagnostics:
        outputs["iterations"] = solver_metric("iterations")
        outputs["objective"] = solver_metric("objective")

    @cvxpy_node(
        outputs=outputs,
        solver_settings={
            "max_iter": 200,
            "tol_gap_abs": 1e-8,
            "tol_gap_rel": 1e-8,
            "tol_feas": 1e-8,
        },
    )
    def node():
        er = cp.Parameter((n_horizons, n_assets), name="er")
        hs = cp.Parameter((n_horizons, n_assets), nonneg=True, name="hs")
        current = cp.Parameter(n_assets, name="current")
        radius = cp.Parameter(n_horizons, nonneg=True, name="radius")
        # One shared factor is deliberately reused across every horizon.
        factor = cp.Parameter((n_assets, n_assets), name="factor")
        weights = cp.Variable((n_horizons, n_assets), name="weights")
        turnover = cp.Variable((n_horizons, n_assets), name="turnover")
        delta = cp.vstack(
            [weights[0] - current, weights[1:] - weights[:-1]]
        )
        constraints = {
            "turnover_pos": turnover >= delta,
            "turnover_neg": turnover >= -delta,
        }
        for horizon in range(n_horizons):
            constraints[f"risk_{horizon}"] = cp.SOC(
                radius[horizon], factor @ weights[horizon]
            )
        problem = cp.Problem(
            cp.Maximize(
                cp.sum(
                    cp.multiply(er, weights)
                    - cp.multiply(hs, turnover)
                )
            ),
            list(constraints.values()),
        )
        return CvxpyNodeBuild(
            problem=problem,
            parameters={
                "er": er,
                "hs": hs,
                "current": current,
                "radius": radius,
                "factor": factor,
            },
            variables={"weights": weights, "turnover": turnover},
            constraints=constraints,
        )

    return node


def make_cases(n_assets: int, n_horizons: int, count: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    cases = []
    for index in range(count):
        loadings = rng.normal(size=(n_assets, min(n_assets, 5)))
        covariance = (
            loadings @ loadings.T / min(n_assets, 5) + np.eye(n_assets) * 0.35
        )
        cases.append(
            {
                "er": rng.normal(scale=2e-4, size=(n_horizons, n_assets)),
                "hs": rng.uniform(0.5, 2.5, size=(n_horizons, n_assets))
                * 1e-4,
                "current": rng.normal(scale=0.01, size=n_assets),
                "radius": np.full(n_horizons, 0.1 / np.sqrt(252.0))
                * (1.0 + index * 1e-5),
                # Factor changes every problem, so canonical A changes every solve.
                "factor": np.linalg.cholesky(covariance).T
                * (1.0 + index * 2e-5),
            }
        )
    return {
        name: np.stack([case[name] for case in cases]) for name in cases[0]
    }


def measure(fn, *, warmups: int, runs: int):
    for _ in range(warmups):
        fn()
    samples = []
    for _ in range(runs):
        started = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - started) * 1e-6)
    values = np.asarray(samples)
    return {
        "runs": runs,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p90_ms": float(np.percentile(values, 90)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--out", default="optimizer_node_benchmark.json")
    args = parser.parse_args()

    result = {
        "environment": {
            "cpu_count": os.cpu_count(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "cvxpy": cp.__version__,
            "clarabel": __import__("clarabel").__version__,
        },
        "regime": "A/q/b changes on every solve",
        "cases": [],
    }
    available = max(1, os.cpu_count() or 1)
    for n_assets in (9, 24, 50):
        n_horizons = 8
        data = make_cases(n_assets, n_horizons, args.batch)
        row = {
            "assets": n_assets,
            "horizons": n_horizons,
            "batch": args.batch,
            "workers": {},
        }
        for workers in sorted({1, 2, 4, min(available, args.batch)}):
            node = make_node(n_assets, n_horizons, include_diagnostics=False)
            compiled = node.compile(workers=workers)
            row["workers"][str(workers)] = measure(
                lambda: compiled.solve_batch(data, workers=workers),
                warmups=args.warmups,
                runs=args.runs,
            )
        base = make_node(n_assets, n_horizons, include_diagnostics=False).compile()
        named = make_node(n_assets, n_horizons, include_diagnostics=True).compile()
        base_time = measure(
            lambda: base.solve_batch(data, workers=1),
            warmups=args.warmups,
            runs=args.runs,
        )
        named_time = measure(
            lambda: named.solve_batch(data, workers=1),
            warmups=args.warmups,
            runs=args.runs,
        )
        row["requested_output_overhead"] = {
            "weights_only": base_time,
            "weights_plus_diagnostics": named_time,
            "mean_overhead_ms": named_time["mean_ms"] - base_time["mean_ms"],
        }
        result["cases"].append(row)
        print(json.dumps(row, indent=2))

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)


if __name__ == "__main__":
    main()
