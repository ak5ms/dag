from __future__ import annotations

from collections import Counter
import json
import os
from statistics import mean, median

from flows.gp import make_pset, random_formula
from flows.riskminer.cpp_stream_eval import build_candidate_score_formula
from trading_dsl_engine.cpp_stream import compile_formula


SAMPLES = int(os.environ.get("GP_PARALLEL_AUDIT_SAMPLES", "24"))
MIN_DEPTH = int(os.environ.get("GP_PARALLEL_AUDIT_MIN_DEPTH", "1"))
MAX_DEPTH = int(os.environ.get("GP_PARALLEL_AUDIT_MAX_DEPTH", "3"))
N = int(os.environ.get("GP_PARALLEL_AUDIT_INSTRUMENTS", "9"))


def _reason_key(reason: str) -> str:
    if "row reduction removes the instrument axis" in reason:
        return "stateful_prefix_then_cross_sectional_reduction"
    if "operator couples instrument lanes" in reason:
        return "cross_sectional_operator"
    if "einsum contracts or permutes" in reason:
        return "cross_lane_einsum"
    if "groupby inner plan couples" in reason:
        return "cross_lane_groupby"
    if "output does not retain" in reason:
        return "non_partitionable_output"
    if "all rows are independent" in reason:
        return "row_independent"
    if "instrument lanes" in reason:
        return "lane_independent"
    return reason.split(";", 1)[0]


def _record(kind: str, seed: int, tree, runtime) -> dict[str, object]:
    return {
        "kind": kind,
        "seed": seed,
        "depth": tree.height,
        "tree": str(tree),
        "mode": runtime.parallel_plan.mode,
        "reason": runtime.parallel_plan.reason,
        "reason_key": _reason_key(runtime.parallel_plan.reason),
        "auto_multicore": runtime.parallel_plan.auto_multicore,
        "work_score": runtime.parallel_plan.work_score,
        "stages": [stage.kind for stage in runtime.plan.stages],
        "output_mode": runtime.plan.output_mode,
        "output_shape": runtime.plan.output_shape,
    }


def main() -> None:
    if SAMPLES <= 0 or N <= 0:
        raise ValueError("samples and instruments must be positive")
    if MIN_DEPTH < 0 or MAX_DEPTH < MIN_DEPTH:
        raise ValueError("require 0 <= min depth <= max depth")

    pset = make_pset()
    records: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for seed in range(SAMPLES):
        tree, alpha = random_formula(
            pset,
            min_depth=MIN_DEPTH,
            max_depth=MAX_DEPTH,
            seed=seed,
        )
        for kind, formula in (
            ("alpha", alpha),
            ("candidate_sharpe", build_candidate_score_formula([alpha])),
        ):
            try:
                runtime = compile_formula(
                    formula,
                    n_instruments=N,
                    default_group_capacity=365 * 15,
                    prefetch_rows=16,
                )
                record = _record(kind, seed, tree, runtime)
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
            except Exception as exc:
                failures.append(
                    {
                        "kind": kind,
                        "seed": seed,
                        "depth": tree.height,
                        "tree": str(tree),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    if failures:
        raise RuntimeError(json.dumps({"failures": failures}, indent=2, sort_keys=True))

    by_kind: dict[str, dict[str, object]] = {}
    for kind in ("alpha", "candidate_sharpe"):
        subset = [item for item in records if item["kind"] == kind]
        modes = Counter(str(item["mode"]) for item in subset)
        reasons = Counter(str(item["reason_key"]) for item in subset)
        scores = [int(item["work_score"]) for item in subset]
        by_kind[kind] = {
            "samples": len(subset),
            "modes": dict(sorted(modes.items())),
            "reason_keys": dict(reasons.most_common()),
            "mean_work_score": mean(scores),
            "median_work_score": median(scores),
            "min_work_score": min(scores),
            "max_work_score": max(scores),
        }

    summary = {
        "samples": SAMPLES,
        "depth": [MIN_DEPTH, MAX_DEPTH],
        "instruments": N,
        "by_kind": by_kind,
    }
    print(json.dumps({"summary": summary}, sort_keys=True), flush=True)

    if len(records) != 2 * SAMPLES:
        raise RuntimeError("parallel audit did not produce every requested record")
    if any(int(item["work_score"]) <= 0 for item in records):
        raise RuntimeError("planner emitted a non-positive work score")
    unsafe = [
        item
        for item in records
        if item["kind"] == "candidate_sharpe" and item["mode"] != "serial"
    ]
    if unsafe:
        raise RuntimeError(
            "candidate Sharpe was incorrectly marked partitionable:\n"
            + json.dumps(unsafe, indent=2, sort_keys=True)
        )


if __name__ == "__main__":
    main()
