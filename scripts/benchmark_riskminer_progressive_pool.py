from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import time

from benchmark_riskminer_cpp_stream import generate_synthetic_sources
from flows.riskminer import (
    CppStreamPoolEvaluator,
    RiskMCTS,
    RiskMinerConfig,
    SchemaPriorPolicy,
    TypedRPNEnvironment,
    build_vocabulary,
    canonical_expr_key,
)
from flows.riskminer.diagnostics import (
    ConsoleProgress,
    DiagnosticCppStreamCandidateEvaluator,
    Heartbeat,
    TracingRiskMCTS,
)


ROWS = int(os.environ.get("RISKMINER_ROWS", "2000"))
INSTRUMENTS = int(os.environ.get("RISKMINER_INSTRUMENTS", "9"))
MAX_DEPTH = int(os.environ.get("RISKMINER_MAX_DEPTH", "4"))
SIMULATIONS_PER_DEPTH = int(
    os.environ.get("RISKMINER_SIMULATIONS_PER_DEPTH", "8")
)
ROLLOUTS = int(os.environ.get("RISKMINER_ROLLOUTS", "2"))
EVALUATION_BATCH = int(os.environ.get("RISKMINER_EVALUATION_BATCH", "4"))
ARCHIVE_SIZE = int(os.environ.get("RISKMINER_ARCHIVE_SIZE", "24"))
POOL_SHORTLIST = int(os.environ.get("RISKMINER_POOL_SHORTLIST", "3"))
MIN_POOL_IMPROVEMENT = float(
    os.environ.get("RISKMINER_MIN_POOL_IMPROVEMENT", "0")
)
SEED = int(os.environ.get("RISKMINER_SEED", "42"))
OUTPUT_DIR = os.environ.get("RISKMINER_OUTPUT_DIR")
HEARTBEAT_SECONDS = float(os.environ.get("RISKMINER_HEARTBEAT_SECONDS", "5"))


def _pool_tree(pool_entries) -> str:
    lines = ["RidgePool(root)"]
    if not pool_entries:
        lines.append("  <empty>")
    for index, entry in enumerate(pool_entries, start=1):
        lines.append(
            f"  alpha_{index:02d} depth={entry.depth} "
            f"individual_score={entry.score:.8g}"
        )
        lines.append(f"    rpn: {entry.rpn}")
        lines.append(f"    expr: {entry.expr!r}")
    return "\n".join(lines)


def main() -> None:
    if MAX_DEPTH <= 0:
        raise ValueError("RISKMINER_MAX_DEPTH must be positive")
    if POOL_SHORTLIST <= 0:
        raise ValueError("RISKMINER_POOL_SHORTLIST must be positive")

    temporary = None
    if OUTPUT_DIR:
        root = Path(OUTPUT_DIR)
        root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="riskminer_progressive_pool_")
        root = Path(temporary.name)

    progress = ConsoleProgress(prefix="progressive-pool")
    progress.emit(
        "benchmark_start",
        backend="trading_dsl_engine.cpp_stream",
        rows=ROWS,
        instruments=INSTRUMENTS,
        depth_sequence=list(range(1, MAX_DEPTH + 1)),
        simulations_per_depth=SIMULATIONS_PER_DEPTH,
        rollouts_per_expansion=ROLLOUTS,
        candidate_batch_size=EVALUATION_BATCH,
        pool_shortlist=POOL_SHORTLIST,
    )

    data_started = time.perf_counter()
    sources = generate_synthetic_sources(root / "data")
    progress.emit(
        "data_ready",
        seconds=round(time.perf_counter() - data_started, 6),
        source_count=len(sources),
    )

    pool_entries = []
    pool_keys: set[tuple] = set()
    current_pool_score = -math.inf
    stage_reports = []
    total_started = time.perf_counter()

    base_config = RiskMinerConfig(
        max_depth=MAX_DEPTH,
        min_formula_depth=1,
        max_tokens=max(16, 4 * MAX_DEPTH + 8),
        max_stack=8,
        simulations=SIMULATIONS_PER_DEPTH,
        rollouts_per_expansion=ROLLOUTS,
        evaluation_batch_size=EVALUATION_BATCH,
        archive_size=ARCHIVE_SIZE,
        dense_rewards=False,
        seed=SEED,
    )

    for depth in range(1, MAX_DEPTH + 1):
        stage_started = time.perf_counter()
        stage_config = replace(
            base_config,
            max_depth=depth,
            min_formula_depth=depth,
            max_tokens=max(8, 4 * depth + 6),
            seed=SEED + depth,
        )
        progress.emit(
            "depth_start",
            depth=depth,
            current_pool_size=len(pool_entries),
            current_pool_score=(
                current_pool_score if math.isfinite(current_pool_score) else None
            ),
        )

        environment = TypedRPNEnvironment(
            config=stage_config,
            vocabulary=build_vocabulary(),
            target_types=("dimensionless",),
        )
        candidate_evaluator = DiagnosticCppStreamCandidateEvaluator(
            sources,
            n_instruments=INSTRUMENTS,
            work_dir=root / f"depth_{depth}" / "candidate_outputs",
            batch_size=EVALUATION_BATCH,
            progress=progress,
            heartbeat_seconds=HEARTBEAT_SECONDS,
        )
        mcts = RiskMCTS(
            environment,
            candidate_evaluator,
            config=stage_config,
            policy=SchemaPriorPolicy(),
        )
        search = TracingRiskMCTS(mcts, progress=progress).search()

        shortlist = []
        for entry in search.archive:
            key = canonical_expr_key(entry.expr)
            if key in pool_keys:
                continue
            shortlist.append(entry)
            if len(shortlist) >= POOL_SHORTLIST:
                break

        progress.emit(
            "pool_shortlist_ready",
            depth=depth,
            archive_size=len(search.archive),
            shortlist_size=len(shortlist),
            formulas=[entry.rpn for entry in shortlist],
        )

        trials = []
        for trial_index, entry in enumerate(shortlist, start=1):
            trial_pool = tuple(item.expr for item in pool_entries) + (entry.expr,)
            progress.emit(
                "pool_trial_start",
                depth=depth,
                trial=trial_index,
                trial_count=len(shortlist),
                alpha_count=len(trial_pool),
                candidate_rpn=entry.rpn,
                candidate_depth=entry.depth,
                baseline_score=(
                    current_pool_score if math.isfinite(current_pool_score) else None
                ),
            )
            pool_evaluator = CppStreamPoolEvaluator(
                sources,
                n_instruments=INSTRUMENTS,
                work_dir=(
                    root
                    / f"depth_{depth}"
                    / "pool_trials"
                    / f"trial_{trial_index}"
                ),
            )
            trial_started = time.perf_counter()
            with Heartbeat(
                progress,
                "pool_trial_still_compiling_or_running",
                interval_seconds=HEARTBEAT_SECONDS,
                payload={
                    "depth": depth,
                    "trial": trial_index,
                    "alpha_count": len(trial_pool),
                    "candidate_rpn": entry.rpn,
                },
            ):
                evaluation = pool_evaluator.evaluate(trial_pool)
            delta = (
                evaluation.score - current_pool_score
                if math.isfinite(current_pool_score)
                else evaluation.score
            )
            trial = {
                "entry": entry,
                "score": evaluation.score,
                "delta": delta,
                "compile_seconds": evaluation.compile_seconds,
                "run_seconds": evaluation.run_seconds,
                "native_seconds": evaluation.native_seconds,
                "runtime_type": evaluation.runtime_type,
                "wall_seconds": time.perf_counter() - trial_started,
            }
            trials.append(trial)
            progress.emit(
                "pool_trial_done",
                depth=depth,
                trial=trial_index,
                alpha_count=len(trial_pool),
                candidate_rpn=entry.rpn,
                pool_score=evaluation.score,
                additive_delta=delta,
                compile_seconds=round(evaluation.compile_seconds, 6),
                run_seconds=round(evaluation.run_seconds, 6),
                native_seconds=evaluation.native_seconds,
                runtime_type=evaluation.runtime_type,
            )

        accepted = None
        if trials:
            best = max(
                trials,
                key=lambda item: (
                    item["score"],
                    -item["entry"].depth,
                    item["entry"].rpn,
                ),
            )
            if (
                not pool_entries
                or best["delta"] > MIN_POOL_IMPROVEMENT
            ):
                accepted = best
                pool_entries.append(best["entry"])
                pool_keys.add(canonical_expr_key(best["entry"].expr))
                current_pool_score = float(best["score"])

        progress.emit(
            "depth_done",
            depth=depth,
            accepted=(accepted["entry"].rpn if accepted is not None else None),
            accepted_pool_score=(
                accepted["score"] if accepted is not None else None
            ),
            accepted_delta=(accepted["delta"] if accepted is not None else None),
            pool_size=len(pool_entries),
            current_pool_score=(
                current_pool_score if math.isfinite(current_pool_score) else None
            ),
            stage_seconds=round(time.perf_counter() - stage_started, 6),
        )
        print(_pool_tree(pool_entries), flush=True)

        stage_reports.append(
            {
                "depth": depth,
                "search": {
                    "simulations": search.metrics.simulations,
                    "rollouts": search.metrics.rollouts,
                    "formula_requests": search.metrics.unique_formula_requests,
                    "finite_formula_scores": search.metrics.finite_formula_scores,
                    "invalid_rollouts": search.metrics.invalid_rollouts,
                    "tree_nodes": search.metrics.tree_nodes,
                    "archive_size": len(search.archive),
                    "compile_seconds": candidate_evaluator.summary.compile_seconds,
                    "run_seconds": candidate_evaluator.summary.run_seconds,
                },
                "shortlist": [
                    {
                        "individual_score": entry.score,
                        "depth": entry.depth,
                        "rpn": entry.rpn,
                        "expr": repr(entry.expr),
                    }
                    for entry in shortlist
                ],
                "pool_trials": [
                    {
                        "rpn": trial["entry"].rpn,
                        "depth": trial["entry"].depth,
                        "individual_score": trial["entry"].score,
                        "pool_score": trial["score"],
                        "additive_delta": trial["delta"],
                        "compile_seconds": trial["compile_seconds"],
                        "run_seconds": trial["run_seconds"],
                        "native_seconds": trial["native_seconds"],
                    }
                    for trial in trials
                ],
                "accepted": (
                    {
                        "rpn": accepted["entry"].rpn,
                        "expr": repr(accepted["entry"].expr),
                        "depth": accepted["entry"].depth,
                        "individual_score": accepted["entry"].score,
                        "pool_score": accepted["score"],
                        "additive_delta": accepted["delta"],
                    }
                    if accepted is not None
                    else None
                ),
                "stage_seconds": time.perf_counter() - stage_started,
            }
        )

    report = {
        "backend": "trading_dsl_engine.cpp_stream",
        "rows": ROWS,
        "instruments": INSTRUMENTS,
        "depth_sequence": list(range(1, MAX_DEPTH + 1)),
        "simulations_per_depth": SIMULATIONS_PER_DEPTH,
        "rollouts_per_expansion": ROLLOUTS,
        "candidate_batch_size": EVALUATION_BATCH,
        "pool_shortlist": POOL_SHORTLIST,
        "final_pool_score": (
            current_pool_score if math.isfinite(current_pool_score) else None
        ),
        "final_pool_size": len(pool_entries),
        "final_pool": [
            {
                "depth": entry.depth,
                "individual_score": entry.score,
                "rpn": entry.rpn,
                "expr": repr(entry.expr),
            }
            for entry in pool_entries
        ],
        "stages": stage_reports,
        "total_seconds": time.perf_counter() - total_started,
    }
    result_path = root / "riskminer_progressive_pool.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    progress.emit(
        "benchmark_done",
        final_pool_size=len(pool_entries),
        final_pool_score=report["final_pool_score"],
        total_seconds=round(report["total_seconds"], 6),
        result_json=str(result_path),
    )
    print("=== final root-level combination tree ===", flush=True)
    print(_pool_tree(pool_entries), flush=True)
    print(f"result_json={result_path}", flush=True)

    if temporary is not None:
        temporary.cleanup()


if __name__ == "__main__":
    main()
