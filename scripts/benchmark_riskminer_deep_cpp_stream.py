from __future__ import annotations

from collections import Counter
import json
import os
from pathlib import Path
import tempfile
import time

from benchmark_riskminer_cpp_stream import generate_synthetic_sources
from flows.riskminer import (
    CppStreamCandidateEvaluator,
    RiskMCTS,
    RiskMinerConfig,
    SchemaPriorPolicy,
    TypedRPNEnvironment,
    build_vocabulary,
)


ROWS = int(os.environ.get("RISKMINER_ROWS", "5000"))
INSTRUMENTS = int(os.environ.get("RISKMINER_INSTRUMENTS", "9"))
SIMULATIONS = int(os.environ.get("RISKMINER_SIMULATIONS", "32"))
ROLLOUTS = int(os.environ.get("RISKMINER_ROLLOUTS", "4"))
EVALUATION_BATCH = int(os.environ.get("RISKMINER_EVALUATION_BATCH", "8"))
ARCHIVE_SIZE = int(os.environ.get("RISKMINER_ARCHIVE_SIZE", "100"))
MAX_DEPTH = int(os.environ.get("RISKMINER_MAX_DEPTH", "8"))
MIN_FORMULA_DEPTH = int(os.environ.get("RISKMINER_MIN_FORMULA_DEPTH", "5"))
MAX_TOKENS = int(os.environ.get("RISKMINER_MAX_TOKENS", "28"))
SEED = int(os.environ.get("RISKMINER_SEED", "42"))
OUTPUT_DIR = os.environ.get("RISKMINER_OUTPUT_DIR")
KEEP_DATA = os.environ.get("RISKMINER_KEEP_DATA", "0") == "1"


class DeepTypedRPNEnvironment(TypedRPNEnvironment):
    """Require a nontrivial minimum expression depth before scoring or END."""

    def formula_value(self, state):
        value = super().formula_value(state)
        if value is None or value.depth < self.config.min_formula_depth:
            return None
        return value


def _entry_payload(entry) -> dict[str, object]:
    return {
        "score": float(entry.score),
        "depth": int(entry.depth),
        "rpn": entry.rpn,
        "expr": repr(entry.expr),
    }


def main() -> None:
    if MIN_FORMULA_DEPTH > MAX_DEPTH:
        raise ValueError("RISKMINER_MIN_FORMULA_DEPTH exceeds RISKMINER_MAX_DEPTH")

    temporary = None
    if OUTPUT_DIR:
        root = Path(OUTPUT_DIR)
        root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="riskminer_deep_cpp_stream_")
        root = Path(temporary.name)

    # benchmark_riskminer_cpp_stream reads the same environment variables at
    # import time, so its bounded-memory data generator uses this run's shape.
    data_started = time.perf_counter()
    sources = generate_synthetic_sources(root / "data")
    data_seconds = time.perf_counter() - data_started

    config = RiskMinerConfig(
        max_depth=MAX_DEPTH,
        min_formula_depth=MIN_FORMULA_DEPTH,
        max_tokens=MAX_TOKENS,
        max_stack=8,
        simulations=SIMULATIONS,
        rollouts_per_expansion=ROLLOUTS,
        evaluation_batch_size=EVALUATION_BATCH,
        archive_size=ARCHIVE_SIZE,
        seed=SEED,
    )
    environment = DeepTypedRPNEnvironment(
        config=config,
        vocabulary=build_vocabulary(),
        target_types=("dimensionless",),
    )
    evaluator = CppStreamCandidateEvaluator(
        sources,
        n_instruments=INSTRUMENTS,
        work_dir=root / "candidate_outputs",
        batch_size=EVALUATION_BATCH,
    )

    search_started = time.perf_counter()
    search = RiskMCTS(
        environment,
        evaluator,
        config=config,
        policy=SchemaPriorPolicy(),
    ).search()
    search_wall_seconds = time.perf_counter() - search_started
    entries = search.archive
    if not entries:
        reasons = list(evaluator.summary.rejection_reasons.values())[:10]
        raise RuntimeError(
            "higher-depth RiskMiner search produced no finite native candidates; "
            f"sample rejections={reasons}"
        )

    depth_histogram = Counter(entry.depth for entry in entries)
    deepest = sorted(
        entries,
        key=lambda entry: (-entry.depth, -entry.score, entry.rpn),
    )[:20]
    first_batch = evaluator.summary.batches[0] if evaluator.summary.batches else None
    max_achieved_depth = max(entry.depth for entry in entries)

    report = {
        "backend": "trading_dsl_engine.cpp_stream",
        "rows": ROWS,
        "instruments": INSTRUMENTS,
        "seed": SEED,
        "max_depth": MAX_DEPTH,
        "min_formula_depth": MIN_FORMULA_DEPTH,
        "max_achieved_depth": max_achieved_depth,
        "max_tokens": MAX_TOKENS,
        "simulations": search.metrics.simulations,
        "rollouts_per_expansion": ROLLOUTS,
        "rollouts": search.metrics.rollouts,
        "tree_nodes": search.metrics.tree_nodes,
        "formula_requests": search.metrics.unique_formula_requests,
        "finite_formula_scores": search.metrics.finite_formula_scores,
        "invalid_rollouts": search.metrics.invalid_rollouts,
        "archive_size": len(entries),
        "depth_histogram": {
            str(depth): count
            for depth, count in sorted(depth_histogram.items())
        },
        "data_seconds": data_seconds,
        "search_wall_seconds": search_wall_seconds,
        "candidate_compile_seconds": evaluator.summary.compile_seconds,
        "candidate_run_seconds": evaluator.summary.run_seconds,
        "compile_rejected": evaluator.summary.compile_rejected,
        "nonfinite": evaluator.summary.nonfinite,
        "candidate_runtime_type": (
            first_batch.runtime_type if first_batch is not None else None
        ),
        "candidate_output_shape": (
            first_batch.output_shape if first_batch is not None else None
        ),
        "candidate_native_cache_path": (
            first_batch.cache_path if first_batch is not None else None
        ),
        "top_by_score": [_entry_payload(entry) for entry in entries[:20]],
        "deepest": [_entry_payload(entry) for entry in deepest],
        "sample_rejections": list(
            evaluator.summary.rejection_reasons.values()
        )[:10],
    }
    result_path = root / "riskminer_deep_benchmark.json"
    result_path.write_text(json.dumps(report, indent=2, sort_keys=True))

    print("=== Higher-depth RiskMiner / cpp_stream run ===")
    print("backend=trading_dsl_engine.cpp_stream")
    print(
        f"shape={ROWS:,}x{INSTRUMENTS} min_depth={MIN_FORMULA_DEPTH} "
        f"max_depth={MAX_DEPTH} achieved={max_achieved_depth}"
    )
    print(
        f"simulations={search.metrics.simulations} "
        f"rollouts={search.metrics.rollouts} "
        f"tree_nodes={search.metrics.tree_nodes}"
    )
    print(
        f"formula_requests={search.metrics.unique_formula_requests} "
        f"finite_scores={search.metrics.finite_formula_scores} "
        f"archive={len(entries)} invalid_rollouts={search.metrics.invalid_rollouts}"
    )
    print(f"depth_histogram={dict(sorted(depth_histogram.items()))}")
    print(
        f"data_seconds={data_seconds:.6f} "
        f"search_wall_seconds={search_wall_seconds:.6f} "
        f"cpp_compile_seconds={evaluator.summary.compile_seconds:.6f} "
        f"cpp_run_seconds={evaluator.summary.run_seconds:.6f}"
    )
    if first_batch is not None:
        print(f"runtime_type={first_batch.runtime_type}")
        print(f"output_shape={first_batch.output_shape}")
        print(f"native_cache_path={first_batch.cache_path}")

    print("--- top by score ---")
    for index, entry in enumerate(entries[:20], start=1):
        print(
            f"{index:02d}. score={entry.score:.10g} depth={entry.depth} "
            f"rpn={entry.rpn}"
        )
        print(f"    expr={entry.expr!r}")

    print("--- deepest formulas ---")
    for index, entry in enumerate(deepest, start=1):
        print(
            f"{index:02d}. depth={entry.depth} score={entry.score:.10g} "
            f"rpn={entry.rpn}"
        )
        print(f"    expr={entry.expr!r}")

    print(f"result_json={result_path}")
    if KEEP_DATA:
        print(f"data_directory={root / 'data'}")
        if temporary is not None:
            temporary.cleanup = lambda: None  # type: ignore[method-assign]


if __name__ == "__main__":
    main()
