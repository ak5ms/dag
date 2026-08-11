from __future__ import annotations

"""Plug-and-play risk-seeking MCTS over the user's InputData files.

Run from the repository root:

    PYTHONPATH=src python scripts/run_riskminer_inputdata.py

The script exposes exactly ``INPUTDATA_ALPHA_KEYS`` to formula generation.  It
constructs roll returns, volatility, half-spread and tradability separately for
reward evaluation, so those target/evaluation fields cannot leak into an alpha.
All progress is emitted as flushed line-oriented JSON.
"""

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from flows.load import InputData
from flows.pov import RollRets
from flows.riskminer import (
    CppStreamOrthogonalEvaluator,
    CppStreamPoolEvaluator,
    INPUTDATA_ALPHA_KEYS,
    JaxGRUPolicy,
    RewardDensePoolModel,
    RidgeAlphaPool,
    RiskMinerConfig,
    RiskSeekingTrainer,
    TypedRPNEnvironment,
    build_vocabulary,
    inputdata_alpha_terminal_metadata,
    split_sources_contiguous,
)
from flows.riskminer.diagnostics import ConsoleProgress, Heartbeat
from flows.riskminer.semantics import DEFAULT_TYPE_GRAPH, NON_VALUE_TYPES
from flows.utils import replace as dsl_replace, streak
from trading_dsl_engine.base.dsl import ewm, ffill, isnan, var, where
from trading_dsl_engine.cpp_stream import compile_formula


# Moderate workstation defaults. Use RISKMINER_ROWS=0 for every available row.
INPUT_GLOB = os.environ.get("RISKMINER_INPUT_GLOB", "/mnt/extra/qrt/data/aks_out3/*.npy")
ROWS = int(os.environ.get("RISKMINER_ROWS", "100000"))
MAX_DEPTH = int(os.environ.get("RISKMINER_MAX_DEPTH", "8"))
MAX_TOKENS = int(os.environ.get("RISKMINER_MAX_TOKENS", "30"))
ITERATIONS_PER_DEPTH = int(os.environ.get("RISKMINER_ITERATIONS_PER_DEPTH", "1"))
# One valid episode performs an exact validation Ridge-pool trial.  The default
# is intentionally diagnostic; use 200 to reproduce the paper's search budget.
SIMULATIONS = int(os.environ.get("RISKMINER_SIMULATIONS", "8"))
ROLLOUTS = int(os.environ.get("RISKMINER_ROLLOUTS", "1"))
EVALUATION_BATCH = int(os.environ.get("RISKMINER_EVALUATION_BATCH", "8"))
ARCHIVE_SIZE = int(os.environ.get("RISKMINER_ARCHIVE_SIZE", "500"))
POOL_CAPACITY = int(os.environ.get("RISKMINER_POOL_CAPACITY", "100"))
POOL_MIN_IMPROVEMENT = float(os.environ.get("RISKMINER_POOL_MIN_IMPROVEMENT", "1e-8"))
RIDGE_RECOMPUTE_EVERY = int(os.environ.get("RISKMINER_RIDGE_RECOMPUTE_EVERY", "1"))
TRAIN_FRACTION = float(os.environ.get("RISKMINER_TRAIN_FRACTION", "0.70"))
VALIDATION_FRACTION = float(os.environ.get("RISKMINER_VALIDATION_FRACTION", "0.15"))
POLICY_EPOCHS = int(os.environ.get("RISKMINER_POLICY_EPOCHS", "1"))
POLICY_BATCH_SIZE = int(os.environ.get("RISKMINER_POLICY_BATCH_SIZE", "32"))
POLICY_LEARNING_RATE = float(
    os.environ.get("RISKMINER_POLICY_LEARNING_RATE", "0.001")
)
QUANTILE_CDF = float(os.environ.get("RISKMINER_QUANTILE_CDF", "0.80"))
QUANTILE_LEARNING_RATE = float(
    os.environ.get("RISKMINER_QUANTILE_LEARNING_RATE", "0.01")
)
EXPLORATION = float(os.environ.get("RISKMINER_EXPLORATION", "1.25"))
ROLLOUT_END_PROBABILITY = float(
    os.environ.get("RISKMINER_ROLLOUT_END_PROBABILITY", "0.25")
)
REPLAY_CAPACITY = int(os.environ.get("RISKMINER_REPLAY_CAPACITY", "256"))
POOL_IMPORTANCE = os.environ.get("RISKMINER_POOL_IMPORTANCE", "mean_abs")
THREADS = int(os.environ.get("RISKMINER_THREADS", "0"))
SEED = int(os.environ.get("RISKMINER_SEED", "42"))
HEARTBEAT_SECONDS = float(os.environ.get("RISKMINER_HEARTBEAT_SECONDS", "5"))
OUTPUT_DIR = Path(os.environ.get("RISKMINER_OUTPUT_DIR", "/tmp/riskminer-inputdata"))
REUSE_DERIVED = os.environ.get("RISKMINER_REUSE_DERIVED", "0").lower() in {
    "1", "true", "yes", "on",
}
RESUME_POLICY = os.environ.get("RISKMINER_RESUME_POLICY", "").strip()
LOG_LEVEL = os.environ.get("RISKMINER_LOG_LEVEL", "trace").strip().lower()
if LOG_LEVEL not in {"summary", "detail", "trace"}:
    raise ValueError("RISKMINER_LOG_LEVEL must be summary, detail, or trace")
LOG_LEVEL_RANK = {"summary": 0, "detail": 1, "trace": 2}
TRACE_EVENTS = {
    "mcts_node_choice",
    "mcts_selection_edge",
    "mcts_rollout_choice",
    "mcts_rollout_step",
    "mcts_backprop_edge",
}
DETAIL_EVENTS = {
    "mcts_search_start", "mcts_search_done",
    "mcts_simulation_start", "mcts_simulation_done",
    "mcts_selection_done", "mcts_rollout_start", "mcts_rollout_done",
    "mcts_episode_invalid", "mcts_episode_done",
    "mcts_candidates_evaluate", "mcts_candidates_scored",
    "mcts_archive_update", "mcts_terminal_evaluate", "mcts_terminal_result",
    "replay_reset", "replay_snapshot", "replay_quantile_update",
    "policy_train_batch_start", "policy_train_batch_done",
}


def _rows(value) -> int:
    shape = tuple(getattr(value, "shape", ()))
    if not shape:
        raise ValueError(f"source has no row dimension: {type(value).__name__}")
    return int(shape[0])


def _slice_sources(sources: dict[str, object], rows: int) -> dict[str, object]:
    out = {}
    for name, value in sources.items():
        if not hasattr(value, "shape"):
            out[name] = value
            continue
        if _rows(value) < rows:
            raise ValueError(
                f"source {name!r} has {_rows(value):,} rows; requested {rows:,}"
            )
        out[name] = value[:rows]
    return out


def _infer_instruments(sources: dict[str, object]) -> int:
    widths = {
        int(sources[name].shape[1])
        for name in INPUTDATA_ALPHA_KEYS
        if len(tuple(getattr(sources[name], "shape", ()))) >= 2
    }
    if len(widths) != 1:
        raise ValueError(f"alpha fields have inconsistent widths: {sorted(widths)}")
    return next(iter(widths))


def _validate_npy(path: Path, rows: int, instruments: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        valid = tuple(value.shape) == (rows, instruments) and value.dtype == np.float64
        del value
        return valid
    except Exception:
        return False


def _run_kwargs() -> dict[str, object]:
    return {"threads": THREADS} if THREADS > 0 else {}


def _materialize(
    name: str,
    formula,
    sources: dict[str, object],
    path: Path,
    *,
    rows: int,
    instruments: int,
    progress: ConsoleProgress,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if REUSE_DERIVED and _validate_npy(path, rows, instruments):
        progress.emit("derived_reused", name=name, path=str(path))
        return path
    progress.emit("derived_compile_start", name=name)
    started = time.perf_counter()
    runtime = compile_formula(formula, sources, n_instruments=instruments)
    progress.emit(
        "derived_compile_done",
        name=name,
        compile_seconds=round(time.perf_counter() - started, 6),
        runtime_type=f"{type(runtime).__module__}.{type(runtime).__name__}",
    )
    progress.emit("derived_run_start", name=name, rows=rows, path=str(path))
    with Heartbeat(
        progress,
        "derived_still_running",
        interval_seconds=HEARTBEAT_SECONDS,
        payload={"name": name, "rows": rows},
    ):
        started = time.perf_counter()
        result = runtime.run(out_path=path, **_run_kwargs())
    progress.emit(
        "derived_run_done",
        name=name,
        wall_seconds=round(time.perf_counter() - started, 6),
        native_seconds=getattr(result, "seconds", None),
        output_shape=list(getattr(result, "output_shape", ())),
    )
    if not _validate_npy(path, rows, instruments):
        raise RuntimeError(f"derived {name!r} did not create {(rows, instruments)} npy")
    return path


def _vol_formula():
    """Mirror the current flows.riskmodel session/gap volatility definition."""
    roll_rets = var("roll_rets")
    ev_ts = var("_ev_ts")
    ev_ts_ffill = ffill(ev_ts) + streak(isnan(ev_ts)) * 60e6
    session_start = ffill(var("session_start0"))
    session_end = ffill(var("session_end0"))
    in_session = (session_start < ev_ts_ffill) & (ev_ts_ffill <= session_end)
    gap = dsl_replace(where(~in_session, roll_rets, float("nan")), 0, float("nan"))
    session = dsl_replace(where(in_session, roll_rets, float("nan")), 0, float("nan"))
    session_vol = ewm(session**2, 1440, ignore_na=True, adjust=True) ** 0.5
    gap_vol = ewm(gap**2, 5, ignore_na=True, adjust=True) ** 0.5
    return where(
        (session_start <= ev_ts_ffill)
        & (ev_ts_ffill < session_end - 60e6 * 10),
        session_vol,
        gap_vol,
    )


def _semantic_summary(terminals) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name, info in terminals.items():
        useful = sorted(DEFAULT_TYPE_GRAPH.closure(info.types) - NON_VALUE_TYPES)
        groups.setdefault("/".join(useful) or "unclassified", []).append(name)
    return groups


def _pool_tree(pool: RidgeAlphaPool) -> str:
    lines = [f"RidgePool(root) score={pool.score:.8g}"]
    if not pool.entries:
        lines.append("  <empty>")
    for index, entry in enumerate(pool.entries, 1):
        lines.extend(
            (
                f"  alpha_{index:03d} depth={entry.depth} orthogonal_score={entry.individual_score:.8g}",
                f"    rpn: {entry.rpn}",
            )
        )
    return "\n".join(lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ConsoleProgress(prefix="riskminer-inputdata")

    def event(event_name: str, payload) -> None:
        required = (
            2 if event_name in TRACE_EVENTS
            else 1 if event_name in DETAIL_EVENTS
            else 0
        )
        if LOG_LEVEL_RANK[LOG_LEVEL] >= required:
            progress.emit(event_name, **dict(payload))

    progress.emit(
        "start",
        input_glob=INPUT_GLOB,
        rows_requested=(ROWS if ROWS > 0 else "all"),
        depth_sequence=list(range(1, MAX_DEPTH + 1)),
        max_tokens=MAX_TOKENS,
        iterations_per_depth=ITERATIONS_PER_DEPTH,
        simulations=SIMULATIONS,
        rollouts=ROLLOUTS,
        maximum_exact_pool_trials=(
            MAX_DEPTH * ITERATIONS_PER_DEPTH * SIMULATIONS * ROLLOUTS
        ),
        pool_capacity=POOL_CAPACITY,
        pool_importance=POOL_IMPORTANCE,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
        ridge_recompute_every=RIDGE_RECOMPUTE_EVERY,
        log_level=LOG_LEVEL,
    )

    started = time.perf_counter()
    input_data = InputData(fp=INPUT_GLOB, nrows=None, idx=None)
    raw = input_data.get_data()
    missing = sorted(set(INPUTDATA_ALPHA_KEYS) - set(raw))
    if missing:
        raise KeyError("InputData is missing alpha fields: " + ", ".join(missing))
    available = min(_rows(value) for value in raw.values() if hasattr(value, "shape"))
    rows = available if ROWS <= 0 else min(ROWS, available)
    sources = _slice_sources(raw, rows)
    instruments = _infer_instruments(sources)
    progress.emit(
        "inputdata_ready",
        load_seconds=round(time.perf_counter() - started, 6),
        fields=len(sources),
        alpha_fields=len(INPUTDATA_ALPHA_KEYS),
        rows=rows,
        instruments=instruments,
    )

    terminals = inputdata_alpha_terminal_metadata()
    vocabulary = build_vocabulary(terminals=terminals)
    progress.emit(
        "vocabulary_ready",
        terminal_count=len(terminals),
        token_count=len(vocabulary),
        semantic_families=_semantic_summary(terminals),
    )

    derived_dir = OUTPUT_DIR / "derived"
    roll_rets_path = _materialize(
        "roll_rets",
        RollRets().roll_rets(),
        sources,
        derived_dir / "roll_rets.npy",
        rows=rows,
        instruments=instruments,
        progress=progress,
    )
    with_roll_rets = dict(sources)
    with_roll_rets["roll_rets"] = roll_rets_path
    vol_path = _materialize(
        "vol",
        _vol_formula(),
        with_roll_rets,
        derived_dir / "vol.npy",
        rows=rows,
        instruments=instruments,
        progress=progress,
    )

    evaluation_sources = dict(sources)
    evaluation_sources.update(
        {
            "roll_rets": roll_rets_path,
            "vol": vol_path,
            "hs": sources["vw_halfspread_out0"],
            "is_tradable": sources["is_tradable_out0"],
        }
    )
    train, validation, test = split_sources_contiguous(
        evaluation_sources,
        train_fraction=TRAIN_FRACTION,
        validation_fraction=VALIDATION_FRACTION,
    )
    progress.emit(
        "splits_ready",
        train=[train.start, train.stop],
        validation=[validation.start, validation.stop],
        test=[test.start, test.stop],
    )

    base_config = RiskMinerConfig(
        max_depth=MAX_DEPTH,
        min_formula_depth=1,
        max_tokens=MAX_TOKENS,
        max_stack=8,
        simulations=SIMULATIONS,
        rollouts_per_expansion=ROLLOUTS,
        evaluation_batch_size=EVALUATION_BATCH,
        archive_size=ARCHIVE_SIZE,
        exploration=EXPLORATION,
        rollout_end_probability=ROLLOUT_END_PROBABILITY,
        invalid_reward=-5.0,
        replay_capacity=REPLAY_CAPACITY,
        policy_train_epochs=POLICY_EPOCHS,
        policy_batch_size=POLICY_BATCH_SIZE,
        policy_learning_rate=POLICY_LEARNING_RATE,
        quantile_cdf=QUANTILE_CDF,
        quantile_learning_rate=QUANTILE_LEARNING_RATE,
        pool_capacity=POOL_CAPACITY,
        pool_min_improvement=POOL_MIN_IMPROVEMENT,
        seed=SEED,
    )
    run_kwargs = _run_kwargs()
    intermediate = CppStreamOrthogonalEvaluator(
        train.sources,
        n_instruments=instruments,
        work_dir=OUTPUT_DIR / "intermediate",
        batch_size=EVALUATION_BATCH,
        run_kwargs=run_kwargs,
        on_event=event,
    )
    validation_evaluator = CppStreamPoolEvaluator(
        validation.sources,
        n_instruments=instruments,
        work_dir=OUTPUT_DIR / "validation_pool",
        run_kwargs=run_kwargs,
        on_event=event,
    )
    pool_kwargs = {
        "ridge_recompute_every": RIDGE_RECOMPUTE_EVERY,
    }
    pool = RidgeAlphaPool(
        validation_evaluator,
        capacity=POOL_CAPACITY,
        min_improvement=POOL_MIN_IMPROVEMENT,
        formula_kwargs=pool_kwargs,
        importance=POOL_IMPORTANCE,
    )
    reward_model = RewardDensePoolModel(intermediate, pool, on_event=event)
    resumed_policy = None
    if RESUME_POLICY:
        resumed_policy, metadata = JaxGRUPolicy.load(RESUME_POLICY)
        progress.emit("policy_resumed", path=RESUME_POLICY, metadata=metadata)
    trainer = RiskSeekingTrainer(
        vocabulary_size=len(vocabulary),
        config=base_config,
        policy=resumed_policy,
        initial_token_priors=tuple(
            token.prior for token in vocabulary
        ),
        output_dir=OUTPUT_DIR / "policy",
        on_event=lambda name, payload: progress.emit(name, **payload),
    )

    reports = []
    iteration = 0
    search_started = time.perf_counter()
    for depth in range(1, MAX_DEPTH + 1):
        for depth_iteration in range(1, ITERATIONS_PER_DEPTH + 1):
            iteration += 1
            stage_config = replace(
                base_config,
                max_depth=depth,
                min_formula_depth=depth,
                max_tokens=MAX_TOKENS,
                seed=SEED + 10000 * depth + depth_iteration,
            )
            environment = TypedRPNEnvironment(
                config=stage_config,
                vocabulary=vocabulary,
                target_types=("dimensionless",),
            )
            progress.emit(
                "depth_iteration_start",
                depth=depth,
                depth_iteration=depth_iteration,
                global_iteration=iteration,
                pool_size=len(pool.entries),
                pool_score=(pool.score if math.isfinite(pool.score) else None),
                reward_quantile=trainer.quantile.value,
            )
            with Heartbeat(
                progress,
                "depth_iteration_still_running",
                interval_seconds=HEARTBEAT_SECONDS,
                payload={
                    "depth": depth,
                    "depth_iteration": depth_iteration,
                    "global_iteration": iteration,
                },
            ):
                report = trainer.run_iteration(
                    environment,
                    reward_model,
                    config=stage_config,
                    iteration=iteration,
                )
            reports.append(report)
            progress.emit(
                "depth_iteration_done",
                depth=depth,
                depth_iteration=depth_iteration,
                trajectories=report.search.metrics.trajectories,
                pool_updates=report.search.metrics.pool_updates,
                pool_size=len(pool.entries),
                pool_score=(pool.score if math.isfinite(pool.score) else None),
                quantile=trainer.quantile.value,
                best_archive_score=(
                    report.search.archive[0].score if report.search.archive else None
                ),
            )
            print(_pool_tree(pool), flush=True)

    test_evaluation = None
    if pool.entries:
        progress.emit("final_test_start", pool_size=len(pool.entries))
        test_evaluator = CppStreamPoolEvaluator(
            test.sources,
            n_instruments=instruments,
            work_dir=OUTPUT_DIR / "test_pool",
            run_kwargs=run_kwargs,
            on_event=event,
        )
        with Heartbeat(
            progress,
            "final_test_still_running",
            interval_seconds=HEARTBEAT_SECONDS,
            payload={"pool_size": len(pool.entries)},
        ):
            test_evaluation = test_evaluator.evaluate(
                pool.expressions, **pool_kwargs
            )
        progress.emit(
            "final_test_done",
            score=test_evaluation.score,
            compile_seconds=test_evaluation.compile_seconds,
            run_seconds=test_evaluation.run_seconds,
            native_seconds=test_evaluation.native_seconds,
        )

    result = {
        "backend": "trading_dsl_engine.cpp_stream",
        "input_glob": INPUT_GLOB,
        "rows": rows,
        "instruments": instruments,
        "split_rows": {
            "train": [train.start, train.stop],
            "validation": [validation.start, validation.stop],
            "test": [test.start, test.stop],
        },
        "alpha_keys": list(INPUTDATA_ALPHA_KEYS),
        "pool_score_validation": pool.score if math.isfinite(pool.score) else None,
        "pool_score_test": test_evaluation.score if test_evaluation else None,
        "pool": [
            {
                "rpn": entry.rpn,
                "depth": entry.depth,
                "orthogonal_score_at_admission": entry.individual_score,
            }
            for entry in pool.entries
        ],
        "policy_quantile": trainer.quantile.value,
        "config": {
            "max_depth": MAX_DEPTH,
            "max_tokens": MAX_TOKENS,
            "iterations_per_depth": ITERATIONS_PER_DEPTH,
            "simulations": SIMULATIONS,
            "rollouts_per_expansion": ROLLOUTS,
            "evaluation_batch": EVALUATION_BATCH,
            "pool_capacity": POOL_CAPACITY,
            "pool_importance": POOL_IMPORTANCE,
            "quantile_cdf": QUANTILE_CDF,
            "policy_learning_rate": POLICY_LEARNING_RATE,
            "log_level": LOG_LEVEL,
        },
        "iterations": [
            {
                "iteration": report.iteration,
                "trajectories": report.search.metrics.trajectories,
                "pool_updates": report.search.metrics.pool_updates,
                "tree_nodes": report.search.metrics.tree_nodes,
                "intermediate_formula_requests": report.search.metrics.intermediate_formula_requests,
                "quantile_before": report.reward_quantile_before,
                "quantile_after": report.reward_quantile_after,
                "trajectory_quantiles": list(report.trajectory_quantiles),
                "mean_reward": report.mean_trajectory_reward,
                "max_reward": report.max_trajectory_reward,
                "policy_losses": list(report.policy_losses),
                "checkpoint": report.policy_checkpoint,
            }
            for report in reports
        ],
        "search_seconds": time.perf_counter() - search_started,
        "orthogonal_rank_saturation_note": (
            "When the rowwise raw pool matrix reaches full instrument rank, "
            "pinv remains valid but the cross-sectional residual space is zero."
        ),
    }
    report_path = OUTPUT_DIR / "riskminer_inputdata_report.json"
    report_path.write_text(json.dumps(result, indent=2, sort_keys=True))
    progress.emit(
        "done",
        report=str(report_path),
        pool_size=len(pool.entries),
        validation_score=result["pool_score_validation"],
        test_score=result["pool_score_test"],
        search_seconds=result["search_seconds"],
    )
    print("=== FINAL ROOT-LEVEL RIDGE POOL ===", flush=True)
    print(_pool_tree(pool), flush=True)


if __name__ == "__main__":
    main()
