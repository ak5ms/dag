from __future__ import annotations

"""Plug-and-play RiskMiner search over the user's InputData dataset.

Run from the repository root:

    PYTHONPATH=src python scripts/run_riskminer_inputdata.py

No CLI is required. Optional environment variables are documented in CONFIG
below and printed at startup. InputData may contain arbitrary additional fields;
only INPUTDATA_ALPHA_KEYS are exposed to the alpha grammar. Evaluation-only
roll returns, volatility, half-spread, and tradability are kept out of the
search vocabulary.
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
    CppStreamPoolEvaluator,
    INPUTDATA_ALPHA_KEYS,
    RiskMCTS,
    RiskMinerConfig,
    SchemaPriorPolicy,
    TypedRPNEnvironment,
    build_vocabulary,
    canonical_expr_key,
    inputdata_alpha_terminal_metadata,
)
from flows.riskminer.diagnostics import (
    ConsoleProgress,
    DiagnosticCppStreamCandidateEvaluator,
    Heartbeat,
    TracingRiskMCTS,
)
from flows.riskminer.semantics import DEFAULT_TYPE_GRAPH, NON_VALUE_TYPES
from flows.utils import replace as dsl_replace, streak
from trading_dsl_engine.base.dsl import (
    ewm,
    ffill,
    isnan,
    shift,
    var,
    where,
)
from trading_dsl_engine.cpp_stream import compile_formula


# ---------------------------------------------------------------------------
# CONFIG: environment variables are optional; these defaults are deliberately
# moderate so the script is useful immediately on a workstation.
# RISKMINER_ROWS=0 means use every row from InputData.
# ---------------------------------------------------------------------------
ROWS = int(os.environ.get("RISKMINER_ROWS", "500000"))
MAX_DEPTH = int(os.environ.get("RISKMINER_MAX_DEPTH", "6"))
ROUNDS_PER_DEPTH = int(os.environ.get("RISKMINER_ROUNDS_PER_DEPTH", "1"))
SIMULATIONS = int(os.environ.get("RISKMINER_SIMULATIONS", "64"))
ROLLOUTS = int(os.environ.get("RISKMINER_ROLLOUTS", "4"))
EVALUATION_BATCH = int(os.environ.get("RISKMINER_EVALUATION_BATCH", "16"))
ARCHIVE_SIZE = int(os.environ.get("RISKMINER_ARCHIVE_SIZE", "256"))
POOL_SHORTLIST = int(os.environ.get("RISKMINER_POOL_SHORTLIST", "3"))
TARGET_POOL_SIZE = int(os.environ.get("RISKMINER_TARGET_POOL_SIZE", "12"))
MIN_POOL_IMPROVEMENT = float(
    os.environ.get("RISKMINER_MIN_POOL_IMPROVEMENT", "1e-8")
)
RIDGE_RECOMPUTE_EVERY = int(
    os.environ.get("RISKMINER_RIDGE_RECOMPUTE_EVERY", "1")
)
THREADS = int(os.environ.get("RISKMINER_THREADS", "0"))
SEED = int(os.environ.get("RISKMINER_SEED", "42"))
HEARTBEAT_SECONDS = float(
    os.environ.get("RISKMINER_HEARTBEAT_SECONDS", "5")
)
LOG_TOP = int(os.environ.get("RISKMINER_LOG_TOP", "10"))
OUTPUT_DIR = Path(
    os.environ.get("RISKMINER_OUTPUT_DIR", "/tmp/riskminer-inputdata")
)
REUSE_DERIVED = os.environ.get("RISKMINER_REUSE_DERIVED", "0").lower() in {
    "1",
    "true",
    "yes",
    "on",
}


def _source_rows(value) -> int:
    shape = tuple(getattr(value, "shape", ()))
    if not shape:
        raise ValueError(f"source has no row shape: {type(value).__name__}")
    return int(shape[0])


def _slice_sources(data: dict[str, object], rows: int) -> dict[str, object]:
    if rows <= 0:
        return dict(data)
    out: dict[str, object] = {}
    for name, value in data.items():
        if not hasattr(value, "shape"):
            out[name] = value
            continue
        if _source_rows(value) < rows:
            raise ValueError(
                f"InputData source {name!r} has {_source_rows(value):,} rows; "
                f"requested {rows:,}"
            )
        out[name] = value[:rows]
    return out


def _infer_n_instruments(
    sources: dict[str, object],
    terminal_names: tuple[str, ...],
) -> int:
    widths = set()
    for name in terminal_names:
        value = sources[name]
        shape = tuple(getattr(value, "shape", ()))
        if len(shape) >= 2 and int(shape[1]) > 0:
            widths.add(int(shape[1]))
    if len(widths) != 1:
        raise ValueError(
            "could not infer one instrument width from alpha fields: "
            f"{sorted(widths)}"
        )
    return next(iter(widths))


def _validate_npy(path: Path, *, rows: int, n_instruments: int) -> bool:
    if not path.is_file():
        return False
    try:
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        shape = tuple(value.shape)
        del value
    except Exception:
        return False
    return shape == (rows, n_instruments)


def _materialize_formula(
    *,
    name: str,
    formula,
    sources: dict[str, object],
    out_path: Path,
    rows: int,
    n_instruments: int,
    progress: ConsoleProgress,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if (
        REUSE_DERIVED
        and _validate_npy(
            out_path,
            rows=rows,
            n_instruments=n_instruments,
        )
    ):
        progress.emit(
            "derived_reused",
            name=name,
            path=str(out_path),
            shape=[rows, n_instruments],
        )
        return out_path

    progress.emit("derived_compile_start", name=name)
    compile_started = time.perf_counter()
    runtime = compile_formula(
        formula,
        sources,
        n_instruments=n_instruments,
    )
    compile_seconds = time.perf_counter() - compile_started
    progress.emit(
        "derived_compile_done",
        name=name,
        compile_seconds=round(compile_seconds, 6),
        runtime_type=f"{type(runtime).__module__}.{type(runtime).__name__}",
        generated_cpp=str(getattr(runtime, "generated_cpp", "")),
    )

    progress.emit(
        "derived_run_start",
        name=name,
        rows=rows,
        instruments=n_instruments,
        threads=THREADS,
        path=str(out_path),
    )
    wall_started = time.perf_counter()
    with Heartbeat(
        progress,
        "derived_still_running",
        interval_seconds=HEARTBEAT_SECONDS,
        payload={"name": name, "rows": rows},
    ):
        result = runtime.run(out_path=out_path, threads=THREADS)
    wall_seconds = time.perf_counter() - wall_started
    if not _validate_npy(
        out_path,
        rows=rows,
        n_instruments=n_instruments,
    ):
        raise RuntimeError(
            f"derived {name} output has unexpected shape; expected "
            f"{(rows, n_instruments)!r}: {out_path}"
        )
    progress.emit(
        "derived_run_done",
        name=name,
        wall_seconds=round(wall_seconds, 6),
        native_seconds=getattr(result, "seconds", None),
        output_shape=list(getattr(result, "output_shape", ())),
    )
    return out_path


def _build_vol_from_roll_rets():
    """Same session/gap volatility logic as flows.riskmodel, reusing roll_rets."""

    roll_rets = var("roll_rets")
    ev_ts = var("_ev_ts")
    ev_ts_ffill = ffill(ev_ts) + streak(isnan(ev_ts)) * 60e6
    session_start = ffill(var("session_start0"))
    session_end = ffill(var("session_end0"))
    in_session = (session_start < ev_ts_ffill) & (ev_ts_ffill <= session_end)

    roll_rets_gap = where(~in_session, roll_rets, float("nan"))
    roll_rets_gap = dsl_replace(roll_rets_gap, 0, float("nan"))
    gap_time = shift(streak(isnan(roll_rets_gap)))
    roll_rets_gap_scaled = roll_rets_gap / (gap_time / 1440) ** 0.5
    roll_rets_session = dsl_replace(
        where(in_session, roll_rets, float("nan")),
        0,
        float("nan"),
    )

    vol_session = ewm(
        roll_rets_session**2,
        1440,
        ignore_na=True,
        adjust=True,
    ) ** 0.5
    vol_gap = ewm(
        roll_rets_gap_scaled**2,
        5,
        ignore_na=True,
        adjust=True,
    ) ** 0.5
    return where(
        (session_start <= ev_ts_ffill)
        & (ev_ts_ffill < session_end - 60e6 * 10),
        vol_session,
        vol_gap,
    )


def _semantic_summary(terminals) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for name, info in terminals.items():
        meaningful = sorted(
            DEFAULT_TYPE_GRAPH.closure(info.types) - NON_VALUE_TYPES
        )
        label = "/".join(meaningful) if meaningful else "unclassified"
        groups.setdefault(label, []).append(name)
    return groups


def _pool_tree(pool_entries) -> str:
    lines = ["RidgePool(root)"]
    if not pool_entries:
        return "RidgePool(root)\n  <empty>"
    for index, entry in enumerate(pool_entries, start=1):
        lines.append(
            f"  alpha_{index:03d} depth={entry.depth} "
            f"individual_score={entry.score:.8g}"
        )
        lines.append(f"    rpn: {entry.rpn}")
        lines.append(f"    expr: {entry.expr!r}")
    return "\n".join(lines)


def main() -> None:
    if MAX_DEPTH <= 0 or ROUNDS_PER_DEPTH <= 0:
        raise ValueError("MAX_DEPTH and ROUNDS_PER_DEPTH must be positive")
    if SIMULATIONS <= 0 or ROLLOUTS <= 0:
        raise ValueError("SIMULATIONS and ROLLOUTS must be positive")
    if RIDGE_RECOMPUTE_EVERY < 1:
        raise ValueError("RISKMINER_RIDGE_RECOMPUTE_EVERY must be >= 1")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = ConsoleProgress(prefix="riskminer-inputdata")
    progress.emit(
        "start",
        rows_requested=(ROWS if ROWS > 0 else "all"),
        max_depth=MAX_DEPTH,
        rounds_per_depth=ROUNDS_PER_DEPTH,
        simulations=SIMULATIONS,
        rollouts_per_expansion=ROLLOUTS,
        evaluation_batch=EVALUATION_BATCH,
        archive_size=ARCHIVE_SIZE,
        pool_shortlist=POOL_SHORTLIST,
        target_pool_size=TARGET_POOL_SIZE,
        ridge_recompute_every=RIDGE_RECOMPUTE_EVERY,
        threads=THREADS,
        output_dir=str(OUTPUT_DIR),
    )

    load_started = time.perf_counter()
    # idx=None keeps _ev_ts as the native array and avoids constructing a
    # multi-million-row pandas DatetimeIndex just for search.
    input_data = InputData(nrows=None, idx=None)
    raw_sources = input_data.get_data()
    available_names = tuple(sorted(raw_sources))
    missing_alpha = sorted(set(INPUTDATA_ALPHA_KEYS) - set(raw_sources))
    if missing_alpha:
        raise KeyError(
            "InputData is missing user-approved alpha fields: "
            + ", ".join(missing_alpha)
        )

    available_rows = min(_source_rows(raw_sources[name]) for name in INPUTDATA_ALPHA_KEYS)
    rows = available_rows if ROWS <= 0 else min(ROWS, available_rows)
    sources = _slice_sources(raw_sources, rows)
    n_instruments = _infer_n_instruments(sources, INPUTDATA_ALPHA_KEYS)
    progress.emit(
        "inputdata_ready",
        load_seconds=round(time.perf_counter() - load_started, 6),
        total_inputdata_fields=len(available_names),
        alpha_terminal_fields=len(INPUTDATA_ALPHA_KEYS),
        extra_inputdata_fields=len(set(available_names) - set(INPUTDATA_ALPHA_KEYS)),
        rows=rows,
        instruments=n_instruments,
    )

    terminals = inputdata_alpha_terminal_metadata()
    progress.emit(
        "alpha_semantics",
        families=_semantic_summary(terminals),
    )

    # Evaluation-only data is generated using the same InputData mapping but is
    # not inserted into `terminals`, so MCTS can never use roll_rets/vol/hs as
    # alpha inputs.
    derived_dir = OUTPUT_DIR / "derived"
    roll_rets_path = _materialize_formula(
        name="roll_rets",
        formula=RollRets().roll_rets(),
        sources=sources,
        out_path=derived_dir / "roll_rets.npy",
        rows=rows,
        n_instruments=n_instruments,
        progress=progress,
    )

    evaluation_sources = dict(sources)
    evaluation_sources["roll_rets"] = roll_rets_path
    vol_path = _materialize_formula(
        name="vol",
        formula=_build_vol_from_roll_rets(),
        sources=evaluation_sources,
        out_path=derived_dir / "vol.npy",
        rows=rows,
        n_instruments=n_instruments,
        progress=progress,
    )
    evaluation_sources["vol"] = vol_path
    evaluation_sources["hs"] = sources["vw_halfspread_out0"]
    evaluation_sources["is_tradable"] = sources["is_tradable_out0"]

    pool_entries = []
    pool_keys: set[tuple] = set()
    current_pool_score = -math.inf
    stage_reports = []
    search_started = time.perf_counter()

    base_config = RiskMinerConfig(
        max_depth=MAX_DEPTH,
        min_formula_depth=1,
        max_tokens=max(20, 4 * MAX_DEPTH + 8),
        max_stack=8,
        simulations=SIMULATIONS,
        rollouts_per_expansion=ROLLOUTS,
        evaluation_batch_size=EVALUATION_BATCH,
        archive_size=ARCHIVE_SIZE,
        exploration=1.25,
        progressive_widening_k=4.0,
        progressive_widening_alpha=0.5,
        rollout_end_probability=0.25,
        dense_rewards=False,
        invalid_reward=-5.0,
        seed=SEED,
    )
    vocabulary = build_vocabulary(terminals=terminals)
    progress.emit(
        "vocabulary_ready",
        token_count=len(vocabulary),
        terminal_count=len(terminals),
    )

    for depth in range(1, MAX_DEPTH + 1):
        for round_index in range(1, ROUNDS_PER_DEPTH + 1):
            if len(pool_entries) >= TARGET_POOL_SIZE:
                break

            stage_started = time.perf_counter()
            stage_seed = SEED + depth * 10_000 + round_index
            stage_config = replace(
                base_config,
                max_depth=depth,
                min_formula_depth=depth,
                max_tokens=max(8, 4 * depth + 6),
                seed=stage_seed,
            )
            progress.emit(
                "search_round_start",
                depth=depth,
                round=round_index,
                seed=stage_seed,
                pool_size=len(pool_entries),
                current_pool_score=(
                    current_pool_score
                    if math.isfinite(current_pool_score)
                    else None
                ),
            )

            environment = TypedRPNEnvironment(
                config=stage_config,
                vocabulary=vocabulary,
                target_types=("dimensionless",),
            )
            candidate_evaluator = DiagnosticCppStreamCandidateEvaluator(
                evaluation_sources,
                n_instruments=n_instruments,
                work_dir=(
                    OUTPUT_DIR
                    / f"depth_{depth}"
                    / f"round_{round_index}"
                    / "candidate_outputs"
                ),
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

            top_archive = []
            shortlist = []
            for entry in search.archive:
                if len(top_archive) < LOG_TOP:
                    top_archive.append(
                        {
                            "score": entry.score,
                            "depth": entry.depth,
                            "rpn": entry.rpn,
                        }
                    )
                key = canonical_expr_key(entry.expr)
                if key in pool_keys:
                    continue
                shortlist.append(entry)
                if len(shortlist) >= POOL_SHORTLIST:
                    break
            progress.emit(
                "archive_top",
                depth=depth,
                round=round_index,
                entries=top_archive,
            )
            progress.emit(
                "pool_shortlist",
                depth=depth,
                round=round_index,
                formulas=[
                    {
                        "score": entry.score,
                        "depth": entry.depth,
                        "rpn": entry.rpn,
                    }
                    for entry in shortlist
                ],
            )

            trials = []
            for trial_index, entry in enumerate(shortlist, start=1):
                trial_pool = tuple(item.expr for item in pool_entries) + (
                    entry.expr,
                )
                progress.emit(
                    "ridge_trial_start",
                    depth=depth,
                    round=round_index,
                    trial=trial_index,
                    candidate=entry.rpn,
                    alpha_count=len(trial_pool),
                    ridge_recompute_every=RIDGE_RECOMPUTE_EVERY,
                )
                pool_evaluator = CppStreamPoolEvaluator(
                    evaluation_sources,
                    n_instruments=n_instruments,
                    work_dir=(
                        OUTPUT_DIR
                        / f"depth_{depth}"
                        / f"round_{round_index}"
                        / "pool_trials"
                        / f"trial_{trial_index}"
                    ),
                )
                with Heartbeat(
                    progress,
                    "ridge_trial_still_working",
                    interval_seconds=HEARTBEAT_SECONDS,
                    payload={
                        "depth": depth,
                        "round": round_index,
                        "trial": trial_index,
                        "alpha_count": len(trial_pool),
                    },
                ):
                    evaluation = pool_evaluator.evaluate(
                        trial_pool,
                        ridge_recompute_every=RIDGE_RECOMPUTE_EVERY,
                    )
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
                }
                trials.append(trial)
                progress.emit(
                    "ridge_trial_done",
                    depth=depth,
                    round=round_index,
                    trial=trial_index,
                    candidate=entry.rpn,
                    pool_score=evaluation.score,
                    additive_delta=delta,
                    compile_seconds=round(evaluation.compile_seconds, 6),
                    run_seconds=round(evaluation.run_seconds, 6),
                    native_seconds=evaluation.native_seconds,
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
                "search_round_done",
                depth=depth,
                round=round_index,
                accepted=(
                    accepted["entry"].rpn if accepted is not None else None
                ),
                additive_delta=(
                    accepted["delta"] if accepted is not None else None
                ),
                pool_score=(
                    current_pool_score
                    if math.isfinite(current_pool_score)
                    else None
                ),
                pool_size=len(pool_entries),
                wall_seconds=round(time.perf_counter() - stage_started, 6),
            )
            print(_pool_tree(pool_entries), flush=True)

            stage_reports.append(
                {
                    "depth": depth,
                    "round": round_index,
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
                    "trials": [
                        {
                            "rpn": item["entry"].rpn,
                            "individual_score": item["entry"].score,
                            "pool_score": item["score"],
                            "additive_delta": item["delta"],
                            "compile_seconds": item["compile_seconds"],
                            "run_seconds": item["run_seconds"],
                            "native_seconds": item["native_seconds"],
                        }
                        for item in trials
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
                }
            )

        if len(pool_entries) >= TARGET_POOL_SIZE:
            break

    report = {
        "backend": "trading_dsl_engine.cpp_stream",
        "inputdata_fp": input_data.fp,
        "rows": rows,
        "instruments": n_instruments,
        "alpha_keys": list(INPUTDATA_ALPHA_KEYS),
        "config": {
            "max_depth": MAX_DEPTH,
            "rounds_per_depth": ROUNDS_PER_DEPTH,
            "simulations": SIMULATIONS,
            "rollouts_per_expansion": ROLLOUTS,
            "evaluation_batch": EVALUATION_BATCH,
            "archive_size": ARCHIVE_SIZE,
            "pool_shortlist": POOL_SHORTLIST,
            "target_pool_size": TARGET_POOL_SIZE,
            "ridge_recompute_every": RIDGE_RECOMPUTE_EVERY,
            "seed": SEED,
        },
        "final_pool_score": (
            current_pool_score if math.isfinite(current_pool_score) else None
        ),
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
        "search_seconds": time.perf_counter() - search_started,
    }
    report_path = OUTPUT_DIR / "riskminer_inputdata_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True))
    progress.emit(
        "done",
        final_pool_size=len(pool_entries),
        final_pool_score=report["final_pool_score"],
        report=str(report_path),
        search_seconds=round(report["search_seconds"], 6),
    )
    print("=== FINAL ROOT-LEVEL RIDGE POOL ===", flush=True)
    print(_pool_tree(pool_entries), flush=True)


if __name__ == "__main__":
    main()
