from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
import threading
import time
from typing import Any

from trading_dsl_engine.base.parser import Expr

from .cpp_stream_eval import CppStreamCandidateEvaluator
from .mcts import RiskMinerSearchResult, SearchMetrics


@dataclass
class ConsoleProgress:
    """Line-oriented progress sink suitable for terminals and CI logs."""

    prefix: str = "riskminer"
    started: float = 0.0

    def __post_init__(self) -> None:
        if not self.started:
            self.started = time.perf_counter()

    def emit(self, event: str, **payload: object) -> None:
        record = {
            "elapsed_seconds": round(time.perf_counter() - self.started, 6),
            "event": event,
            **payload,
        }
        print(
            f"[{self.prefix}] " + json.dumps(record, sort_keys=True, default=str),
            flush=True,
        )


class Heartbeat:
    def __init__(
        self,
        progress: ConsoleProgress,
        event: str,
        *,
        interval_seconds: float,
        payload: Mapping[str, object],
    ) -> None:
        self.progress = progress
        self.event = event
        self.interval_seconds = max(0.5, float(interval_seconds))
        self.payload = dict(payload)
        self.stop = threading.Event()
        self.started = 0.0
        self.thread: threading.Thread | None = None

    def __enter__(self):
        self.started = time.perf_counter()

        def run() -> None:
            while not self.stop.wait(self.interval_seconds):
                self.progress.emit(
                    self.event,
                    stage_elapsed_seconds=round(
                        time.perf_counter() - self.started,
                        6,
                    ),
                    **self.payload,
                )

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        del exc_type, exc, traceback
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)


class DiagnosticCppStreamCandidateEvaluator(CppStreamCandidateEvaluator):
    """CppStream evaluator that exposes each slow compile as it happens."""

    def __init__(
        self,
        *args,
        progress: ConsoleProgress | None = None,
        heartbeat_seconds: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.diagnostic_progress = progress or ConsoleProgress()
        self.heartbeat_seconds = float(heartbeat_seconds)
        self._diagnostic_batch = 0

    def evaluate(self, candidates: Sequence[Expr]) -> dict[tuple, float]:
        before_unique = self.summary.unique
        before_batches = len(self.summary.batches)
        started = time.perf_counter()
        self.diagnostic_progress.emit(
            "candidate_wave_start",
            requested=len(candidates),
            cached_scores=len(self.score_cache),
            configured_batch_size=self.batch_size,
        )
        values = super().evaluate(candidates)
        new_batches = self.summary.batches[before_batches:]
        self.diagnostic_progress.emit(
            "candidate_wave_done",
            requested=len(candidates),
            newly_unique=self.summary.unique - before_unique,
            finite=sum(math.isfinite(value) for value in values.values()),
            batches=len(new_batches),
            compile_seconds=round(
                sum(batch.compile_seconds for batch in new_batches),
                6,
            ),
            run_seconds=round(
                sum(batch.run_seconds for batch in new_batches),
                6,
            ),
            elapsed_seconds_for_wave=round(time.perf_counter() - started, 6),
        )
        return values

    def _evaluate_batch(self, expressions, keys):
        self._diagnostic_batch += 1
        batch_number = self._diagnostic_batch
        previews = []
        for expr in expressions[:3]:
            text = repr(expr)
            previews.append(text if len(text) <= 220 else text[:217] + "...")
        payload = {
            "batch": batch_number,
            "candidate_count": len(expressions),
            "formula_previews": previews,
        }
        self.diagnostic_progress.emit("native_batch_start", **payload)
        before = len(self.summary.batches)
        started = time.perf_counter()
        with Heartbeat(
            self.diagnostic_progress,
            "native_batch_still_working",
            interval_seconds=self.heartbeat_seconds,
            payload=payload,
        ):
            values = super()._evaluate_batch(expressions, keys)
        elapsed = time.perf_counter() - started
        batch = self.summary.batches[-1] if len(self.summary.batches) > before else None
        self.diagnostic_progress.emit(
            "native_batch_done",
            **payload,
            wall_seconds=round(elapsed, 6),
            compile_seconds=(
                round(batch.compile_seconds, 6) if batch is not None else None
            ),
            run_seconds=(round(batch.run_seconds, 6) if batch is not None else None),
            native_seconds=(
                batch.native_seconds if batch is not None else None
            ),
            output_shape=(batch.output_shape if batch is not None else None),
            runtime_type=(batch.runtime_type if batch is not None else None),
            cache_path=(batch.cache_path if batch is not None else None),
        )
        return values


class TracingRiskMCTS:
    """Thin wrapper that runs the existing MCTS and reports aggregate progress.

    The evaluator emits the expensive batch-level events. This wrapper reports
    search-level start/end information without changing the underlying tree
    policy or reward semantics.
    """

    def __init__(self, mcts: Any, *, progress: ConsoleProgress | None = None) -> None:
        self.mcts = mcts
        self.progress = progress or ConsoleProgress()

    def search(self) -> RiskMinerSearchResult:
        config = self.mcts.config
        self.progress.emit(
            "mcts_start",
            max_depth=config.max_depth,
            min_formula_depth=config.min_formula_depth,
            simulations=config.simulations,
            rollouts_per_expansion=config.rollouts_per_expansion,
            evaluation_batch_size=config.evaluation_batch_size,
            archive_size=config.archive_size,
        )
        started = time.perf_counter()
        result = self.mcts.search()
        metrics: SearchMetrics = result.metrics
        self.progress.emit(
            "mcts_done",
            simulations=metrics.simulations,
            rollouts=metrics.rollouts,
            unique_formula_requests=metrics.unique_formula_requests,
            finite_formula_scores=metrics.finite_formula_scores,
            invalid_rollouts=metrics.invalid_rollouts,
            tree_nodes=metrics.tree_nodes,
            archive_size=len(result.archive),
            best_score=(result.archive[0].score if result.archive else None),
            wall_seconds=round(time.perf_counter() - started, 6),
        )
        return result


__all__ = [
    "ConsoleProgress",
    "Heartbeat",
    "DiagnosticCppStreamCandidateEvaluator",
    "TracingRiskMCTS",
]
