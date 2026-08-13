from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import math
import os
from pathlib import Path
import threading
import time
from typing import Any

import numpy as np

from trading_dsl_engine.base.dsl import cat, div, emit, einsum, fillna, mul, reduction, shift, var
from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple

from .rpn import canonical_expr_key


@dataclass(frozen=True)
class BatchExecution:
    candidate_count: int
    compile_seconds: float
    run_seconds: float
    native_seconds: float | None
    output_shape: tuple[int, ...]
    runtime_type: str
    output_path: str
    cache_path: str | None = None
    requested_threads: int = 0
    actual_threads: int | None = None


@dataclass
class EvaluationSummary:
    requested: int = 0
    unique: int = 0
    finite: int = 0
    nonfinite: int = 0
    compile_rejected: int = 0
    execution_rejected: int = 0
    batches: list[BatchExecution] = field(default_factory=list)
    rejection_reasons: dict[tuple, str] = field(default_factory=dict)
    peak_batch_workers: int = 1

    @property
    def compile_seconds(self) -> float:
        return sum(item.compile_seconds for item in self.batches)

    @property
    def run_seconds(self) -> float:
        return sum(item.run_seconds for item in self.batches)


def build_candidate_score_formula(candidates: Sequence[Expr], *, roll_rets_name: str = "roll_rets") -> Expr:
    """Compute shifted-alpha PnL mean/std scores as one final native formula."""
    if not candidates:
        raise ValueError("at least one candidate is required")
    roll_rets = var(roll_rets_name)
    shifted = tuple(fillna(shift(candidate, 1, 1), 0.0) for candidate in candidates)
    if len(shifted) == 1:
        pnl = reduction("sum", mul(shifted[0], roll_rets), axis=1)
    else:
        contributions = einsum("nf,n->nf", cat(*shifted), roll_rets)
        pnl = reduction("sum", contributions, axis=1)
    pnl_mean = reduction("mean", pnl, axis=0)
    pnl_std = reduction("std", pnl, axis=0, ddof=0)
    return emit(div(pnl_mean, pnl_std), mode="last")


def _identifier_names(expr: Expr) -> frozenset[str]:
    if isinstance(expr, Identifier):
        return frozenset() if expr.name in {"True", "False", "self_"} else frozenset({expr.name})
    if isinstance(expr, KeyTuple):
        return frozenset().union(*(_identifier_names(item) for item in expr.items))
    if isinstance(expr, Call):
        parts = [_identifier_names(arg) for arg in expr.args]
        parts.extend(_identifier_names(value) for _, value in expr.kwargs)
        return frozenset().union(*parts) if parts else frozenset()
    if hasattr(expr, "args"):
        parts = [_identifier_names(arg) for arg in expr.args]
        return frozenset().union(*parts) if parts else frozenset()
    return frozenset()


def _available_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


class CppStreamCandidateEvaluator:
    """Batched native evaluator with bounded population-level concurrency."""

    def __init__(self, sources: Mapping[str, Any], *, n_instruments: int,
                 work_dir: str | Path, roll_rets_name: str = "roll_rets",
                 batch_size: int = 32, workers: int = 0,
                 compile_kwargs: Mapping[str, Any] | None = None) -> None:
        if n_instruments <= 0:
            raise ValueError("n_instruments must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if int(workers) < 0:
            raise ValueError("workers must be >= 0; zero selects available CPUs")
        self.sources = dict(sources)
        self.n_instruments = int(n_instruments)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.roll_rets_name = roll_rets_name
        self.batch_size = int(batch_size)
        self.workers = int(workers)
        self.compile_kwargs = dict(compile_kwargs or {})
        self.score_cache: dict[tuple, float] = {}
        self.runtime_cache: dict[tuple[tuple, ...], Any] = {}
        self.summary = EvaluationSummary()
        self._lock = threading.RLock()

    def _batch_worker_count(self, batch_count: int) -> int:
        if batch_count <= 1:
            return 1
        requested = self.workers or _available_cpus()
        return max(1, min(batch_count, requested, _available_cpus()))

    def evaluate(self, candidates: Sequence[Expr]) -> dict[tuple, float]:
        with self._lock:
            self.summary.requested += len(candidates)
        unique: list[Expr] = []
        keys: list[tuple] = []
        seen: set[tuple] = set()
        for expr in candidates:
            key = canonical_expr_key(expr)
            if key not in seen:
                seen.add(key)
                unique.append(expr)
                keys.append(key)
        with self._lock:
            cached = frozenset(self.score_cache)
        pending = [(expr, key) for expr, key in zip(unique, keys) if key not in cached]
        with self._lock:
            self.summary.unique += len(pending)
        batches = [pending[start:start + self.batch_size] for start in range(0, len(pending), self.batch_size)]
        worker_count = self._batch_worker_count(len(batches))
        with self._lock:
            self.summary.peak_batch_workers = max(self.summary.peak_batch_workers, worker_count)
        run_threads = 1 if worker_count > 1 else 0

        def run_batch(batch: list[tuple[Expr, tuple]]) -> None:
            self._evaluate_or_bisect(
                [item[0] for item in batch],
                [item[1] for item in batch],
                run_threads=run_threads,
            )

        if worker_count == 1:
            for batch in batches:
                run_batch(batch)
        else:
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="cpp-stream-candidates") as executor:
                futures = [executor.submit(run_batch, batch) for batch in batches]
                for future in futures:
                    future.result()
        with self._lock:
            result = {key: self.score_cache.get(key, -math.inf) for key in keys}
            new_values = [self.score_cache.get(key, -math.inf) for _, key in pending]
            finite = sum(math.isfinite(value) for value in new_values)
            self.summary.finite += finite
            self.summary.nonfinite += len(new_values) - finite
        return result

    def _evaluate_or_bisect(self, expressions: Sequence[Expr], keys: Sequence[tuple], *, run_threads: int) -> None:
        try:
            values = self._evaluate_batch(expressions, keys, run_threads=run_threads)
        except Exception as exc:
            if len(expressions) == 1:
                with self._lock:
                    self.score_cache[keys[0]] = -math.inf
                    self.summary.compile_rejected += 1
                    self.summary.rejection_reasons[keys[0]] = f"{type(exc).__name__}: {exc}"
                return
            midpoint = len(expressions) // 2
            self._evaluate_or_bisect(expressions[:midpoint], keys[:midpoint], run_threads=run_threads)
            self._evaluate_or_bisect(expressions[midpoint:], keys[midpoint:], run_threads=run_threads)
            return
        with self._lock:
            for key, value in zip(keys, values):
                self.score_cache[key] = float(value) if math.isfinite(value) else -math.inf

    def _evaluate_batch(self, expressions: Sequence[Expr], keys: Sequence[tuple], *, run_threads: int) -> np.ndarray:
        from trading_dsl_engine.cpp_stream import compile_formula
        formula = build_candidate_score_formula(expressions, roll_rets_name=self.roll_rets_name)
        required = _identifier_names(formula)
        missing = sorted(required - self.sources.keys())
        if missing:
            raise KeyError(f"missing cpp_stream sources: {missing}")
        bound_sources = {name: self.sources[name] for name in sorted(required)}
        batch_key = tuple(keys)
        compile_started = time.perf_counter()
        with self._lock:
            runtime = self.runtime_cache.get(batch_key)
        if runtime is None:
            compiled = compile_formula(formula, bound_sources, n_instruments=self.n_instruments, **self.compile_kwargs)
            with self._lock:
                runtime = self.runtime_cache.setdefault(batch_key, compiled)
        compile_seconds = time.perf_counter() - compile_started
        digest = hashlib.sha256(repr(batch_key).encode("utf-8")).hexdigest()[:16]
        output_path = self.work_dir / f"candidate_scores_{digest}.bin"
        run_started = time.perf_counter()
        run_result = runtime.run(out_path=output_path, threads=run_threads)
        run_seconds = time.perf_counter() - run_started
        result_path = Path(getattr(run_result, "output_path", output_path))
        output_shape = tuple(getattr(run_result, "output_shape", (len(expressions),)))
        values = np.fromfile(result_path, dtype=np.float64)
        if values.size != len(expressions):
            raise RuntimeError(f"cpp_stream candidate score output has {values.size} values; expected {len(expressions)} (reported shape={output_shape!r})")
        native_seconds = getattr(run_result, "seconds", None)
        cache_path = next((str(getattr(runtime, attr)) for attr in ("library_path", "shared_library_path", "artifact_path") if getattr(runtime, attr, None) is not None), None)
        execution = BatchExecution(
            candidate_count=len(expressions),
            compile_seconds=compile_seconds,
            run_seconds=run_seconds,
            native_seconds=float(native_seconds) if native_seconds is not None else None,
            output_shape=output_shape,
            runtime_type=f"{type(runtime).__module__}.{type(runtime).__name__}",
            output_path=str(result_path),
            cache_path=cache_path,
            requested_threads=run_threads,
            actual_threads=int(run_result.threads) if hasattr(run_result, "threads") else None,
        )
        with self._lock:
            self.summary.batches.append(execution)
        return values.reshape(-1)


__all__ = ["BatchExecution", "CppStreamCandidateEvaluator", "EvaluationSummary", "build_candidate_score_formula"]
