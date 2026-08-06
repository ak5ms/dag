from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from trading_dsl_engine.base.dsl import cat, einsum, shift, var
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.cpp_stream import compile_formula

from flows.riskminer.canonical import canonical_string, expression_identifiers


@dataclass
class EvaluationStats:
    requested: int = 0
    cache_hits: int = 0
    compiled_batches: int = 0
    compile_failures: int = 0
    execution_failures: int = 0
    nonfinite_scores: int = 0
    compile_seconds: float = 0.0
    run_seconds: float = 0.0
    rejection_messages: dict[str, str] = field(default_factory=dict)
    last_runtime_type: str | None = None
    last_output_mode: str | None = None
    last_output_shape: tuple[int, ...] | None = None
    last_input_names: tuple[str, ...] = ()
    last_native_path: str | None = None


class CppStreamCandidateEvaluator:
    """Evaluate completed alpha formulas only through cpp_stream."""

    def __init__(self, sources: Mapping[str, object], *, n_instruments: int, returns_name: str = "roll_rets", work_dir: str | Path, max_batch_size: int = 64) -> None:
        if n_instruments <= 0 or max_batch_size <= 0:
            raise ValueError("n_instruments and max_batch_size must be positive")
        if returns_name not in sources:
            raise KeyError(f"missing returns source {returns_name!r}")
        self.sources = dict(sources)
        self.n_instruments = int(n_instruments)
        self.returns_name = str(returns_name)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.max_batch_size = int(max_batch_size)
        self.cache: dict[str, float] = {}
        self.stats = EvaluationStats()

    def _score_expression(self, candidates: Sequence[Expr]) -> Expr:
        returns = var(self.returns_name)
        if len(candidates) == 1:
            pnl = (shift(candidates[0], 1, 1) * returns).sum(axis=1)
            return pnl.mean(axis=0) / pnl.std(axis=0)
        alpha_matrix = cat(*candidates)
        contributions = einsum("nf,n->nf", shift(alpha_matrix, 1, 1), returns)
        pnl = contributions.sum(axis=1)
        return pnl.mean(axis=0) / pnl.std(axis=0)

    def score_batch(self, candidates: Sequence[Expr]) -> list[float]:
        self.stats.requested += len(candidates)
        keys = [canonical_string(candidate) for candidate in candidates]
        unique_uncached: dict[str, Expr] = {}
        for key, candidate in zip(keys, candidates):
            if key in self.cache:
                self.stats.cache_hits += 1
            else:
                unique_uncached.setdefault(key, candidate)
        pending = list(unique_uncached.items())
        for start in range(0, len(pending), self.max_batch_size):
            self._evaluate_or_bisect(pending[start : start + self.max_batch_size])
        return [self.cache.get(key, -math.inf) for key in keys]

    def _evaluate_or_bisect(self, items: Sequence[tuple[str, Expr]]) -> None:
        if not items:
            return
        try:
            values = self._evaluate_native(items)
        except Exception as exc:
            if len(items) > 1:
                midpoint = len(items) // 2
                self._evaluate_or_bisect(items[:midpoint])
                self._evaluate_or_bisect(items[midpoint:])
                return
            key, _ = items[0]
            self.stats.compile_failures += 1
            self.stats.rejection_messages[key] = f"{type(exc).__name__}: {exc}"
            self.cache[key] = -math.inf
            return
        for (key, _), value in zip(items, values):
            numeric = float(value)
            if not math.isfinite(numeric):
                self.stats.nonfinite_scores += 1
                numeric = -math.inf
            self.cache[key] = numeric

    def _evaluate_native(self, items: Sequence[tuple[str, Expr]]) -> np.ndarray:
        keys = [key for key, _ in items]
        candidates = [candidate for _, candidate in items]
        expression = self._score_expression(candidates)
        digest = hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()[:20]
        output_path = self.work_dir / f"candidate-scores-{digest}.bin"
        compile_started = time.perf_counter()
        required = expression_identifiers(expression)
        missing = sorted(required - self.sources.keys())
        if missing:
            raise KeyError(f"missing cpp_stream sources: {missing}")
        bound_sources = {name: self.sources[name] for name in required}
        runtime = compile_formula(expression, bound_sources, n_instruments=self.n_instruments)
        self.stats.compile_seconds += time.perf_counter() - compile_started
        self.stats.compiled_batches += 1
        if runtime.plan.output_mode != "final":
            raise RuntimeError(f"candidate score output must be final-only, got {runtime.plan.output_mode!r}")
        run_started = time.perf_counter()
        try:
            result = runtime.run(out_path=output_path)
        except Exception:
            self.stats.execution_failures += 1
            raise
        self.stats.run_seconds += time.perf_counter() - run_started
        values = np.fromfile(result.output_path, dtype=np.float64).reshape(-1)
        if values.size != len(items):
            raise RuntimeError(f"expected {len(items)} final scores, received {values.size}; output_shape={result.output_shape!r}")
        self.stats.last_runtime_type = f"{type(runtime).__module__}.{type(runtime).__qualname__}"
        self.stats.last_output_mode = runtime.plan.output_mode
        self.stats.last_output_shape = tuple(result.output_shape or ())
        self.stats.last_input_names = tuple(runtime.program.input_names)
        native = getattr(runtime, "library_path", None) or getattr(runtime, "shared_library_path", None) or getattr(runtime, "_library_path", None)
        self.stats.last_native_path = None if native is None else str(native)
        return values


__all__ = ["CppStreamCandidateEvaluator", "EvaluationStats"]
