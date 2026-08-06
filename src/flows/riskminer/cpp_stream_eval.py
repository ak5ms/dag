from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from trading_dsl_engine.base.dsl import (
    cat,
    div,
    emit,
    einsum,
    reduction,
    shift,
    var,
)
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

    @property
    def compile_seconds(self) -> float:
        return sum(item.compile_seconds for item in self.batches)

    @property
    def run_seconds(self) -> float:
        return sum(item.run_seconds for item in self.batches)


def build_candidate_score_formula(
    candidates: Sequence[Expr],
    *,
    roll_rets_name: str = "roll_rets",
) -> Expr:
    """Build final-only native scores for a batch of row-vector alphas.

    For every candidate ``w`` this computes exactly::

        pnl = shift(w, 1, 1).mul(roll_rets).sum(axis=1)
        score = pnl.mean(axis=0) / pnl.std(axis=0, ddof=0)

    ``cat`` creates row shape ``(instrument, candidate)``. Reduction axes address
    ``(time, *row_shape)``, so axis 1 is instruments and axis 0 is time.
    """

    if not candidates:
        raise ValueError("at least one candidate is required")
    alpha_matrix = cat(*candidates)
    contributions = einsum(
        "nf,n->nf",
        shift(alpha_matrix, 1, 1),
        var(roll_rets_name),
    )
    pnl = reduction("sum", contributions, axis=1)
    pnl_mean = reduction("mean", pnl, axis=0)
    pnl_std = reduction("std", pnl, axis=0, ddof=0)
    return emit(div(pnl_mean, pnl_std), mode="last")


def _identifier_names(expr: Expr) -> frozenset[str]:
    if isinstance(expr, Identifier):
        if expr.name in {"True", "False", "self_"}:
            return frozenset()
        return frozenset({expr.name})
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


class CppStreamCandidateEvaluator:
    """Batched candidate evaluator with no non-cpp_stream fallback."""

    def __init__(
        self,
        sources: Mapping[str, Any],
        *,
        n_instruments: int,
        work_dir: str | Path,
        roll_rets_name: str = "roll_rets",
        batch_size: int = 32,
        compile_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        if n_instruments <= 0:
            raise ValueError("n_instruments must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.sources = dict(sources)
        self.n_instruments = int(n_instruments)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.roll_rets_name = roll_rets_name
        self.batch_size = int(batch_size)
        self.compile_kwargs = dict(compile_kwargs or {})
        self.score_cache: dict[tuple, float] = {}
        self.runtime_cache: dict[tuple[tuple, ...], Any] = {}
        self.summary = EvaluationSummary()

    def evaluate(self, candidates: Sequence[Expr]) -> dict[tuple, float]:
        self.summary.requested += len(candidates)
        unique: list[Expr] = []
        keys: list[tuple] = []
        seen: set[tuple] = set()
        for expr in candidates:
            key = canonical_expr_key(expr)
            if key in seen:
                continue
            seen.add(key)
            unique.append(expr)
            keys.append(key)

        pending_exprs: list[Expr] = []
        pending_keys: list[tuple] = []
        for key, expr in zip(keys, unique):
            if key not in self.score_cache:
                pending_keys.append(key)
                pending_exprs.append(expr)
        self.summary.unique += len(pending_exprs)

        for start in range(0, len(pending_exprs), self.batch_size):
            stop = min(len(pending_exprs), start + self.batch_size)
            self._evaluate_or_bisect(
                pending_exprs[start:stop],
                pending_keys[start:stop],
            )

        result = {key: self.score_cache.get(key, -math.inf) for key in keys}
        new_values = [self.score_cache.get(key, -math.inf) for key in pending_keys]
        finite = sum(math.isfinite(value) for value in new_values)
        self.summary.finite += finite
        self.summary.nonfinite += len(new_values) - finite
        return result

    def _evaluate_or_bisect(
        self,
        expressions: Sequence[Expr],
        keys: Sequence[tuple],
    ) -> None:
        try:
            values = self._evaluate_batch(expressions, keys)
        except Exception as exc:
            if len(expressions) == 1:
                key = keys[0]
                self.score_cache[key] = -math.inf
                self.summary.compile_rejected += 1
                self.summary.rejection_reasons[key] = (
                    f"{type(exc).__name__}: {exc}"
                )
                return
            midpoint = len(expressions) // 2
            self._evaluate_or_bisect(
                expressions[:midpoint],
                keys[:midpoint],
            )
            self._evaluate_or_bisect(
                expressions[midpoint:],
                keys[midpoint:],
            )
            return

        for key, value in zip(keys, values):
            self.score_cache[key] = float(value) if math.isfinite(value) else -math.inf

    def _evaluate_batch(
        self,
        expressions: Sequence[Expr],
        keys: Sequence[tuple],
    ) -> np.ndarray:
        from trading_dsl_engine.cpp_stream import compile_formula

        formula = build_candidate_score_formula(
            expressions,
            roll_rets_name=self.roll_rets_name,
        )
        required = _identifier_names(formula)
        missing = sorted(required - self.sources.keys())
        if missing:
            raise KeyError(f"missing cpp_stream sources: {missing}")
        bound_sources = {name: self.sources[name] for name in sorted(required)}
        batch_key = tuple(keys)

        compile_started = time.perf_counter()
        runtime = self.runtime_cache.get(batch_key)
        if runtime is None:
            runtime = compile_formula(
                formula,
                bound_sources,
                n_instruments=self.n_instruments,
                **self.compile_kwargs,
            )
            self.runtime_cache[batch_key] = runtime
        compile_seconds = time.perf_counter() - compile_started

        digest = hashlib.sha256(repr(batch_key).encode("utf-8")).hexdigest()[:16]
        output_path = self.work_dir / f"candidate_scores_{digest}.bin"
        run_started = time.perf_counter()
        run_result = runtime.run(out_path=output_path)
        run_seconds = time.perf_counter() - run_started

        result_path = Path(getattr(run_result, "output_path", output_path))
        output_shape = tuple(getattr(run_result, "output_shape", (len(expressions),)))
        values = np.fromfile(result_path, dtype=np.float64)
        if values.size != len(expressions):
            raise RuntimeError(
                "cpp_stream candidate score output has "
                f"{values.size} values; expected {len(expressions)} "
                f"(reported shape={output_shape!r})"
            )

        native_seconds = getattr(run_result, "seconds", None)
        cache_path = None
        for attribute in ("library_path", "shared_library_path", "artifact_path"):
            candidate = getattr(runtime, attribute, None)
            if candidate is not None:
                cache_path = str(candidate)
                break
        self.summary.batches.append(
            BatchExecution(
                candidate_count=len(expressions),
                compile_seconds=compile_seconds,
                run_seconds=run_seconds,
                native_seconds=(
                    float(native_seconds) if native_seconds is not None else None
                ),
                output_shape=output_shape,
                runtime_type=f"{type(runtime).__module__}.{type(runtime).__name__}",
                output_path=str(result_path),
                cache_path=cache_path,
            )
        )
        return values.reshape(-1)


__all__ = [
    "BatchExecution",
    "CppStreamCandidateEvaluator",
    "EvaluationSummary",
    "build_candidate_score_formula",
]
