from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from trading_dsl_engine.base.dsl import (
    Ridge,
    cat,
    div,
    emit,
    einsum,
    fillna,
    get_residuals,
    gt,
    mul,
    reduction,
    shift,
    var,
    where,
)
from trading_dsl_engine.base.parser import Expr

from .cpp_stream_eval import _identifier_names
from .pool import PoolAlpha, PoolTransition, RidgeAlphaPool
from .rpn import StackValue, canonical_expr_key


@dataclass(frozen=True)
class OrthogonalBatchExecution:
    candidate_count: int
    pool_size: int
    compile_seconds: float
    run_seconds: float
    native_seconds: float | None
    output_path: str
    rank_saturation_possible: bool


@dataclass
class OrthogonalEvaluationSummary:
    requested: int = 0
    finite: int = 0
    zero_or_nonfinite: int = 0
    batches: list[OrthogonalBatchExecution] = field(default_factory=list)


def build_cross_sectional_orthogonal_alpha(
    candidate: Expr,
    pool: Sequence[Expr],
) -> Expr:
    """Return the raw candidate residual after raw cross-sectional projection.

    Neither X nor y is volatility scaled. ``hl=0`` makes every timestamp an
    independent instrument-cross-sectional regression. ``lambda_=0`` permits
    the native pseudo-inverse fallback to define the minimum-norm fit for
    singular systems and for K greater than the number of instruments.
    """

    if not pool:
        return candidate
    model = Ridge(
        *tuple(pool),
        y=candidate,
        weights=1.0,
        hl=0.0,
        lambda_=0.0,
        nonneg=False,
    )
    return get_residuals(model)


def build_orthogonal_score_formula(
    candidates: Sequence[Expr],
    pool: Sequence[Expr],
    *,
    roll_rets_name: str = "roll_rets",
    epsilon: float = 1.0e-14,
) -> Expr:
    """Emit Sharpe scores of raw candidates orthogonalized to the pool."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    roll_rets = var(roll_rets_name)
    orthogonal = tuple(
        build_cross_sectional_orthogonal_alpha(candidate, pool)
        for candidate in candidates
    )
    shifted = tuple(fillna(shift(alpha, 1, 1), 0.0) for alpha in orthogonal)
    if len(shifted) == 1:
        pnl = reduction("sum", mul(shifted[0], roll_rets), axis=1)
    else:
        pnl = reduction(
            "sum",
            einsum("nf,n->nf", cat(*shifted), roll_rets),
            axis=1,
        )
    mean = reduction("mean", pnl, axis=0)
    std = reduction("std", pnl, axis=0, ddof=0)
    score = where(gt(std, float(epsilon)), div(mean, std), 0.0)
    return emit(score, mode="last")


class CppStreamOrthogonalEvaluator:
    """Batched native intermediate reward evaluator with pool-aware caching."""

    def __init__(
        self,
        sources: Mapping[str, Any],
        *,
        n_instruments: int,
        work_dir: str | Path,
        batch_size: int = 16,
        compile_kwargs: Mapping[str, Any] | None = None,
        run_kwargs: Mapping[str, Any] | None = None,
        on_event: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> None:
        if int(n_instruments) <= 0 or int(batch_size) <= 0:
            raise ValueError("n_instruments and batch_size must be positive")
        self.sources = dict(sources)
        self.n_instruments = int(n_instruments)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = int(batch_size)
        self.compile_kwargs = dict(compile_kwargs or {})
        self.run_kwargs = dict(run_kwargs or {})
        self.score_cache: dict[tuple, float] = {}
        self.summary = OrthogonalEvaluationSummary()
        self.on_event = on_event

    def _emit(self, event: str, **payload: object) -> None:
        if self.on_event is not None:
            self.on_event(event, payload)

    def evaluate(
        self,
        candidates: Sequence[Expr],
        pool: Sequence[Expr],
    ) -> dict[tuple, float]:
        self.summary.requested += len(candidates)
        pool_key = tuple(canonical_expr_key(alpha) for alpha in pool)
        unique: list[Expr] = []
        candidate_keys: list[tuple] = []
        seen: set[tuple] = set()
        for candidate in candidates:
            key = canonical_expr_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
            candidate_keys.append(key)
        pending = [
            (candidate, key, (pool_key, key))
            for candidate, key in zip(unique, candidate_keys)
            if (pool_key, key) not in self.score_cache
        ]
        for start in range(0, len(pending), self.batch_size):
            chunk = pending[start:start + self.batch_size]
            self._evaluate_batch(
                [item[0] for item in chunk],
                [item[2] for item in chunk],
                pool,
            )
        return {
            key: self.score_cache.get((pool_key, key), 0.0)
            for key in candidate_keys
        }

    def _evaluate_batch(
        self,
        candidates: Sequence[Expr],
        cache_keys: Sequence[tuple],
        pool: Sequence[Expr],
    ) -> None:
        from trading_dsl_engine.cpp_stream import compile_formula

        formula = build_orthogonal_score_formula(candidates, pool)
        required = _identifier_names(formula)
        missing = sorted(required - self.sources.keys())
        if missing:
            raise KeyError(f"missing orthogonal reward sources: {missing}")
        bound = {name: self.sources[name] for name in sorted(required)}
        self._emit("orthogonal_compile_start", candidate_count=len(candidates), pool_size=len(pool))
        started = time.perf_counter()
        runtime = compile_formula(
            formula, bound, n_instruments=self.n_instruments, **self.compile_kwargs
        )
        compile_seconds = time.perf_counter() - started
        self._emit("orthogonal_compile_done", candidate_count=len(candidates), pool_size=len(pool), compile_seconds=compile_seconds)
        digest = hashlib.sha256(repr(cache_keys).encode()).hexdigest()[:16]
        output_path = self.work_dir / f"orthogonal_reward_{digest}.bin"
        self._emit("orthogonal_run_start", candidate_count=len(candidates), pool_size=len(pool))
        started = time.perf_counter()
        result = runtime.run(out_path=output_path, **self.run_kwargs)
        run_seconds = time.perf_counter() - started
        self._emit("orthogonal_run_done", candidate_count=len(candidates), pool_size=len(pool), run_seconds=run_seconds, native_seconds=getattr(result, "seconds", None))
        result_path = Path(getattr(result, "output_path", output_path))
        values = np.fromfile(result_path, dtype=np.float64)
        if values.size != len(candidates):
            raise RuntimeError(
                f"orthogonal score emitted {values.size}; expected {len(candidates)}"
            )
        finite = zero_or_nonfinite = 0
        for cache_key, value in zip(cache_keys, values):
            score = float(value)
            if not math.isfinite(score):
                score = 0.0
                zero_or_nonfinite += 1
            else:
                finite += 1
                if score == 0.0:
                    zero_or_nonfinite += 1
            self.score_cache[cache_key] = score
        self.summary.finite += finite
        self.summary.zero_or_nonfinite += zero_or_nonfinite
        native = getattr(result, "seconds", None)
        self.summary.batches.append(
            OrthogonalBatchExecution(
                candidate_count=len(candidates),
                pool_size=len(pool),
                compile_seconds=compile_seconds,
                run_seconds=run_seconds,
                native_seconds=float(native) if native is not None else None,
                output_path=str(result_path),
                rank_saturation_possible=len(pool) >= self.n_instruments,
            )
        )


@dataclass(frozen=True)
class TerminalReward:
    transition: PoolTransition

    @property
    def reward(self) -> float:
        return self.transition.terminal_reward


class RewardDensePoolModel:
    """Orthogonal intermediate reward plus exact validation pool terminal reward."""

    def __init__(
        self,
        intermediate_evaluator: CppStreamOrthogonalEvaluator,
        pool: RidgeAlphaPool,
        *,
        on_event: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> None:
        self.intermediate_evaluator = intermediate_evaluator
        self.pool = pool
        self.on_event = on_event
        self.rank_saturation_reported = False

    def _emit(self, event: str, **payload: object) -> None:
        if self.on_event is not None:
            self.on_event(event, payload)

    def intermediate_rewards(
        self,
        values: Sequence[StackValue],
    ) -> dict[tuple, float]:
        if not values:
            return {}
        if (
            len(self.pool.entries) >= self.intermediate_evaluator.n_instruments
            and not self.rank_saturation_reported
        ):
            self.rank_saturation_reported = True
            self._emit(
                "orthogonal_rank_saturation",
                pool_size=len(self.pool.entries),
                instruments=self.intermediate_evaluator.n_instruments,
                explanation=(
                    "pinv is defined, but a full-rank K>=N pool spans the entire "
                    "instrument cross-section and leaves a zero residual"
                ),
            )
        scores = self.intermediate_evaluator.evaluate(
            [value.expr for value in values], self.pool.expressions
        )
        self._emit(
            "intermediate_reward_batch",
            candidate_count=len(values),
            pool_size=len(self.pool.entries),
            rank_saturation_possible=(
                len(self.pool.entries) >= self.intermediate_evaluator.n_instruments
            ),
        )
        return scores

    def terminal_reward(
        self,
        value: StackValue,
        *,
        rpn: str,
        individual_score: float = float("nan"),
    ) -> TerminalReward:
        transition = self.pool.consider(
            PoolAlpha(
                expr=value.expr,
                canonical_key=value.canonical_key,
                rpn=rpn,
                depth=value.depth,
                individual_score=float(individual_score),
            )
        )
        self._emit(
            "pool_transition",
            candidate=rpn,
            committed=transition.committed,
            evicted=(transition.evicted.rpn if transition.evicted else None),
            pool_size=transition.pool_size,
            pool_score=transition.resulting_score,
            additive_delta=transition.additive_delta,
        )
        return TerminalReward(transition)


__all__ = [
    "CppStreamOrthogonalEvaluator", "OrthogonalBatchExecution",
    "OrthogonalEvaluationSummary", "RewardDensePoolModel", "TerminalReward",
    "build_cross_sectional_orthogonal_alpha", "build_orthogonal_score_formula",
]
