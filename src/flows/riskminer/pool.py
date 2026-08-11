from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import time
from typing import Any

import numpy as np

from trading_dsl_engine.base.dsl import (
    Ridge,
    abs as dsl_abs,
    cat,
    div,
    emit,
    einsum,
    ewm_std,
    ffill,
    fillna,
    get_beta,
    mul,
    ne,
    pow as dsl_pow,
    reduction,
    shift,
    var,
    where,
)
from trading_dsl_engine.base.parser import Expr

from .cpp_stream_eval import _identifier_names
from .rpn import canonical_expr_key


def halflife_to_span(halflife: float) -> float:
    value = float(halflife)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("halflife must be finite and positive")
    alpha = 1.0 - math.exp(math.log(0.5) / value)
    return 2.0 / alpha - 1.0


def _build_ridge_model(
    alphas: Sequence[Expr],
    *,
    roll_rets_name: str,
    hs_name: str,
    vol_name: str,
    ridge_halflife: float,
    ridge_lambda: float,
    ridge_recompute_every: int,
):
    if not alphas:
        raise ValueError("at least one alpha is required")
    if int(ridge_recompute_every) < 1:
        raise ValueError("ridge_recompute_every must be >= 1")
    roll_rets = var(roll_rets_name)
    hs = var(hs_name)
    vol = var(vol_name)
    clean_rets = where(ne(roll_rets, 0.0), roll_rets, float("nan"))
    ridge_weights = dsl_pow(hs, -2.0)
    scaled_alphas = tuple(mul(alpha, vol) for alpha in alphas)
    kwargs: dict[str, object] = {
        "y": clean_rets,
        "weights": ridge_weights,
        "hl": float(ridge_halflife),
        "lambda_": float(ridge_lambda),
        "nonneg": False,
    }
    # Current cpp-stream-operators supports this optional solve cadence.  Do not
    # emit the kwarg for k=1 so older caches/formula hashes remain compatible.
    if int(ridge_recompute_every) != 1:
        kwargs["recompute_every"] = int(ridge_recompute_every)
    reg = Ridge(*(shift(alpha, 1, 1) for alpha in scaled_alphas), **kwargs)
    return reg, scaled_alphas, clean_rets


def build_ridge_pool_score_formula(
    alphas: Sequence[Expr],
    *,
    roll_rets_name: str = "roll_rets",
    hs_name: str = "hs",
    vol_name: str = "vol",
    is_tradable_name: str = "is_tradable",
    ridge_halflife: float = 1440.0 * 5.0,
    ridge_lambda: float = 0.0,
    ridge_recompute_every: int = 1,
    risk_halflife: float = 1440.0 * 5.0,
) -> Expr:
    """Build the user's exact root-level online Ridge/yhat pool Sharpe."""

    reg, scaled_alphas, clean_rets = _build_ridge_model(
        alphas,
        roll_rets_name=roll_rets_name,
        hs_name=hs_name,
        vol_name=vol_name,
        ridge_halflife=ridge_halflife,
        ridge_lambda=ridge_lambda,
        ridge_recompute_every=ridge_recompute_every,
    )
    yhat = einsum("f,nf->n", get_beta(reg), cat(*scaled_alphas))
    span = halflife_to_span(risk_halflife)
    denominator = mul(
        ewm_std(yhat, span=span),
        ewm_std(clean_rets, span=span),
    )
    position = ffill(
        where(
            var(is_tradable_name),
            div(yhat, denominator),
            float("nan"),
        )
    )
    pool_pnl = reduction(
        "sum",
        fillna(mul(shift(position, 1, 1), clean_rets), 0.0),
        axis=1,
    )
    return emit(
        div(
            reduction("mean", pool_pnl, axis=0),
            reduction("std", pool_pnl, axis=0, ddof=0),
        ),
        mode="last",
    )


def build_ridge_pool_beta_formula(
    alphas: Sequence[Expr],
    *,
    roll_rets_name: str = "roll_rets",
    hs_name: str = "hs",
    vol_name: str = "vol",
    ridge_halflife: float = 1440.0 * 5.0,
    ridge_lambda: float = 0.0,
    ridge_recompute_every: int = 1,
    importance: str = "final_abs",
) -> Expr:
    """Emit coefficient importance used for paper-style capacity eviction."""

    reg, _, _ = _build_ridge_model(
        alphas,
        roll_rets_name=roll_rets_name,
        hs_name=hs_name,
        vol_name=vol_name,
        ridge_halflife=ridge_halflife,
        ridge_lambda=ridge_lambda,
        ridge_recompute_every=ridge_recompute_every,
    )
    beta_abs = dsl_abs(get_beta(reg))
    if importance == "final_abs":
        return emit(beta_abs, mode="last")
    if importance == "mean_abs":
        return emit(reduction("mean", beta_abs, axis=0), mode="last")
    raise ValueError("importance must be 'final_abs' or 'mean_abs'")


@dataclass(frozen=True)
class PoolEvaluation:
    score: float
    alpha_count: int
    compile_seconds: float
    run_seconds: float
    native_seconds: float | None
    runtime_type: str
    output_path: str
    output_shape: tuple[int, ...]
    coefficient_importance: tuple[float, ...] = ()


class CppStreamPoolEvaluator:
    """Native pool scorer with formula/result caching."""

    def __init__(
        self,
        sources: Mapping[str, Any],
        *,
        n_instruments: int,
        work_dir: str | Path,
        compile_kwargs: Mapping[str, Any] | None = None,
        run_kwargs: Mapping[str, Any] | None = None,
        on_event: Callable[[str, Mapping[str, object]], None] | None = None,
    ) -> None:
        if int(n_instruments) <= 0:
            raise ValueError("n_instruments must be positive")
        self.sources = dict(sources)
        self.n_instruments = int(n_instruments)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.compile_kwargs = dict(compile_kwargs or {})
        self.run_kwargs = dict(run_kwargs or {})
        self._cache: dict[tuple, PoolEvaluation] = {}
        self.on_event = on_event

    def _emit(self, event: str, **payload: object) -> None:
        if self.on_event is not None:
            self.on_event(event, payload)

    def _run_formula(
        self, formula: Expr, output_path: Path
    ) -> tuple[np.ndarray, float, float, float | None, str, tuple[int, ...]]:
        from trading_dsl_engine.cpp_stream import compile_formula

        required = _identifier_names(formula)
        missing = sorted(required - self.sources.keys())
        if missing:
            raise KeyError(f"missing cpp_stream pool sources: {missing}")
        bound = {name: self.sources[name] for name in sorted(required)}
        self._emit("pool_compile_start", output_path=str(output_path))
        started = time.perf_counter()
        runtime = compile_formula(
            formula, bound, n_instruments=self.n_instruments, **self.compile_kwargs
        )
        compile_seconds = time.perf_counter() - started
        self._emit(
            "pool_compile_done",
            output_path=str(output_path),
            compile_seconds=compile_seconds,
        )
        self._emit("pool_run_start", output_path=str(output_path))
        started = time.perf_counter()
        result = runtime.run(out_path=output_path, **self.run_kwargs)
        run_seconds = time.perf_counter() - started
        self._emit(
            "pool_run_done",
            output_path=str(output_path),
            run_seconds=run_seconds,
            native_seconds=getattr(result, "seconds", None),
        )
        result_path = Path(getattr(result, "output_path", output_path))
        values = np.fromfile(result_path, dtype=np.float64)
        native = getattr(result, "seconds", None)
        return (
            values,
            compile_seconds,
            run_seconds,
            float(native) if native is not None else None,
            f"{type(runtime).__module__}.{type(runtime).__name__}",
            tuple(getattr(result, "output_shape", ())),
        )

    def evaluate(
        self,
        alphas: Sequence[Expr],
        *,
        include_importance: bool = False,
        importance: str = "final_abs",
        **formula_kwargs,
    ) -> PoolEvaluation:
        alpha_keys = tuple(canonical_expr_key(alpha) for alpha in alphas)
        cache_key = (
            alpha_keys,
            bool(include_importance),
            importance,
            tuple(sorted((name, repr(value)) for name, value in formula_kwargs.items())),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]
        digest = hashlib.sha256(repr(cache_key).encode()).hexdigest()[:16]
        score_formula = build_ridge_pool_score_formula(alphas, **formula_kwargs)
        score_path = self.work_dir / f"ridge_pool_score_{digest}.bin"
        values, compile_s, run_s, native_s, runtime_type, output_shape = self._run_formula(
            score_formula, score_path
        )
        if values.size != 1:
            raise RuntimeError(f"pool score emitted {values.size} values, expected one")

        coefficient_importance: tuple[float, ...] = ()
        if include_importance:
            beta_kwargs = {
                name: value
                for name, value in formula_kwargs.items()
                if name not in {"is_tradable_name", "risk_halflife"}
            }
            beta_formula = build_ridge_pool_beta_formula(
                alphas, importance=importance, **beta_kwargs
            )
            beta_path = self.work_dir / f"ridge_pool_beta_{digest}.bin"
            beta, beta_compile, beta_run, beta_native, _, _ = self._run_formula(
                beta_formula, beta_path
            )
            if beta.size != len(alphas):
                raise RuntimeError(
                    f"pool beta emitted {beta.size} values, expected {len(alphas)}"
                )
            coefficient_importance = tuple(float(value) for value in beta)
            compile_s += beta_compile
            run_s += beta_run
            if native_s is not None and beta_native is not None:
                native_s += beta_native

        evaluation = PoolEvaluation(
            score=float(values[0]),
            alpha_count=len(alphas),
            compile_seconds=compile_s,
            run_seconds=run_s,
            native_seconds=native_s,
            runtime_type=runtime_type,
            output_path=str(score_path),
            output_shape=output_shape,
            coefficient_importance=coefficient_importance,
        )
        self._cache[cache_key] = evaluation
        return evaluation


@dataclass(frozen=True)
class PoolAlpha:
    expr: Expr
    canonical_key: tuple
    rpn: str
    depth: int
    individual_score: float = float("nan")


@dataclass(frozen=True)
class PoolTransition:
    candidate: PoolAlpha
    terminal_reward: float
    previous_score: float
    resulting_score: float
    additive_delta: float
    committed: bool
    evicted: PoolAlpha | None
    pool_size: int
    evaluation: PoolEvaluation


class RidgeAlphaPool:
    """Mutable alpha pool with exact score admission and weight eviction."""

    def __init__(
        self,
        evaluator: CppStreamPoolEvaluator,
        *,
        capacity: int = 100,
        min_improvement: float = 0.0,
        formula_kwargs: Mapping[str, Any] | None = None,
        importance: str = "mean_abs",
    ) -> None:
        if int(capacity) <= 0:
            raise ValueError("capacity must be positive")
        self.evaluator = evaluator
        self.capacity = int(capacity)
        self.min_improvement = float(min_improvement)
        self.formula_kwargs = dict(formula_kwargs or {})
        if importance not in {"final_abs", "mean_abs"}:
            raise ValueError("importance must be 'final_abs' or 'mean_abs'")
        self.importance = importance
        self.entries: list[PoolAlpha] = []
        self.score = -math.inf
        self.transitions: list[PoolTransition] = []

    @property
    def expressions(self) -> tuple[Expr, ...]:
        return tuple(entry.expr for entry in self.entries)

    @property
    def keys(self) -> frozenset[tuple]:
        return frozenset(entry.canonical_key for entry in self.entries)

    def _empty_evaluation(self, *, runtime_type: str) -> PoolEvaluation:
        return PoolEvaluation(
            score=float(self.score), alpha_count=len(self.entries),
            compile_seconds=0.0, run_seconds=0.0, native_seconds=None,
            runtime_type=runtime_type, output_path="", output_shape=(),
        )

    def consider(self, candidate: PoolAlpha) -> PoolTransition:
        previous_score = float(self.score)
        if candidate.canonical_key in self.keys:
            transition = PoolTransition(
                candidate=candidate,
                terminal_reward=previous_score,
                previous_score=previous_score,
                resulting_score=previous_score,
                additive_delta=0.0,
                committed=False,
                evicted=None,
                pool_size=len(self.entries),
                evaluation=self._empty_evaluation(runtime_type="duplicate"),
            )
            self.transitions.append(transition)
            return transition

        trial = self.entries + [candidate]
        evaluation = self.evaluator.evaluate(
            tuple(entry.expr for entry in trial),
            include_importance=len(trial) > self.capacity,
            importance=self.importance,
            **self.formula_kwargs,
        )
        resulting_entries = list(trial)
        resulting_evaluation = evaluation
        evicted: PoolAlpha | None = None

        if len(trial) > self.capacity:
            importance = evaluation.coefficient_importance
            if len(importance) != len(trial):
                raise RuntimeError("pool eviction requires one coefficient per alpha")
            evict_index = min(
                range(len(trial)), key=lambda index: (abs(importance[index]), index)
            )
            evicted = trial[evict_index]
            resulting_entries = [
                entry for index, entry in enumerate(trial) if index != evict_index
            ]
            if resulting_entries == self.entries:
                resulting_score = previous_score
                resulting_evaluation = self._empty_evaluation(runtime_type="candidate_evicted")
            else:
                resulting_evaluation = self.evaluator.evaluate(
                    tuple(entry.expr for entry in resulting_entries),
                    **self.formula_kwargs,
                )
                resulting_score = float(resulting_evaluation.score)
        else:
            resulting_score = float(evaluation.score)

        delta = (
            resulting_score - previous_score
            if math.isfinite(previous_score)
            else resulting_score
        )
        committed = math.isfinite(resulting_score) and (
            not self.entries or delta > self.min_improvement
        )
        if committed:
            self.entries = resulting_entries
            self.score = resulting_score
        transition = PoolTransition(
            candidate=candidate,
            terminal_reward=resulting_score,
            previous_score=previous_score,
            resulting_score=resulting_score,
            additive_delta=float(delta),
            committed=committed,
            evicted=(evicted if committed else None),
            pool_size=len(self.entries),
            evaluation=resulting_evaluation,
        )
        self.transitions.append(transition)
        return transition


__all__ = [
    "CppStreamPoolEvaluator", "PoolAlpha", "PoolEvaluation", "PoolTransition",
    "RidgeAlphaPool", "build_ridge_pool_beta_formula",
    "build_ridge_pool_score_formula", "halflife_to_span",
]
