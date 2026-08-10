from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


def halflife_to_span(halflife: float) -> float:
    value = float(halflife)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("halflife must be finite and positive")
    alpha = 1.0 - math.exp(math.log(0.5) / value)
    return 2.0 / alpha - 1.0


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
    """Build the user's native Ridge-pool Sharpe graph."""

    if not alphas:
        raise ValueError("at least one alpha is required")
    if int(ridge_recompute_every) < 1:
        raise ValueError("ridge_recompute_every must be >= 1")
    roll_rets = var(roll_rets_name)
    hs = var(hs_name)
    vol = var(vol_name)
    is_tradable = var(is_tradable_name)
    nan = float("nan")

    clean_rets = where(ne(roll_rets, 0.0), roll_rets, nan)
    ridge_weights = dsl_pow(hs, -2.0)
    scaled_alphas = tuple(mul(alpha, vol) for alpha in alphas)
    scaled_matrix = cat(*scaled_alphas)

    # Keep every alpha as a separate Ridge feature. Passing the cat matrix as one
    # feature makes the native Ridge feature count disagree with the matrix width.
    reg = Ridge(
        *(shift(alpha, 1, 1) for alpha in scaled_alphas),
        y=clean_rets,
        weights=ridge_weights,
        hl=float(ridge_halflife),
        lambda_=float(ridge_lambda),
        nonneg=False,
        recompute_every=int(ridge_recompute_every),
    )
    yhat = einsum(
        "f,nf->n",
        get_beta(reg),
        scaled_matrix,
    )
    risk_span = halflife_to_span(risk_halflife)
    denominator = mul(
        ewm_std(yhat, span=risk_span),
        ewm_std(clean_rets, span=risk_span),
    )
    session_position = ffill(
        where(
            is_tradable,
            div(yhat, denominator),
            nan,
        )
    )
    pool_contributions = fillna(
        mul(
            shift(session_position, 1, 1),
            clean_rets,
        ),
        0.0,
    )
    pool_pnl = reduction(
        "sum",
        pool_contributions,
        axis=1,
    )
    score = div(
        reduction("mean", pool_pnl, axis=0),
        reduction("std", pool_pnl, axis=0, ddof=0),
    )
    return emit(score, mode="last")


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


class CppStreamPoolEvaluator:
    def __init__(
        self,
        sources: Mapping[str, Any],
        *,
        n_instruments: int,
        work_dir: str | Path,
        compile_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        self.sources = dict(sources)
        self.n_instruments = int(n_instruments)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.compile_kwargs = dict(compile_kwargs or {})

    def evaluate(
        self,
        alphas: Sequence[Expr],
        **formula_kwargs,
    ) -> PoolEvaluation:
        from trading_dsl_engine.cpp_stream import compile_formula

        formula = build_ridge_pool_score_formula(
            alphas,
            **formula_kwargs,
        )
        required = _identifier_names(formula)
        missing = sorted(required - self.sources.keys())
        if missing:
            raise KeyError(f"missing cpp_stream pool sources: {missing}")
        bound_sources = {
            name: self.sources[name]
            for name in sorted(required)
        }

        compile_started = time.perf_counter()
        runtime = compile_formula(
            formula,
            bound_sources,
            n_instruments=self.n_instruments,
            **self.compile_kwargs,
        )
        compile_seconds = time.perf_counter() - compile_started

        output_path = self.work_dir / "ridge_pool_score.bin"
        run_started = time.perf_counter()
        result = runtime.run(out_path=output_path)
        run_seconds = time.perf_counter() - run_started
        result_path = Path(getattr(result, "output_path", output_path))
        values = np.fromfile(result_path, dtype=np.float64)
        if values.size != 1:
            raise RuntimeError(
                f"pool score emitted {values.size} values, expected one"
            )
        native_seconds = getattr(result, "seconds", None)
        return PoolEvaluation(
            score=float(values[0]),
            alpha_count=len(alphas),
            compile_seconds=compile_seconds,
            run_seconds=run_seconds,
            native_seconds=(
                float(native_seconds)
                if native_seconds is not None
                else None
            ),
            runtime_type=f"{type(runtime).__module__}.{type(runtime).__name__}",
            output_path=str(result_path),
            output_shape=tuple(getattr(result, "output_shape", ())),
        )


__all__ = [
    "CppStreamPoolEvaluator",
    "PoolEvaluation",
    "build_ridge_pool_score_formula",
    "halflife_to_span",
]
