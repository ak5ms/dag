from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np

from trading_dsl_engine.base.dsl import Ridge, cat, einsum, ewm_std, ffill, get_beta, shift, var, where
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.cpp_stream import compile_formula

from flows.riskminer.canonical import expression_identifiers


def halflife_to_span(halflife: float) -> float:
    value = float(halflife)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("halflife must be positive and finite")
    alpha = 1.0 - math.exp(math.log(0.5) / value)
    return 2.0 / alpha - 1.0


def build_ridge_pool_sharpe(alphas: Sequence[Expr], *, roll_rets_name: str = "roll_rets", hs_name: str = "hs", vol_name: str = "vol", is_tradable_name: str = "is_tradable", ridge_halflife: float = 1440.0 * 5.0, ridge_lambda: float = 0.0) -> Expr:
    if not alphas:
        raise ValueError("Ridge pool requires at least one alpha")
    roll_rets = var(roll_rets_name)
    hs = var(hs_name)
    vol = var(vol_name)
    is_tradable = var(is_tradable_name)
    nan = float("nan")
    clean_rets = where(roll_rets != 0.0, roll_rets, nan)
    scaled_features = tuple(alpha * vol for alpha in alphas)
    scaled_alphas = cat(*scaled_features)
    model = Ridge(
        *(shift(feature, 1, 1) for feature in scaled_features),
        y=clean_rets,
        weights=hs ** -2.0,
        hl=float(ridge_halflife),
        lambda_=float(ridge_lambda),
        nonneg=False,
    )
    yhat = einsum("f,nf->n", get_beta(model), scaled_alphas)
    risk_span = halflife_to_span(ridge_halflife)
    denominator = ewm_std(yhat, span=risk_span) * ewm_std(clean_rets, span=risk_span)
    session_position = ffill(where(is_tradable, yhat / denominator, nan))
    pool_pnl = (shift(session_position, 1, 1) * clean_rets).sum(axis=1)
    return pool_pnl.mean(axis=0) / pool_pnl.std(axis=0)


@dataclass(frozen=True)
class PoolEvaluation:
    sharpe: float
    compile_seconds: float
    run_seconds: float
    output_mode: str
    output_shape: tuple[int, ...]
    runtime_type: str
    input_names: tuple[str, ...]


class CppStreamRidgePoolEvaluator:
    def __init__(self, sources: Mapping[str, object], *, n_instruments: int, work_dir: str | Path, ridge_halflife: float = 1440.0 * 5.0, ridge_lambda: float = 0.0) -> None:
        self.sources = dict(sources)
        self.n_instruments = int(n_instruments)
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.ridge_halflife = float(ridge_halflife)
        self.ridge_lambda = float(ridge_lambda)

    def evaluate(self, alphas: Sequence[Expr]) -> PoolEvaluation:
        expression = build_ridge_pool_sharpe(alphas, ridge_halflife=self.ridge_halflife, ridge_lambda=self.ridge_lambda)
        compile_started = time.perf_counter()
        required = expression_identifiers(expression)
        missing = sorted(required - self.sources.keys())
        if missing:
            raise KeyError(f"missing cpp_stream pool sources: {missing}")
        bound_sources = {name: self.sources[name] for name in required}
        runtime = compile_formula(expression, bound_sources, n_instruments=self.n_instruments)
        compile_seconds = time.perf_counter() - compile_started
        if runtime.plan.output_mode != "final":
            raise RuntimeError(f"pool score must be final-only, got {runtime.plan.output_mode!r}")
        output_path = self.work_dir / f"ridge-pool-{len(alphas)}.bin"
        run_started = time.perf_counter()
        result = runtime.run(out_path=output_path)
        run_seconds = time.perf_counter() - run_started
        values = np.fromfile(result.output_path, dtype=np.float64).reshape(-1)
        if values.size != 1:
            raise RuntimeError(f"pool score expected one scalar, got {values.size} with shape {result.output_shape!r}")
        return PoolEvaluation(float(values[0]), compile_seconds, run_seconds, runtime.plan.output_mode, tuple(result.output_shape or ()), f"{type(runtime).__module__}.{type(runtime).__qualname__}", tuple(runtime.program.input_names))


__all__ = ["CppStreamRidgePoolEvaluator", "PoolEvaluation", "build_ridge_pool_sharpe", "halflife_to_span"]
