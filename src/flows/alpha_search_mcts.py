from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
import math

import numpy as np

from flows.alpha_mcts import AlphaMCTS, SearchConfig, SearchResult, SemanticInfo, market_terminal_semantics
from flows.alpha_operator_schemas import all_operator_schemas
from trading_dsl_engine.base.dsl import abs as dsl_abs, clip, ensure_expr, var
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.base.terminals import alpha_search_field_metadata

AlphaEvaluator = Callable[[Expr], np.ndarray]


def sharpe_ratio(pnl: np.ndarray) -> float:
    """Return the requested search score: pnl.sum() / pnl.std()."""
    values = np.asarray(pnl, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("-inf")
    std = float(values.std())
    if not np.isfinite(std) or std <= 0.0:
        return float("-inf")
    return float(values.sum() / std)


def make_sharpe_fitness(
    evaluate_alpha: AlphaEvaluator,
    returns: np.ndarray,
    *,
    is_tradable: np.ndarray | None = None,
    chunk_rows: int = 262_144,
) -> Callable[[Expr], float]:
    """Build the sole search reward using a bounded-memory exact reduction.

    This implements::

        w = alpha
        pnl = w.shift().mul(r).sum(1)
        fitness = pnl.sum() / pnl.std()

    The first shifted row contributes zero because the row reduction uses the
    DSL's skip-NaN semantics. Population standard deviation (``ddof=0``) is
    reconstructed from float64 sums and squared sums across row chunks.
    """
    returns_array = np.asarray(returns)
    tradable = (
        np.ones(returns_array.shape, dtype=bool)
        if is_tradable is None
        else np.asarray(is_tradable, dtype=bool)
    )
    if returns_array.ndim != 2:
        raise ValueError("returns must be a 2-D time-by-instrument array")
    if returns_array.shape != tradable.shape:
        raise ValueError("returns and is_tradable must have identical shapes")
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be positive")

    def fitness(expr: Expr) -> float:
        w = np.asarray(evaluate_alpha(expr))
        if w.shape != returns_array.shape:
            raise ValueError(f"candidate shape {w.shape} does not match returns shape {returns_array.shape}")
        n_rows = returns_array.shape[0]
        if n_rows < 2:
            return float("-inf")

        total = 0.0
        total_sq = 0.0
        count = 1  # pnl[0] == 0 after skip-NaN row reduction.
        for start in range(1, n_rows, chunk_rows):
            stop = min(n_rows, start + chunk_rows)
            prev_w = np.asarray(w[start - 1:stop - 1], dtype=np.float64)
            r_chunk = np.asarray(returns_array[start:stop], dtype=np.float64)
            valid = (
                tradable[start:stop]
                & np.isfinite(prev_w)
                & np.isfinite(r_chunk)
            )
            contributions = np.where(valid, prev_w * r_chunk, np.nan)
            pnl_chunk = np.nansum(contributions, axis=1, dtype=np.float64)
            total += float(pnl_chunk.sum(dtype=np.float64))
            total_sq += float(np.dot(pnl_chunk, pnl_chunk))
            count += int(pnl_chunk.size)

        variance = total_sq / count - (total / count) ** 2
        std = math.sqrt(max(0.0, variance))
        if not math.isfinite(std) or std <= 0.0:
            return float("-inf")
        return total / std

    return fitness


def adaptive_parameter_terminals(
    fields: Mapping[str, Mapping[str, object]],
    *,
    min_span: float,
    max_span: float,
) -> dict[str, tuple[Expr, SemanticInfo]]:
    """Create bounded expression-valued parameter sources from every field."""
    out: dict[str, tuple[Expr, SemanticInfo]] = {}
    lo = ensure_expr(float(min_span))
    hi = ensure_expr(float(max_span))
    for name in fields:
        expr = clip(dsl_abs(var(name)), lo, hi)
        info = SemanticInfo(
            types=frozenset({"numeric", "dimensionless", "parameter", "adaptive_parameter"}),
            shape="scalar",
            lower=float(min_span),
            upper=float(max_span),
            integer=False,
        )
        out[f"adaptive_span:{name}"] = (expr, info)
    return out


def search_market_alphas(
    evaluate_alpha: AlphaEvaluator,
    returns: np.ndarray,
    *,
    is_tradable: np.ndarray | None = None,
    field_metadata: Mapping[str, Mapping[str, object]] | None = None,
    config: SearchConfig = SearchConfig(),
    fitness_chunk_rows: int = 262_144,
) -> SearchResult:
    fields = dict(alpha_search_field_metadata() if field_metadata is None else field_metadata)
    terminals = market_terminal_semantics(fields)
    if config.dynamic_parameters:
        terminals.update(adaptive_parameter_terminals(
            fields,
            min_span=config.min_span,
            max_span=config.max_span,
        ))
    if not config.target_types:
        config = replace(config, target_types=frozenset({"dimensionless"}))
    fitness = make_sharpe_fitness(
        evaluate_alpha,
        returns,
        is_tradable=is_tradable,
        chunk_rows=fitness_chunk_rows,
    )
    return AlphaMCTS(
        terminals,
        fitness,
        operators=all_operator_schemas(),
        config=config,
    ).search()


__all__ = [
    "adaptive_parameter_terminals", "make_sharpe_fitness",
    "search_market_alphas", "sharpe_ratio",
]
