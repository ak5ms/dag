from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace

import numpy as np

from flows.alpha_mcts import AlphaMCTS, SearchConfig, SearchResult, SemanticInfo, market_terminal_semantics
from trading_dsl_engine.base.dsl import abs as dsl_abs, clip, ensure_expr, var
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.base.terminals import alpha_search_field_metadata

AlphaEvaluator = Callable[[Expr], np.ndarray]


def sharpe_ratio(pnl: np.ndarray) -> float:
    values = np.asarray(pnl, dtype=float).reshape(-1)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return float("-inf")
    std = float(values.std())
    if not np.isfinite(std) or std <= 0.0:
        return float("-inf")
    return float(values.mean() / std)


def make_sharpe_fitness(
    evaluate_alpha: AlphaEvaluator,
    forward_returns: np.ndarray,
    *,
    is_tradable: np.ndarray | None = None,
) -> Callable[[Expr], float]:
    """Build the sole search reward: realized Sharpe of the candidate alpha."""
    returns = np.asarray(forward_returns, dtype=float)
    tradable = np.ones_like(returns, dtype=bool) if is_tradable is None else np.asarray(is_tradable, dtype=bool)
    if returns.shape != tradable.shape:
        raise ValueError("forward_returns and is_tradable must have identical shapes")

    def fitness(expr: Expr) -> float:
        alpha = np.asarray(evaluate_alpha(expr), dtype=float)
        if alpha.shape != returns.shape:
            raise ValueError(f"candidate shape {alpha.shape} does not match returns shape {returns.shape}")
        pnl = np.where(tradable & np.isfinite(alpha) & np.isfinite(returns), alpha * returns, np.nan)
        if pnl.ndim > 1:
            pnl = np.nansum(pnl, axis=tuple(range(1, pnl.ndim)))
        return sharpe_ratio(pnl)

    return fitness


def adaptive_parameter_terminals(
    fields: Mapping[str, Mapping[str, object]],
    *,
    min_span: float,
    max_span: float,
) -> dict[str, tuple[Expr, SemanticInfo]]:
    """Create bounded expression-valued parameter sources from every field.

    The generic abs+clip transform guarantees positivity without manually
    enumerating field-specific parameter recipes. Stateful operators consume
    the current row's value, so these are tagged scalar-like for grammar use.
    """
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
    forward_returns: np.ndarray,
    *,
    is_tradable: np.ndarray | None = None,
    field_metadata: Mapping[str, Mapping[str, object]] | None = None,
    config: SearchConfig = SearchConfig(),
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
    fitness = make_sharpe_fitness(evaluate_alpha, forward_returns, is_tradable=is_tradable)
    return AlphaMCTS(terminals, fitness, config=config).search()


__all__ = [
    "adaptive_parameter_terminals", "make_sharpe_fitness",
    "search_market_alphas", "sharpe_ratio",
]
