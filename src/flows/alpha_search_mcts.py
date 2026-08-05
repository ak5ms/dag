from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from typing import Any

import numpy as np

from flows.alpha_mcts import AlphaMCTS, SearchConfig, SearchResult, market_terminal_semantics
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
    """Build the sole MCTS reward: realized Sharpe of the candidate alpha.

    Candidate values at t are applied to forward_returns at t.  Callers that
    use contemporaneous returns should shift either the alpha or returns before
    constructing this function.
    """
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
    # Alpha output is dimensionless by default.  This remains a semantic target,
    # independent of the existing unit checker.
    if not config.target_types:
        config = replace(config, target_types=frozenset({"dimensionless"}))
    fitness = make_sharpe_fitness(evaluate_alpha, forward_returns, is_tradable=is_tradable)
    return AlphaMCTS(terminals, fitness, config=config).search()


__all__ = ["make_sharpe_fitness", "search_market_alphas", "sharpe_ratio"]
