import math

import numpy as np

from flows.alpha_mcts import (
    AlphaMCTS,
    Hole,
    SExpr,
    SearchConfig,
    canonical_key,
    default_operator_schemas,
    market_terminal_semantics,
)
from flows.alpha_search_mcts import adaptive_parameter_terminals, make_sharpe_fitness, sharpe_ratio
from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.base.terminals import alpha_search_field_metadata


def test_market_metadata_covers_requested_microstructure_fields():
    fields = alpha_search_field_metadata()
    assert set(fields) == {
        "ap0", "bp0", "av0", "bv0", "volume", "vwap",
        "open", "high", "low", "close", "soft_side_wavg",
    }
    assert {"price", "ask", "best_quote"}.issubset(fields["ap0"]["types"])
    assert {"contract_quantity", "quoted_size", "bid"}.issubset(fields["bv0"]["types"])
    assert {"dimensionless", "signed_trade_side", "order_flow"}.issubset(fields["soft_side_wavg"]["types"])
    assert fields["soft_side_wavg"]["range"] == (-1.0, 1.0)


def test_multi_type_terminals_and_dynamic_parameter_expressions_are_available():
    fields = alpha_search_field_metadata()
    terminals = market_terminal_semantics(fields)
    adaptive = adaptive_parameter_terminals(fields, min_span=2.0, max_span=100.0)
    assert terminals["ap0"][1].types.issuperset({"price", "quote_price", "ask"})
    assert adaptive
    assert all("adaptive_parameter" in info.types for _, info in adaptive.values())
    assert all(info.lower == 2.0 and info.upper == 100.0 for _, info in adaptive.values())
    assert all("clip" in repr(expr) and "abs" in repr(expr) for expr, _ in adaptive.values())


def test_operator_inventory_is_expansive_and_contains_dynamic_families():
    schemas = default_operator_schemas()
    names = {schema.name for schema in schemas}
    assert len(schemas) >= 35
    assert {
        "add", "sub", "mul", "div", "xs_rank", "xs_pct_rank", "xs_norm",
        "ewm", "ewm_std", "ewm_var", "ewm_skewness", "ewm_kurtosis",
        "rolling_quantile", "rolling_theilsen", "where", "clip",
    }.issubset(names)
    assert {schema.family for schema in schemas}.issuperset({"ewm", "rolling", "rolling_q", "shift", "comparison"})


def test_canonicalization_deduplicates_commutative_and_idempotent_forms():
    x = SExpr(terminal=var("x"))
    y = SExpr(terminal=var("y"))
    lhs = SExpr(op="add", children=(x, y))
    rhs = SExpr(op="add", children=(y, x))
    assert canonical_key(lhs) == canonical_key(rhs)

    inner = SExpr(op="xs_rank", children=(x,))
    ranked = SExpr(op="xs_rank", children=(inner,))
    assert canonical_key(ranked) == canonical_key(inner)


def test_fitness_uses_shift_row_sum_and_total_sum_over_std():
    returns = np.array([[0.01, -0.01], [0.02, -0.02], [0.03, -0.03]])
    signal = np.array([[1.0, -1.0], [1.0, -1.0], [1.0, -1.0]])
    fitness = make_sharpe_fitness(lambda expr: signal, returns)
    expected_pnl = np.array([0.0, 0.04, 0.06])
    expected = expected_pnl.sum() / expected_pnl.std()
    assert fitness(var("anything")) == expected
    assert fitness(var("anything")) == sharpe_ratio(expected_pnl)


def test_mcts_exposes_dimensionless_terminal_and_progressive_actions():
    terminals = market_terminal_semantics(alpha_search_field_metadata())
    search = AlphaMCTS(
        terminals,
        lambda expr: 1.0 if "soft_side_wavg" in repr(expr) else 0.0,
        config=SearchConfig(simulations=20, max_depth=2, seed=3),
    )
    root = SExpr.unresolved(Hole(
        required_types=frozenset({"dimensionless"}),
        shape="row",
        max_depth=2,
        role="alpha",
    ))
    actions = search._actions(root)
    assert any(action.label == "terminal:soft_side_wavg" for action in actions)
    assert any(action.label.startswith("op:xs_rank") for action in actions)
    result = search.search()
    assert math.isfinite(result.sharpe)
    assert result.sharpe in (0.0, 1.0)
