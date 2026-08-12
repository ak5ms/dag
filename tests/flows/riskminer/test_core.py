import pytest

from flows.riskminer import (
    DEFAULT_TYPE_GRAPH,
    RiskMCTS,
    RiskMinerConfig,
    TypedRPNEnvironment,
    alpha_terminal_metadata,
    build_vocabulary,
    canonical_expr_key,
)
from flows.riskminer.rpn import TokenKind
from flows.riskminer.semantics import compatible, division_output
from trading_dsl_engine.base.parser import Call


def _small_config(**updates):
    values = dict(
        max_depth=8,
        max_tokens=16,
        max_stack=6,
        simulations=1,
        rollouts_per_expansion=1,
        evaluation_batch_size=1,
        archive_size=1,
    )
    values.update(updates)
    return RiskMinerConfig(**values)


def _apply_names(environment, *names):
    state = environment.initial_state()
    for name in names:
        state = environment.apply(
            state,
            environment.vocabulary.by_name[name].token_id,
        )
    return state


def test_requested_market_terminal_semantics_and_ranges():
    fields = alpha_terminal_metadata()
    assert set(fields) == {
        "ap0", "bp0", "av0", "bv0", "volume", "vwap",
        "open", "high", "low", "close", "soft_side_wavg",
    }
    assert {"price", "quote_price", "ask"}.issubset(fields["ap0"].types)
    assert {"quantity", "quoted_size", "bid"}.issubset(fields["bv0"].types)
    assert {"dimensionless", "signed_trade_side", "order_flow"}.issubset(
        fields["soft_side_wavg"].types
    )
    assert fields["soft_side_wavg"].lower == -1.0
    assert fields["soft_side_wavg"].upper == 1.0


def test_type_closure_gives_algorithmic_compatibility():
    fields = alpha_terminal_metadata()
    assert compatible(fields["ap0"], fields["bp0"])
    assert compatible(fields["av0"], fields["volume"])
    assert not compatible(fields["close"], fields["volume"])
    assert "quantity" in DEFAULT_TYPE_GRAPH.closure(fields["av0"].types)
    assert "quote_side" in DEFAULT_TYPE_GRAPH.closure(fields["ap0"].types)
    result = division_output(fields["close"], fields["vwap"])
    assert {"dimensionless", "ratio"}.issubset(result.types)


def test_typed_rpn_constructs_a_dimensionless_formula():
    environment = TypedRPNEnvironment(
        config=_small_config(),
        vocabulary=build_vocabulary(),
    )
    state = _apply_names(
        environment,
        "ap0",
        "bp0",
        "sub",
        "xs_rank",
        "END",
    )
    assert state.terminated
    value = environment.formula_value(state)
    assert value is not None
    assert isinstance(value.expr, Call)
    assert value.expr.fn == "xs_rank"
    assert value.depth <= 8
    assert "dimensionless" in value.semantics.types


def test_price_plus_quantity_is_removed_from_legal_action_mask():
    environment = TypedRPNEnvironment(config=_small_config())
    state = _apply_names(environment, "close", "volume")
    legal = set(environment.legal_actions(state))
    add_id = environment.vocabulary.by_name["add"].token_id
    assert add_id not in legal
    with pytest.raises(ValueError, match="illegal RPN token"):
        environment.apply(state, add_id)


def test_literal_parameter_builds_native_static_ewm_expression():
    environment = TypedRPNEnvironment(config=_small_config())
    state = _apply_names(
        environment,
        "close",
        "CONST[20]",
        "ewm",
        "xs_rank",
        "END",
    )
    value = environment.formula_value(state)
    assert value is not None
    assert "ewm" in repr(value.expr)
    assert "xs_rank" in repr(value.expr)


def test_target_only_fields_are_not_in_formula_vocabulary():
    vocabulary = build_vocabulary()
    terminal_names = {
        token.name
        for token in vocabulary
        if token.kind is TokenKind.TERMINAL
    }
    assert {"roll_rets", "hs", "vol", "is_tradable"}.isdisjoint(
        terminal_names
    )


class StubEvaluator:
    def evaluate(self, candidates):
        return {
            canonical_expr_key(expr): (
                2.0 if "soft_side_wavg" in repr(expr) else 0.5
            )
            for expr in candidates
        }


def _run_mcts():
    config = _small_config(
        max_depth=4,
        max_tokens=12,
        max_stack=5,
        simulations=12,
        rollouts_per_expansion=3,
        evaluation_batch_size=4,
        archive_size=8,
        seed=7,
    )
    environment = TypedRPNEnvironment(config=config)
    return RiskMCTS(environment, StubEvaluator(), config=config).search()


def test_mcts_is_deterministic_and_collects_finite_unique_formulas():
    first = _run_mcts()
    second = _run_mcts()
    assert first.archive
    assert [
        (entry.canonical_key, entry.score, entry.rpn)
        for entry in first.archive
    ] == [
        (entry.canonical_key, entry.score, entry.rpn)
        for entry in second.archive
    ]
    assert len({entry.canonical_key for entry in first.archive}) == len(first.archive)
    assert first.metrics.simulations == 12
    assert first.metrics.rollouts == 36
    assert first.metrics.finite_formula_scores > 0
    assert any("soft_side_wavg" in repr(entry.expr) for entry in first.archive)
