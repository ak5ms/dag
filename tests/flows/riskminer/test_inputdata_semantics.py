from __future__ import annotations

import math

from flows.riskminer import (
    DEFAULT_TYPE_GRAPH,
    INPUTDATA_ALPHA_KEYS,
    RiskMinerConfig,
    TypedRPNEnvironment,
    build_vocabulary,
    inputdata_alpha_terminal_metadata,
)
from flows.riskminer.semantics import (
    compatible,
    division_output,
    subtraction_output,
)


def test_inputdata_alpha_metadata_covers_exact_user_keyset():
    metadata = inputdata_alpha_terminal_metadata()
    assert tuple(metadata) == INPUTDATA_ALPHA_KEYS
    assert len(metadata) == 69


def test_market_field_type_relations_are_value_based_not_location_based():
    metadata = inputdata_alpha_terminal_metadata()

    ap = metadata["ap0_out0"]
    bp = metadata["bp9_out0"]
    ask_volume = metadata["volume_a0_out0"]
    trade_volume = metadata["volume_out0"]
    vwap = metadata["vwap_out0"]
    mid = metadata["mp_out0.close"]

    assert "price" in DEFAULT_TYPE_GRAPH.closure(ap.types)
    assert "price" in DEFAULT_TYPE_GRAPH.closure(bp.types)
    assert "price" in DEFAULT_TYPE_GRAPH.closure(vwap.types)
    assert "price" in DEFAULT_TYPE_GRAPH.closure(mid.types)
    assert "quantity" in DEFAULT_TYPE_GRAPH.closure(ask_volume.types)
    assert "quantity" in DEFAULT_TYPE_GRAPH.closure(trade_volume.types)

    assert compatible(ap, bp)
    assert compatible(ask_volume, trade_volume)
    assert not compatible(ap, ask_volume)


def test_trade_cross_semantics_follow_source_definition():
    metadata = inputdata_alpha_terminal_metadata()
    count = metadata["trade_cross_pct_out0.count"]
    total = metadata["trade_cross_pct_out0.sum"]

    assert "dimensionless" in DEFAULT_TYPE_GRAPH.closure(count.types)
    assert count.nonnegative
    assert "quantity" in DEFAULT_TYPE_GRAPH.closure(total.types)
    # TradeQty * ((trade_price-bid)/(ask-bid)) can be signed when a trade
    # occurs outside the quoted spread.
    assert math.isinf(total.lower) and total.lower < 0.0


def test_time_subtraction_produces_duration_and_duration_ratio_is_dimensionless():
    metadata = inputdata_alpha_terminal_metadata()
    elapsed = subtraction_output(
        metadata["_ev_ts"],
        metadata["session_start0"],
    )
    session_length = subtraction_output(
        metadata["session_end0"],
        metadata["session_start0"],
    )
    ratio = division_output(elapsed, session_length)

    assert "duration" in DEFAULT_TYPE_GRAPH.closure(elapsed.types)
    assert "dimensionless" in DEFAULT_TYPE_GRAPH.closure(ratio.types)


def test_timestamp_plus_timestamp_is_not_legal_but_timestamp_difference_is():
    metadata = inputdata_alpha_terminal_metadata()
    vocabulary = build_vocabulary(terminals=metadata)
    env = TypedRPNEnvironment(
        config=RiskMinerConfig(
            max_depth=4,
            min_formula_depth=1,
            max_tokens=12,
            max_stack=8,
            simulations=1,
            rollouts_per_expansion=1,
            evaluation_batch_size=1,
            archive_size=1,
        ),
        vocabulary=vocabulary,
        target_types=("dimensionless",),
    )

    state = env.initial_state()
    state = env.apply(state, vocabulary.by_name["_ev_ts"].token_id)
    state = env.apply(state, vocabulary.by_name["session_start0"].token_id)
    legal = {vocabulary.by_id[token_id].name for token_id in env.legal_actions(state)}
    assert "sub" in legal
    assert "add" not in legal


def test_time_of_session_ratio_can_be_constructed_and_terminated():
    metadata = inputdata_alpha_terminal_metadata()
    vocabulary = build_vocabulary(terminals=metadata)
    env = TypedRPNEnvironment(
        config=RiskMinerConfig(
            max_depth=4,
            min_formula_depth=1,
            max_tokens=12,
            max_stack=8,
            simulations=1,
            rollouts_per_expansion=1,
            evaluation_batch_size=1,
            archive_size=1,
        ),
        vocabulary=vocabulary,
        target_types=("dimensionless",),
    )

    state = env.initial_state()
    for token_name in (
        "_ev_ts",
        "session_start0",
        "sub",
        "session_end0",
        "session_start0",
        "sub",
        "div",
        "END",
    ):
        state = env.apply(state, vocabulary.by_name[token_name].token_id)

    assert state.terminated
    value = env.formula_value(state)
    assert value is not None
    assert "dimensionless" in DEFAULT_TYPE_GRAPH.closure(value.semantics.types)


def test_special_field_ranges():
    metadata = inputdata_alpha_terminal_metadata()
    assert metadata["is_tradable_out0"].lower == 0.0
    assert metadata["is_tradable_out0"].upper == 1.0
    assert metadata["is_tradable_out0"].integer
    assert metadata["wdte_out0"].lower == 0.0
    assert metadata["volume_b9_out0"].lower == 0.0
    assert metadata["ap9_out0"].lower > 0.0


def test_gp_alpha_search_terminal_metadata_adds_roll_rets():
    from flows.gp import GPConfig, GrammarPolicy, make_pset
    from flows.gp.types import DimensionlessRow
    from flows.riskminer.semantics import (
        gp_alpha_search_terminal_metadata,
        gp_derived_alpha_terminal_metadata,
        inputdata_alpha_terminal_metadata,
    )

    derived = gp_derived_alpha_terminal_metadata()
    assert tuple(derived) == ("roll_rets",)
    assert "dimensionless" in DEFAULT_TYPE_GRAPH.closure(
        derived["roll_rets"].types
    )

    metadata = gp_alpha_search_terminal_metadata()
    assert len(metadata) == len(inputdata_alpha_terminal_metadata()) + 1
    assert "roll_rets" in metadata

    pset = make_pset(
        GPConfig(
            fields=metadata,
            grammar=GrammarPolicy(exclude_sections=("utils.group",)),
        )
    )
    terminal_names = {terminal.name for terminal in pset.terminals[DimensionlessRow]}
    assert "field_roll_rets" in terminal_names
