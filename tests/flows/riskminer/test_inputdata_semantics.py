from flows.riskminer import (
    DEFAULT_TYPE_GRAPH,
    INPUTDATA_ALPHA_KEYS,
    RiskMinerConfig,
    TypedRPNEnvironment,
    build_vocabulary,
    inputdata_alpha_terminal_metadata,
)
from flows.riskminer.semantics import compatible


def test_every_user_inputdata_key_has_semantics():
    values = inputdata_alpha_terminal_metadata()
    assert tuple(values) == INPUTDATA_ALPHA_KEYS
    assert len(values) == 69


def test_inputdata_field_definitions_and_compatibility():
    values = inputdata_alpha_terminal_metadata()
    assert {"price", "ask_level_price", "level_9"} <= values["ap9_out0"].types
    assert {"quantity", "level_volume", "bid"} <= values["volume_b3_out0"].types
    assert {"trade_vwap", "trade_price"} <= values["vwap_out0"].types
    assert "half_spread_fraction" in values["vw_halfspread_out0"].types
    assert "cross_weighted_trade_quantity" in values["trade_cross_pct_out0.sum"].types
    assert compatible(values["ap3_out0"], values["bp7_out0"])
    assert not compatible(values["ap3_out0"], values["volume_a3_out0"])
    assert "timestamp" in DEFAULT_TYPE_GRAPH.closure(values["session_start0"].types)


def test_timestamp_subtraction_produces_duration_ratio_alpha():
    values = inputdata_alpha_terminal_metadata()
    config = RiskMinerConfig(
        max_depth=3, min_formula_depth=3, max_tokens=8, max_stack=4,
        simulations=1, evaluation_batch_size=1, archive_size=1,
    )
    vocabulary = build_vocabulary(terminals=values, operators=tuple(
        schema for schema in __import__("flows.riskminer", fromlist=["default_operator_catalog"]).default_operator_catalog()
        if schema.name in {"sub", "div"}
    ))
    environment = TypedRPNEnvironment(config=config, vocabulary=vocabulary)
    state = environment.initial_state()
    for token in (
        "_ev_ts", "session_start0", "sub",
        "session_end0", "session_start0", "sub", "div", "END",
    ):
        state = environment.apply(state, vocabulary.by_name[token].token_id)
    value = environment.formula_value(state)
    assert value is not None
    assert "dimensionless" in value.semantics.types
