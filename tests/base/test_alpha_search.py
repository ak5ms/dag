from math import isfinite

import pytest
from deap import gp

from trading_dsl_engine.base.alpha_search import (
    PositiveScalar,
    PositiveIntScalar,
    feature_names_with_tags,
    futures_field_metadata,
    futures_type_relations,
    default_alpha_pnl,
    individual_to_expr,
    make_alpha_pset,
    ridge_pool_alpha_pnl,
    search_formulas,
)
from trading_dsl_engine.base.dsl import clip, xs_rank, var
from trading_dsl_engine.base.metadata import analyze_formula_metadata, metadata
from trading_dsl_engine.base.parser import Call, Expr


def test_default_pnl_and_ridge_pool_build_expected_formula_shapes():
    alpha = var("alpha")
    pnl = default_alpha_pnl(alpha, roll_rets=var("roll_rets"), is_tradable=var("is_tradable"), hl=1440)
    assert isinstance(pnl, Call)
    assert pnl.fn == "einsum"

    pooled = ridge_pool_alpha_pnl(
        alpha,
        [var("alpha0")],
        roll_rets=var("roll_rets"),
        hs=var("hs"),
        is_tradable=var("is_tradable"),
        hl=1440,
    )
    assert isinstance(pooled, Call)
    assert pooled.fn == "einsum"
    assert "Ridge" in repr(pooled)


def test_make_alpha_pset_uses_standard_deap_typed_groups():
    pset = make_alpha_pset(["x"], halflives=[5, 30.0, var("adaptive_hl")], shift_lags=[1, var("adaptive_lag")])
    assert isinstance(pset, gp.PrimitiveSetTyped)
    assert Expr in pset.terminals
    assert PositiveScalar in pset.terminals
    assert len(pset.terminals[Expr]) == 1
    assert len(pset.terminals[PositiveScalar]) == 3
    assert len(pset.terminals[PositiveIntScalar]) == 2
    assert any(primitive.name == "ewm" and primitive.args == [Expr, PositiveScalar] for primitive in pset.primitives[Expr])
    assert any(primitive.name == "shift" and primitive.args == [Expr, PositiveIntScalar] for primitive in pset.primitives[Expr])

    with pytest.raises(ValueError, match="Positive scalar"):
        make_alpha_pset(["x"], halflives=[0])
    with pytest.raises(ValueError, match="Positive integer"):
        make_alpha_pset(["x"], shift_lags=[1.5])


def test_futures_field_metadata_expands_user_schema_and_filters_tags():
    fields = futures_field_metadata(levels=range(2))
    assert fields["ap1_out0"]["range"] == ">0"
    assert {"price", "ask", "level_1", "front_month_contract"}.issubset(fields["ap1_out0"]["types"])
    assert fields["is_tradable_out0"]["range"] == "boolean"
    front_prices = feature_names_with_tags(fields, include=("price", "front_month_contract"))
    assert "mp_out0.close" in front_prices
    assert "mp_out1.close" not in front_prices


def test_filter_alpha_candidates_by_static_range_and_type_tags():
    fields = futures_field_metadata(levels=range(1))

    def is_bounded_dimensionless(expr):
        meta = analyze_formula_metadata(expr, fields)
        value_range = meta.get_range()
        return (
            "dimensionless" in meta.get_types()
            and isfinite(value_range.lower)
            and isfinite(value_range.upper)
        )

    candidates = [var("vw_halfspread_out0"), var("vwap_out0"), var("volume_out0")]
    valid = [candidate for candidate in candidates if is_bounded_dimensionless(candidate)]

    assert valid == [var("vw_halfspread_out0")]
    valid_meta = analyze_formula_metadata(valid[0], fields)
    assert valid_meta.get_range().as_tuple() == (0.0, 1.0)
    assert "dimensionless" in valid_meta.get_types()


def test_complex_alpha_node_units_and_type_relations():
    fields = futures_field_metadata(levels=range(1))
    config = metadata(fields, type_relations=futures_type_relations(levels=range(1)))
    expr = clip(xs_rank(var("mp_out0.open") / var("mp_out0.close")), -3.0, 3.0) + var("vw_halfspread_out0")
    meta = analyze_formula_metadata(expr, config)
    by_label = {node.label: node.metadata for node in meta.get_node_metadata()}

    assert by_label["mp_out0.open"].units.as_dict() == {"price": 1.0}
    assert by_label["mp_out0.close"].units.as_dict() == {"price": 1.0}
    assert "price" in by_label["mp_out0.open"].types
    assert "book_level" in by_label["mp_out0.open"].types
    assert by_label["div"].units.as_dict() == {}
    assert "ratio" in by_label["div"].types
    assert by_label["xs_rank"].units.as_dict() == {}
    assert "dimensionless" in by_label["xs_rank"].types
    assert by_label["clip"].units.as_dict() == {}
    assert by_label["vw_halfspread_out0"].units.as_dict() == {}
    assert "dimensionless" in by_label["vw_halfspread_out0"].types
    assert meta.get_units().as_dict() == {}
    assert "dimensionless" in meta.get_types()


def test_positive_price_ratio_range_metadata_is_nonnegative():
    fields = futures_field_metadata(levels=range(1))
    expr = var("mp_out0.open") / var("mp_out0.close")
    meta = analyze_formula_metadata(expr, fields)

    assert meta.get_range().lower == 0.0
    assert meta.get_range().upper == float("inf")


def test_rank_clip_range_metadata_for_price_ratio_alpha():
    fields = futures_field_metadata(levels=range(1))
    expr = clip(xs_rank(var("mp_out0.open") / var("mp_out0.close")), -3.0, 3.0)
    meta = analyze_formula_metadata(expr, fields)

    assert meta.get_range().as_tuple() == (0.0, 1.0)
    assert "dimensionless" in meta.get_types()


def test_depth_three_generation_example_filters_dimensionless_not_in_pool():
    fields = futures_field_metadata(levels=range(1))
    features = feature_names_with_tags(fields, include=("dimensionless", "front_month_contract"))
    pset = make_alpha_pset(features, halflives=[2, 5], shift_lags=[1])
    pool = [var("vw_halfspread_out0")]
    pool_keys = {repr(expr) for expr in pool}

    def is_bounded_dimensionless_new(expr):
        meta = analyze_formula_metadata(expr, fields)
        value_range = meta.get_range()
        return (
            repr(expr) not in pool_keys
            and "dimensionless" in meta.get_types()
            and isfinite(value_range.lower)
            and isfinite(value_range.upper)
        )

    out = search_formulas(
        pset,
        lambda candidate, pool_: 1.0,
        max_depth=3,
        initial_pool=pool,
        filters=[is_bounded_dimensionless_new],
        additive=lambda candidate, pool_, fitness: fitness > 0.0 and len(pool_) < 3,
        population_size=32,
        generations_per_depth=2,
        seed=11,
    )

    assert out
    assert all(depth <= 3 for _, _, depth in out)
    assert all(repr(expr) not in pool_keys for expr, _, _ in out)
    assert any("xs_rank" in repr(expr) and "clip" in repr(expr) for expr, _, _ in out)


def test_individual_to_expr_and_search_formulas_use_standard_deap_individuals():
    pset = make_alpha_pset(["x", "y"], halflives=[2, 5])
    individual = gp.PrimitiveTree.from_string("ewm(x, hl_0)", pset)
    expr = individual_to_expr(individual, pset)
    assert isinstance(expr, Call)
    assert expr.fn == "ewm"

    accepted_pool_sizes = []

    def objective(candidate, pool):
        accepted_pool_sizes.append(len(pool))
        return 1.0

    out = search_formulas(
        pset,
        objective,
        max_depth=2,
        filters=[lambda candidate: True],
        additive=lambda candidate, pool, fitness: fitness > 0.0 and len(pool) < 2,
        population_size=12,
        generations_per_depth=1,
        seed=7,
    )
    assert out
    assert len(out) <= 2
    assert all(isinstance(candidate, tuple) and len(candidate) == 3 for candidate in out)
    assert all(depth <= 2 for _, _, depth in out)
    assert accepted_pool_sizes[0] == 0
