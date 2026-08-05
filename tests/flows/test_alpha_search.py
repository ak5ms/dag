from math import isfinite

import pytest
from deap import gp

import flows.alpha_search as alpha_search
from flows.alpha_search import (
    OperatorSpec,
    PositiveScalar,
    PositiveIntScalar,
    default_alpha_pnl,
    individual_to_expr,
    make_alpha_pset,
    ridge_pool_alpha_pnl,
    search_formulas,
)
from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.base.metadata import analyze_formula_metadata
from trading_dsl_engine.base.terminals import feature_names_with_tags, futures_field_metadata
from trading_dsl_engine.base.parser import Call, Expr


def test_default_pnl_and_ridge_pool_build_expected_formula_shapes():
    alpha = var("alpha")
    pnl = default_alpha_pnl(alpha, roll_rets=var("roll_rets"), is_tradable=var("is_tradable"), hl=1440)
    assert isinstance(pnl, Call)
    assert pnl.fn == "mul"

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
    assert any(primitive.name == "positive" and primitive.args == [Expr] for primitive in pset.primitives[PositiveScalar])
    assert any(primitive.name == "positive_int" and primitive.args == [Expr] for primitive in pset.primitives[PositiveIntScalar])

    with pytest.raises(ValueError, match="Positive scalar"):
        make_alpha_pset(["x"], halflives=[0])
    with pytest.raises(ValueError, match="Positive integer"):
        make_alpha_pset(["x"], shift_lags=[1.5])


def test_dynamic_parameter_subtrees_compile_to_normal_dsl_expressions():
    pset = make_alpha_pset(["x", "adaptive"], halflives=[5], shift_lags=[1])

    ewm_tree = gp.PrimitiveTree.from_string("ewm(x, positive(xs_rank(adaptive)))", pset)
    ewm_expr = individual_to_expr(ewm_tree, pset)
    assert isinstance(ewm_expr, Call)
    assert ewm_expr.fn == "ewm"
    assert "xs_rank" in repr(ewm_expr)

    shift_tree = gp.PrimitiveTree.from_string("shift(x, positive_int(abs(adaptive)))", pset)
    shift_expr = individual_to_expr(shift_tree, pset)
    assert isinstance(shift_expr, Call)
    assert shift_expr.fn == "shift"
    assert "floor" in repr(shift_expr)


def test_same_type_family_uses_multi_type_intersection_algorithm(monkeypatch):
    class FakeMeta:
        def __init__(self, types):
            self._types = types

        def get_types(self):
            return set(self._types)

    def fake_analyze(expr, config):
        text = repr(expr)
        if "price" in text:
            return FakeMeta({"price", "market_data"})
        if "volume" in text:
            return FakeMeta({"volume", "market_data"})
        return FakeMeta({"dimensionless"})

    monkeypatch.setattr(alpha_search, "analyze_formula_metadata", fake_analyze)
    pset = make_alpha_pset(
        ["price", "volume"],
        operators=[OperatorSpec("add", alpha_search.add, 2, "same_type")],
    )

    # Multiple node types are supported: the shared market_data type permits it.
    expr = individual_to_expr(gp.PrimitiveTree.from_string("add(price, volume)", pset), pset)
    assert expr.fn == "add"

    def disjoint_analyze(expr, config):
        return FakeMeta({"price"} if "price" in repr(expr) else {"volume"})

    monkeypatch.setattr(alpha_search, "analyze_formula_metadata", disjoint_analyze)
    with pytest.raises(ValueError, match="intersecting semantic type"):
        individual_to_expr(gp.PrimitiveTree.from_string("add(price, volume)", pset), pset)


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
