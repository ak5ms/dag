import pytest
from deap import gp

from trading_dsl_engine.base.alpha_search import (
    PositiveScalar,
    default_alpha_pnl,
    individual_to_expr,
    make_alpha_pset,
    ridge_pool_alpha_pnl,
    search_formulas,
)
from trading_dsl_engine.base.dsl import var
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
    pset = make_alpha_pset(["x"], halflives=[5, 30.0, var("adaptive_hl")])
    assert isinstance(pset, gp.PrimitiveSetTyped)
    assert Expr in pset.terminals
    assert PositiveScalar in pset.terminals
    assert len(pset.terminals[Expr]) == 1
    assert len(pset.terminals[PositiveScalar]) == 3
    assert any(primitive.name == "ewm" and primitive.args == [Expr, PositiveScalar] for primitive in pset.primitives[Expr])

    with pytest.raises(ValueError, match="Positive scalar"):
        make_alpha_pset(["x"], halflives=[0])


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
