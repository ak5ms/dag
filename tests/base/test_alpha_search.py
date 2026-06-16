import pytest

from trading_dsl_engine.base.alpha_search import (
    POSITIVE_SCALAR,
    VECTOR,
    AlphaPrimitive,
    AlphaPrimitiveSet,
    FormulaAlphaSearch,
    SearchScheme,
    default_alpha_pnl,
    dimensionless_filter,
    expr_depth,
    ridge_pool_alpha_pnl,
)
from trading_dsl_engine.base.dsl import ewm, var
from trading_dsl_engine.base.parser import Call


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


def test_typed_primitive_set_groups_halflives_and_rejects_nonpositive_literals():
    pset = AlphaPrimitiveSet.default(["x"], halflives=[5, 30.0, var("adaptive_hl")])
    assert [terminal.kind for terminal in pset.terminals_for(VECTOR)] == [VECTOR]
    assert len(pset.terminals_for(POSITIVE_SCALAR)) == 3

    ewm_primitives = [primitive for primitive in pset.primitives_for(VECTOR) if primitive.name == "ewm"]
    assert ewm_primitives == [AlphaPrimitive("ewm", (VECTOR, POSITIVE_SCALAR), VECTOR, ewm)]

    with pytest.raises(ValueError, match="Positive scalar"):
        AlphaPrimitiveSet.default(["x"], halflives=[0])


def test_formula_alpha_search_respects_types_depth_filters_and_additive_pool():
    pset = AlphaPrimitiveSet.default(["x", "y"], halflives=[2, 5])
    accepted_pool_sizes = []

    def objective(expr, pool):
        accepted_pool_sizes.append(len(pool))
        return 1.0 if expr_depth(expr) <= 2 else -1.0

    search = FormulaAlphaSearch(
        pset,
        objective,
        filters=[dimensionless_filter({"fields": {"x": {}, "y": {}}})],
        additive=lambda expr, pool, fitness: fitness > 0.0 and len(pool) < 2,
        scheme=SearchScheme(population_size=12, generations_per_depth=1, seed=7),
    )
    out = search.search(2)
    assert out
    assert len(out) <= 2
    assert all(candidate.kind == VECTOR for candidate in out)
    assert all(candidate.depth <= 2 for candidate in out)
    assert all(expr_depth(candidate.expr) <= candidate.depth for candidate in out)
    assert accepted_pool_sizes[0] == 0
