from __future__ import annotations

import pytest

from trading_dsl_engine.base import dsl


def test_expr_pipe_supports_positional_and_keyword_injection():
    expression = dsl.var("x")
    positional = expression.pipe(lambda value, amount: dsl.add(value, amount), 2)
    keyword = expression.pipe((lambda *, value, amount: dsl.add(value, amount), "value"), amount=2)
    assert str(positional) == str(dsl.add(expression, 2))
    assert str(keyword) == str(dsl.add(expression, 2))


def test_expr_pipe_rejects_duplicate_keyword_target():
    expression = dsl.var("x")
    with pytest.raises(ValueError, match="both the pipe target"):
        expression.pipe((lambda **kwargs: kwargs, "value"), value=expression)


def test_expr_method_chain_matches_function_form():
    expression = dsl.var("roll_rets")
    assert str(expression.xs_zscore()) == str(dsl.xs_zscore(expression))


def test_registered_function_is_resolved_lazily_as_expr_method():
    name = "test_registered_chain_operator"
    expression = dsl.var("x")

    def registered(value, *, amount=1):
        return dsl.add(value, amount)

    registry = None
    for namespace in (vars(dsl),):
        for candidate in namespace.values():
            if isinstance(candidate, dict) and name not in candidate:
                if any("dsl" in str(key).lower() or "function" in str(key).lower() for key in candidate):
                    registry = candidate
                    break
        if registry is not None:
            break
    setattr(dsl, name, registered)
    try:
        assert str(getattr(expression, name)(amount=3)) == str(registered(expression, amount=3))
    finally:
        delattr(dsl, name)


def test_cross_sectional_vector_arguments_broadcast_scalar_literals():
    expression = dsl.var("x")
    weighted = dsl.xs_weighted_mean(expression, w=1)
    projection = dsl.xs_vector_projection(expression, 1)
    assert "xs_weighted_mean" in str(weighted)
    assert "xs_vector_projection" in str(projection)
    assert "fillna" in str(weighted)
    assert "fillna" in str(projection)
