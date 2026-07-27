from inspect import signature

from trading_dsl_engine.base.dsl import Ridge, ewm, round as dsl_round, shift, var


def test_python_dsl_helpers_expose_operator_signatures_for_ides():
    assert list(signature(Ridge).parameters) == ["features", "y", "weights", "hl", "lambda_", "lam", "nonneg"]
    assert list(signature(ewm).parameters) == ["x", "span", "min_periods", "ignore_na", "adjust"]
    assert list(signature(shift).parameters) == ["x", "lag", "max_lag"]


def test_round_helper_without_frequency_builds_builtin_call_without_recursion():
    expr = dsl_round(var("close"))
    assert expr.fn == "round"
    assert expr.args == (var("close"),)
