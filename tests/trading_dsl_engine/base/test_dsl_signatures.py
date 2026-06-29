from inspect import signature

from trading_dsl_engine.base.dsl import Ridge, ewm, shift


def test_python_dsl_helpers_expose_operator_signatures_for_ides():
    assert list(signature(Ridge).parameters) == ["features", "y", "weights", "hl", "lambda_", "lam"]
    assert list(signature(ewm).parameters) == ["x", "span", "min_periods", "ignore_na", "adjust"]
    assert list(signature(shift).parameters) == ["x", "lag", "max_lag"]
