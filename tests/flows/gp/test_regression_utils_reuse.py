from __future__ import annotations

from flows.gp.regression import (
    temporal_poly_regression_residual,
    temporal_ridge_projection,
    xs_regression_neutralize,
)
from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.parser import Call
from trading_dsl_engine.cpp_stream.python import utils as cpp_stream_utils


def test_temporal_residual_delegates_to_utils_ts_regression():
    y = dsl.var("y")
    x = dsl.var("x")
    expected = cpp_stream_utils.ts_regression(
        y,
        x,
        20,
        lag=0,
        rettype="residual",
        weights=1.0,
        lambda_=0.0,
    )
    actual = temporal_ridge_projection("residual", y, x, 20)
    assert actual == expected


def test_scalar_temporal_projection_only_adds_gp_row_broadcast():
    y = dsl.var("y")
    x = dsl.var("x")
    expected = cpp_stream_utils.ts_regression(
        y,
        x,
        20,
        lag=0,
        rettype="r2",
        weights=1.0,
        lambda_=0.0,
    )
    actual = temporal_ridge_projection("r2", y, x, 20)
    assert isinstance(actual, Call) and actual.fn == "where"
    assert actual.args[1] == expected
    assert actual.args[2] == expected


def test_poly_regression_delegates_to_utils():
    y = dsl.var("y")
    x = dsl.var("x")
    for degree in (1, 2, 3):
        expected = cpp_stream_utils.ts_poly_regression(
            y,
            x,
            20,
            k=degree,
            weights=1.0,
            lambda_=0.0,
        )
        assert temporal_poly_regression_residual(y, x, 20, degree) == expected


def test_xs_regression_neutralization_delegates_to_utils():
    y = dsl.var("y")
    x = dsl.var("x")
    assert xs_regression_neutralize(y, x) == cpp_stream_utils.xs_regression_neut(y, x)
