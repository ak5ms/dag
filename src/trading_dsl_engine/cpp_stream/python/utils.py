"""Cheap cpp_stream DSL compositions built from native primitives."""

from __future__ import annotations

import math

from trading_dsl_engine.base.dsl import (
    Ridge,
    call,
    div,
    ewm_moment,
    ewm_var,
    get_coefficient,
    get_effective_df,
    get_effective_n,
    get_preds,
    get_r2,
    get_residual_variance,
    get_residuals,
    get_sse,
    get_sst,
    get_standard_error,
    get_tstat,
    mean,
    mul,
    pow,
    register_dsl_function,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    sub,
    sum,
)
from trading_dsl_engine.base.parser import Expr, Number, String


@register_dsl_function("ewm_std")
def ewm_std(x: Expr, halflife: float, min_periods: int = 0) -> Expr:
    return pow(ewm_var(x, halflife, min_periods), 0.5)


@register_dsl_function("ewm_skewness")
def ewm_skewness(x: Expr, halflife: float, min_periods: int = 0) -> Expr:
    variance = ewm_moment(x, halflife, 2, min_periods)
    third = ewm_moment(x, halflife, 3, min_periods)
    return div(third, pow(variance, 1.5))


@register_dsl_function("ewm_kurtosis")
def ewm_kurtosis(x: Expr, halflife: float, min_periods: int = 0) -> Expr:
    variance = ewm_moment(x, halflife, 2, min_periods)
    fourth = ewm_moment(x, halflife, 4, min_periods)
    return div(fourth, mul(variance, variance))


@register_dsl_function("xs_demean")
def xs_demean(x: Expr) -> Expr:
    return sub(x, mean(x, axis=1))


@register_dsl_function("xs_zscore")
def xs_zscore(x: Expr) -> Expr:
    return div(xs_demean(x), call("std", x, axis=1))


@register_dsl_function("xs_scale")
def xs_scale(x: Expr, scale: float = 1.0) -> Expr:
    absolute = pow(mul(x, x), 0.5)
    return mul(div(x, sum(absolute, axis=1)), scale)


@register_dsl_function("xs_direction")
def xs_direction(x: Expr) -> Expr:
    return div(x, pow(sum(mul(x, x), axis=1), 0.5))


@register_dsl_function("xs_vector_proj")
def xs_vector_proj(x: Expr, y: Expr) -> Expr:
    coefficient = div(sum(mul(x, y), axis=1), sum(mul(y, y), axis=1))
    return mul(coefficient, y)


@register_dsl_function("xs_vector_neut")
def xs_vector_neut(x: Expr, y: Expr) -> Expr:
    return sub(x, xs_vector_proj(x, y))


@register_dsl_function("rolling_range")
def rolling_range(x: Expr, periods: int, min_periods: int | None = None) -> Expr:
    kwargs = {} if min_periods is None else {"min_periods": min_periods}
    return sub(
        rolling_max(x, periods, **kwargs),
        rolling_min(x, periods, **kwargs),
    )


@register_dsl_function("rolling_zscore")
def rolling_zscore(x: Expr, periods: int, min_periods: int | None = None) -> Expr:
    kwargs = {} if min_periods is None else {"min_periods": min_periods}
    return div(
        sub(x, rolling_mean(x, periods, **kwargs)),
        rolling_std(x, periods, **kwargs),
    )


@register_dsl_function("rolling_scale")
def rolling_scale(
    x: Expr,
    periods: int,
    constant: float = 0.0,
    min_periods: int | None = None,
) -> Expr:
    kwargs = {} if min_periods is None else {"min_periods": min_periods}
    low = rolling_min(x, periods, **kwargs)
    high = rolling_max(x, periods, **kwargs)
    return call("add", div(sub(x, low), sub(high, low)), constant)


def _rettype_name(value: str | String) -> str:
    if isinstance(value, String):
        return value.value
    if isinstance(value, str):
        return value
    raise TypeError(
        "ts_regression rettype must be a descriptive name, not a numeric selector"
    )


def _numeric_literal(value: int | float | Number, name: str) -> float:
    if isinstance(value, Number):
        return float(value.value)
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"ts_regression {name} must be a numeric literal")


@register_dsl_function("ts_regression")
def ts_regression(
    y: Expr,
    x: Expr,
    periods: int,
    lag: int = 0,
    rettype: str = "residual",
    weights: Expr | float = 1.0,
    lambda_: float = 0.0,
) -> Expr:
    """EWM weighted-ridge regression with a named result projection.

    ``periods`` is the EWM half-life measured in input rows. The constant feature
    is coefficient 0 and ``x`` is coefficient 1.
    """

    period_value = _numeric_literal(periods, "periods")
    if not math.isfinite(period_value) or not period_value > 0.0:
        raise ValueError("ts_regression periods must be finite and > 0")
    lag_value = _numeric_literal(lag, "lag")
    lag_int = int(lag_value)
    if lag_value != lag_int or lag_int < 0:
        raise ValueError("ts_regression lag must be a nonnegative integer")
    lagged_x = x if lag_int == 0 else call("shift", x, lag_int, lag_int)
    model = Ridge(
        1.0,
        lagged_x,
        y=y,
        weights=weights,
        hl=periods,
        lambda_=lambda_,
    )
    name = _rettype_name(rettype).strip().lower()
    projections = {
        "residual": get_residuals,
        "residuals": get_residuals,
        "prediction": get_preds,
        "predictions": get_preds,
        "intercept": lambda value: get_coefficient(value, 0),
        "beta": lambda value: get_coefficient(value, 1),
        "slope": lambda value: get_coefficient(value, 1),
        "sse": get_sse,
        "sst": get_sst,
        "r2": get_r2,
        "residual_variance": get_residual_variance,
        "intercept_stderr": lambda value: get_standard_error(value, 0),
        "beta_stderr": lambda value: get_standard_error(value, 1),
        "slope_stderr": lambda value: get_standard_error(value, 1),
        "intercept_tstat": lambda value: get_tstat(value, 0),
        "beta_tstat": lambda value: get_tstat(value, 1),
        "slope_tstat": lambda value: get_tstat(value, 1),
        "effective_df": get_effective_df,
        "effective_n": get_effective_n,
    }
    try:
        return projections[name](model)
    except KeyError as exc:
        raise ValueError(
            f"unknown ts_regression rettype {name!r}; expected one of "
            f"{', '.join(sorted(projections))}"
        ) from exc


__all__ = [
    "ewm_std",
    "ewm_skewness",
    "ewm_kurtosis",
    "xs_demean",
    "xs_zscore",
    "xs_scale",
    "xs_direction",
    "xs_vector_proj",
    "xs_vector_neut",
    "rolling_range",
    "rolling_zscore",
    "rolling_scale",
    "ts_regression",
]
