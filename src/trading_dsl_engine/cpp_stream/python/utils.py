"""Cheap cpp_stream DSL compositions built from native primitives."""

from __future__ import annotations

import math
from itertools import product

from trading_dsl_engine.base.dsl import (
    Ridge,
    add,
    and_,
    call,
    div,
    ewm,
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
    ge,
    gt,
    eq,
    isfinite,
    isnan,
    le,
    logical_not,
    lt,
    maximum,
    mean,
    minimum,
    mul,
    pow,
    register_dsl_function,
    reduce_max,
    reduce_min,
    rolling_max,
    rolling_mean,
    rolling_min,
    rolling_std,
    rolling_sum,
    sqrt,
    std,
    sub,
    sum,
    self_,
    where,
    vec_quantile,
    xs_count,
    xs_generalized_rank,
    xs_max,
    xs_mean,
    xs_min,
    xs_pct_rank,
    xs_quantile_value,
    xs_regression_projection,
    xs_std,
    xs_sum,
    xs_vector_projection,
    xs_weighted_mean,
)
from trading_dsl_engine.base.parser import Expr, Number, String


def _literal_int(value: int | float | Number, name: str) -> int:
    raw = float(value.value) if isinstance(value, Number) else float(value)
    result = int(raw)
    if not math.isfinite(raw) or raw != result:
        raise TypeError(f"{name} must be an integer literal")
    return result


def _product_expr(values) -> Expr:
    values = list(values)
    if not values:
        return Number(1.0)
    result = values[0]
    for value in values[1:]:
        result = mul(result, value)
    return result


def _sum_expr(values) -> Expr:
    values = list(values)
    if not values:
        return Number(0.0)
    result = values[0]
    for value in values[1:]:
        result = add(result, value)
    return result


def _complete_case(inputs: tuple[Expr, ...]) -> Expr:
    flags = [isfinite(value) for value in inputs]
    result = flags[0]
    for flag in flags[1:]:
        result = and_(result, flag)
    return result


def _ewm_raw_moment(
    inputs: tuple[Expr, ...],
    orders: tuple[int, ...],
    span,
    min_periods,
    ignore_na,
    adjust,
) -> Expr:
    if not any(orders):
        return Number(1.0)
    monomial = _product_expr(
        pow(value, order) for value, order in zip(inputs, orders) if order
    )
    complete = _complete_case(inputs)
    observation = where(complete, monomial, float("nan"))
    return ewm(
        observation,
        span,
        min_periods=min_periods,
        ignore_na=ignore_na,
        adjust=adjust,
    )


def _ewm_central_moment(
    inputs: tuple[Expr, ...],
    orders: tuple[int, ...],
    span,
    min_periods,
    ignore_na,
    adjust,
) -> Expr:
    means = tuple(
        _ewm_raw_moment(
            inputs,
            tuple(1 if index == mean_index else 0 for index in range(len(inputs))),
            span,
            min_periods,
            ignore_na,
            adjust,
        )
        for mean_index in range(len(inputs))
    )
    terms = []
    for exponents in product(*(range(order + 1) for order in orders)):
        coefficient = 1
        factors = []
        for order, exponent, mean_value in zip(orders, exponents, means):
            coefficient *= math.comb(order, exponent)
            centered_power = order - exponent
            if centered_power:
                if centered_power & 1:
                    coefficient = -coefficient
                factors.append(pow(mean_value, centered_power))
        factors.append(
            _ewm_raw_moment(
                inputs,
                tuple(exponents),
                span,
                min_periods,
                ignore_na,
                adjust,
            )
        )
        term = _product_expr(factors)
        terms.append(term if coefficient == 1 else mul(float(coefficient), term))
    return _sum_expr(terms)


def _safe_variance(value: Expr) -> Expr:
    return maximum(value, 0.0)


def _safe_ratio(numerator: Expr, denominator: Expr) -> Expr:
    valid = and_(gt(denominator, 0.0), isfinite(denominator))
    return where(valid, div(numerator, denominator), float("nan"))


@register_dsl_function("ewm_moment")
def ewm_moment(
    x: Expr,
    span: float,
    k: int = 2,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    """Exponentially weighted central moment composed from :func:`ewm`."""

    order = _literal_int(k, "ewm_moment k")
    if not 1 <= order <= 4:
        raise ValueError("ewm_moment k must be in [1, 4]")
    return _ewm_central_moment(
        (x,), (order,), span, min_periods, ignore_na, adjust
    )


@register_dsl_function("ewm_var")
def ewm_var(
    x: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    return _safe_variance(
        _ewm_central_moment(
            (x,), (2,), span, min_periods, ignore_na, adjust
        )
    )


@register_dsl_function("ewm_std")
def ewm_std(
    x: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    return sqrt(ewm_var(x, span, min_periods, ignore_na, adjust))


@register_dsl_function("ewm_skewness")
def ewm_skewness(
    x: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    variance = ewm_var(x, span, min_periods, ignore_na, adjust)
    third = _ewm_central_moment(
        (x,), (3,), span, min_periods, ignore_na, adjust
    )
    return _safe_ratio(third, mul(variance, sqrt(variance)))


@register_dsl_function("ewm_kurtosis")
def ewm_kurtosis(
    x: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    variance = ewm_var(x, span, min_periods, ignore_na, adjust)
    fourth = _ewm_central_moment(
        (x,), (4,), span, min_periods, ignore_na, adjust
    )
    return _safe_ratio(fourth, mul(variance, variance))


@register_dsl_function("ewm_cov")
def ewm_cov(
    x: Expr,
    y: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    return _ewm_central_moment(
        (x, y), (1, 1), span, min_periods, ignore_na, adjust
    )


@register_dsl_function("ewm_corr")
def ewm_corr(
    x: Expr,
    y: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    inputs = (x, y)
    covariance = _ewm_central_moment(
        inputs, (1, 1), span, min_periods, ignore_na, adjust
    )
    var_x = _safe_variance(_ewm_central_moment(
        inputs, (2, 0), span, min_periods, ignore_na, adjust
    ))
    var_y = _safe_variance(_ewm_central_moment(
        inputs, (0, 2), span, min_periods, ignore_na, adjust
    ))
    return _safe_ratio(covariance, sqrt(mul(var_x, var_y)))


@register_dsl_function("ewm_co_skewness")
def ewm_co_skewness(
    y: Expr,
    x: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    inputs = (y, x)
    central = _ewm_central_moment(
        inputs, (1, 2), span, min_periods, ignore_na, adjust
    )
    var_y = _safe_variance(_ewm_central_moment(
        inputs, (2, 0), span, min_periods, ignore_na, adjust
    ))
    var_x = _safe_variance(_ewm_central_moment(
        inputs, (0, 2), span, min_periods, ignore_na, adjust
    ))
    return _safe_ratio(central, mul(sqrt(var_y), var_x))


@register_dsl_function("ewm_co_kurtosis")
def ewm_co_kurtosis(
    y: Expr,
    x: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    inputs = (y, x)
    central = _ewm_central_moment(
        inputs, (1, 3), span, min_periods, ignore_na, adjust
    )
    var_y = _safe_variance(_ewm_central_moment(
        inputs, (2, 0), span, min_periods, ignore_na, adjust
    ))
    var_x = _safe_variance(_ewm_central_moment(
        inputs, (0, 2), span, min_periods, ignore_na, adjust
    ))
    return _safe_ratio(central, mul(sqrt(var_y), mul(var_x, sqrt(var_x))))


@register_dsl_function("ewm_triple_corr")
def ewm_triple_corr(
    x: Expr,
    y: Expr,
    z: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    inputs = (x, y, z)
    central = _ewm_central_moment(
        inputs, (1, 1, 1), span, min_periods, ignore_na, adjust
    )
    variances = [
        _safe_variance(_ewm_central_moment(
            inputs,
            tuple(2 if index == target else 0 for index in range(3)),
            span,
            min_periods,
            ignore_na,
            adjust,
        ))
        for target in range(3)
    ]
    return _safe_ratio(central, sqrt(_product_expr(variances)))


@register_dsl_function("ewm_partial_corr")
def ewm_partial_corr(
    x: Expr,
    y: Expr,
    z: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    inputs = (x, y, z)
    variances = [
        _safe_variance(_ewm_central_moment(
            inputs,
            tuple(2 if index == target else 0 for index in range(3)),
            span,
            min_periods,
            ignore_na,
            adjust,
        ))
        for target in range(3)
    ]

    def corr(left: int, right: int) -> Expr:
        orders = tuple(
            1 if index in {left, right} else 0 for index in range(3)
        )
        covariance = _ewm_central_moment(
            inputs, orders, span, min_periods, ignore_na, adjust
        )
        return _safe_ratio(
            covariance, sqrt(mul(variances[left], variances[right]))
        )

    rxy, rxz, ryz = corr(0, 1), corr(0, 2), corr(1, 2)
    numerator = sub(rxy, mul(rxz, ryz))
    denominator = mul(
        sqrt(_safe_variance(sub(1.0, mul(rxz, rxz)))),
        sqrt(_safe_variance(sub(1.0, mul(ryz, ryz)))),
    )
    return _safe_ratio(numerator, denominator)


def _literal_bool(value, name: str) -> bool:
    from trading_dsl_engine.base.parser import Identifier

    if isinstance(value, bool):
        return value
    if isinstance(value, Identifier) and value.name in {"True", "False"}:
        return value.name == "True"
    raw = float(value.value) if isinstance(value, Number) else float(value)
    if raw not in (0.0, 1.0):
        raise TypeError(f"{name} must be a boolean literal")
    return bool(raw)


def _literal_text(value, name: str) -> str:
    if isinstance(value, String):
        return value.value
    if isinstance(value, str):
        return value
    raise TypeError(f"{name} must be a string literal")


def _literal_number(value, name: str) -> float:
    raw = float(value.value) if isinstance(value, Number) else float(value)
    if not math.isfinite(raw):
        raise TypeError(f"{name} must be a finite numeric literal")
    return raw


@register_dsl_function("log")
def log(x: Expr) -> Expr:
    return call("ln", x)


@register_dsl_function("inverse")
def inverse(x: Expr) -> Expr:
    return div(1.0, x)


@register_dsl_function("log_diff")
def log_diff(x: Expr) -> Expr:
    return sub(call("ln", x), call("ln", call("shift", x, 1, 1)))


@register_dsl_function("max")
def elementwise_max(x: Expr, y: Expr, *rest: Expr) -> Expr:
    result = maximum(x, y)
    for value in rest:
        result = maximum(result, value)
    return result


@register_dsl_function("min")
def elementwise_min(x: Expr, y: Expr, *rest: Expr) -> Expr:
    result = minimum(x, y)
    for value in rest:
        result = minimum(result, value)
    return result


@register_dsl_function("clamp")
def clamp(
    x: Expr,
    lower: float = 0.0,
    upper: float = 0.0,
    inverse: bool = False,
    mask="",
) -> Expr:
    if _literal_bool(inverse, "clamp inverse"):
        if isinstance(mask, String):
            replacement = float("nan") if mask.value == "" else float(mask.value)
        elif isinstance(mask, str):
            replacement = float("nan") if mask == "" else float(mask)
        else:
            replacement = mask
        return where(and_(ge(x, lower), le(x, upper)), replacement, x)
    return minimum(maximum(x, lower), upper)


@register_dsl_function("nan_mask")
def nan_mask(x: Expr, mask: Expr) -> Expr:
    return where(lt(mask, 0.0), float("nan"), x)


@register_dsl_function("nan_out")
def nan_out(
    x: Expr,
    lower: float | None = None,
    upper: float | None = None,
) -> Expr:
    if lower is None and upper is None:
        raise ValueError("nan_out requires lower and/or upper")
    result = x
    if lower is not None:
        result = where(lt(x, lower), float("nan"), result)
    if upper is not None:
        result = where(gt(x, upper), float("nan"), result)
    return result


@register_dsl_function("replace")
def replace(x: Expr, target, dest) -> Expr:
    targets = _literal_text(target, "replace target").replace(",", " ").split()
    destinations = _literal_text(dest, "replace dest").replace(",", " ").split()
    if len(targets) != len(destinations) or not targets:
        raise ValueError("replace target and dest must contain equally many values")

    def parse_value(text: str) -> float:
        return float("nan") if text.strip().lower() == "nan" else float(text)

    result = x
    for source_text, dest_text in zip(targets, destinations):
        source = parse_value(source_text)
        destination = parse_value(dest_text)
        condition = isnan(x) if math.isnan(source) else eq(x, source)
        result = where(condition, destination, result)
    return result


@register_dsl_function("reverse")
def reverse(x: Expr) -> Expr:
    return mul(-1.0, x)


@register_dsl_function("round_df")
def round_df(x: Expr, decimals: int) -> Expr:
    factor = pow(10.0, decimals)
    return div(call("round", mul(x, factor)), factor)


@register_dsl_function("round_down")
def round_down(x: Expr, f: float = 1.0) -> Expr:
    return mul(call("floor", div(x, f)), f)


@register_dsl_function("signed_power")
@register_dsl_function("signedpower")
def signed_power(x: Expr, y: Expr) -> Expr:
    return mul(call("sign", x), pow(call("abs", x), y))


@register_dsl_function("s_log_1p")
@register_dsl_function("clip_with_log")
def s_log_1p(x: Expr) -> Expr:
    return mul(call("sign", x), call("ln", add(1.0, call("abs", x))))


@register_dsl_function("to_nan")
def to_nan(x: Expr, value: float = 0.0, reverse: bool = False) -> Expr:
    if _literal_bool(reverse, "to_nan reverse"):
        return where(isnan(x), value, x)
    return where(eq(x, value), float("nan"), x)


@register_dsl_function("negate")
def negate(x: Expr) -> Expr:
    return logical_not(x)


@register_dsl_function("is_not_nan")
def is_not_nan(x: Expr) -> Expr:
    return logical_not(isnan(x))


@register_dsl_function("logical_and")
@register_dsl_function("and")
def logical_and(x: Expr, y: Expr) -> Expr:
    return and_(x, y)


@register_dsl_function("logical_or")
@register_dsl_function("or")
def logical_or(x: Expr, y: Expr) -> Expr:
    return call("or_", x, y)


@register_dsl_function("equal")
def equal(x: Expr, y: Expr) -> Expr:
    return eq(x, y)


@register_dsl_function("less")
def less(x: Expr, y: Expr) -> Expr:
    return lt(x, y)


@register_dsl_function("if_else")
def if_else(condition: Expr, true: Expr, false: Expr) -> Expr:
    return where(condition, true, false)


@register_dsl_function("is_nan")
def is_nan(x: Expr) -> Expr:
    return isnan(x)


@register_dsl_function("is_finite")
def is_finite(x: Expr) -> Expr:
    return isfinite(x)


@register_dsl_function("is_not_finite")
def is_not_finite(x: Expr) -> Expr:
    return logical_not(isfinite(x))


@register_dsl_function("convert_float")
def convert_float(x: Expr) -> Expr:
    return add(x, 0.0)


@register_dsl_function("arc_cos")
def arc_cos(x: Expr) -> Expr:
    return call("acos", x)


@register_dsl_function("arc_sin")
def arc_sin(x: Expr) -> Expr:
    return call("asin", x)


@register_dsl_function("arc_tan")
def arc_tan(x: Expr) -> Expr:
    return call("arctan", x)


@register_dsl_function("sigmoid")
def sigmoid(x: Expr) -> Expr:
    return div(1.0, add(1.0, call("exp", mul(-1.0, x))))


@register_dsl_function("left_tail")
def left_tail(x: Expr, maximum: float = 0.0) -> Expr:
    return where(gt(x, maximum), float("nan"), x)


@register_dsl_function("right_tail")
def right_tail(x: Expr, minimum: float = 0.0) -> Expr:
    return where(lt(x, minimum), float("nan"), x)


@register_dsl_function("tail")
def tail(x: Expr, lower: float = 0.0, upper: float = 0.0, newval: float = 0.0) -> Expr:
    return where(and_(gt(x, lower), lt(x, upper)), newval, x)


@register_dsl_function("left_right_tail")
def left_right_tail(x: Expr, minimum: float, maximum: float) -> Expr:
    return where(
        and_(le(minimum, x), le(x, maximum)),
        x,
        float("nan"),
    )


@register_dsl_function("pasteurize")
def pasteurize(x: Expr) -> Expr:
    return call("purify", x)


@register_dsl_function("get_df")
def get_df(inp: Expr, val: float) -> Expr:
    return where(isfinite(add(mul(inp, 0.0), 0.0)), val, val)


@register_dsl_function("bucket")
def bucket(
    x: Expr,
    buckets=None,
    range_=None,
    skip_begin: bool = False,
    skip_end: bool = False,
    skip_both: bool = False,
    nan_group: bool = False,
    **kwargs,
) -> Expr:
    range_value = kwargs.pop("range", range_)
    skip_begin = kwargs.pop("skipBegin", skip_begin)
    skip_end = kwargs.pop("skipEnd", skip_end)
    skip_both = kwargs.pop("skipBoth", skip_both)
    nan_group = kwargs.pop("NANGroup", nan_group)
    if kwargs:
        raise TypeError(f"unknown bucket arguments: {sorted(kwargs)}")
    if (buckets is None) == (range_value is None):
        raise ValueError("bucket requires exactly one of buckets or range")
    if buckets is not None:
        boundaries = [
            float(value)
            for value in _literal_text(buckets, "bucket buckets")
            .replace(",", " ")
            .split()
        ]
    else:
        values = [
            float(value)
            for value in _literal_text(range_value, "bucket range")
            .replace(",", " ")
            .split()
        ]
        if len(values) != 3 or values[2] <= 0.0:
            raise ValueError("bucket range must be 'start,end,positive_step'")
        start, end, step = values
        count = max(0, int(math.ceil((end - start) / step)))
        boundaries = [start + step * index for index in range(1, count + 1)]
    boundaries = sorted(set(boundaries))
    index = _sum_expr(where(ge(x, boundary), 1.0, 0.0) for boundary in boundaries)
    drop_begin = _literal_bool(skip_begin, "bucket skip_begin")
    drop_end = _literal_bool(skip_end, "bucket skip_end")
    if _literal_bool(skip_both, "bucket skip_both"):
        drop_begin = drop_end = True
    if drop_begin:
        index = where(eq(index, 0.0), float("nan"), sub(index, 1.0))
    if drop_end:
        last = float(len(boundaries) - (1 if drop_begin else 0))
        index = where(eq(index, last), float("nan"), index)
    if _literal_bool(nan_group, "bucket nan_group"):
        return where(isfinite(x), index, float(len(boundaries) + 1))
    return where(isfinite(x), index, float("nan"))


@register_dsl_function("vec_avg")
def vec_avg(x: Expr) -> Expr:
    return mean(x, axis=-1)


@register_dsl_function("vec_choose")
def vec_choose(x: Expr, nth: int) -> Expr:
    return call("col", x, nth)


@register_dsl_function("vec_count")
def vec_count(x: Expr) -> Expr:
    return sum(isfinite(x), axis=-1)


@register_dsl_function("vec_ir")
def vec_ir(x: Expr) -> Expr:
    return _safe_ratio(mean(x, axis=-1), std(x, axis=-1))


def _vec_raw_moments(x: Expr) -> tuple[Expr, Expr, Expr, Expr]:
    return tuple(mean(pow(x, order), axis=-1) for order in range(1, 5))


@register_dsl_function("vec_skewness")
def vec_skewness(x: Expr) -> Expr:
    first, second, third, _ = _vec_raw_moments(x)
    variance = _safe_variance(sub(second, mul(first, first)))
    central = add(
        sub(third, mul(3.0, mul(first, second))),
        mul(2.0, pow(first, 3.0)),
    )
    return _safe_ratio(central, mul(variance, sqrt(variance)))


@register_dsl_function("vec_kurtosis")
def vec_kurtosis(x: Expr) -> Expr:
    first, second, third, fourth = _vec_raw_moments(x)
    variance = _safe_variance(sub(second, mul(first, first)))
    central = add(
        sub(fourth, mul(4.0, mul(first, third))),
        sub(mul(6.0, mul(mul(first, first), second)), mul(3.0, pow(first, 4.0))),
    )
    return _safe_ratio(central, mul(variance, variance))


@register_dsl_function("vec_max")
def vec_max(x: Expr) -> Expr:
    return reduce_max(x, axis=-1)


@register_dsl_function("vec_min")
def vec_min(x: Expr) -> Expr:
    return reduce_min(x, axis=-1)


@register_dsl_function("vec_norm")
def vec_norm(x: Expr) -> Expr:
    return sum(call("abs", x), axis=-1)


@register_dsl_function("vec_percentage")
def vec_percentage(x: Expr, percentage: float = 0.5) -> Expr:
    return vec_quantile(x, percentage)


@register_dsl_function("vec_powersum")
def vec_powersum(x: Expr, constant: float = 2.0) -> Expr:
    return sum(pow(x, constant), axis=-1)


@register_dsl_function("vec_range")
def vec_range(x: Expr) -> Expr:
    return sub(reduce_max(x, axis=-1), reduce_min(x, axis=-1))


@register_dsl_function("vec_stddev")
def vec_stddev(x: Expr) -> Expr:
    return std(x, axis=-1)


@register_dsl_function("vec_sum")
def vec_sum(x: Expr) -> Expr:
    return sum(x, axis=-1)


@register_dsl_function("xs_demean")
def xs_demean(x: Expr) -> Expr:
    return sub(x, xs_mean(x))


@register_dsl_function("xs_zscore")
def xs_zscore(x: Expr) -> Expr:
    return _safe_ratio(xs_demean(x), xs_std(x))


@register_dsl_function("xs_scale")
def xs_scale(
    x: Expr,
    scale: float = 1.0,
    longscale=None,
    shortscale=None,
) -> Expr:
    if longscale is None and shortscale is None:
        return mul(_safe_ratio(x, xs_sum(call("abs", x))), scale)
    long_book = 0.0 if longscale is None else longscale
    short_book = 0.0 if shortscale is None else shortscale
    positive = where(gt(x, 0.0), x, 0.0)
    negative = where(lt(x, 0.0), x, 0.0)
    long_leg = mul(_safe_ratio(positive, xs_sum(positive)), long_book)
    short_leg = mul(
        _safe_ratio(negative, xs_sum(call("abs", negative))),
        short_book,
    )
    return add(long_leg, short_leg)


@register_dsl_function("xs_direction")
def xs_direction(x: Expr) -> Expr:
    return _safe_ratio(x, sqrt(xs_sum(mul(x, x))))


@register_dsl_function("xs_vector_proj")
def xs_vector_proj(x: Expr, y: Expr) -> Expr:
    return xs_vector_projection(x, y)


@register_dsl_function("xs_vector_neut")
def xs_vector_neut(x: Expr, y: Expr) -> Expr:
    return sub(x, xs_vector_proj(x, y))


@register_dsl_function("xs_normalize")
def xs_normalize(
    x: Expr,
    use_std: bool = False,
    limit: float = 0.0,
    **kwargs,
) -> Expr:
    if "useStd" in kwargs:
        use_std = kwargs.pop("useStd")
    if kwargs:
        raise TypeError(f"unknown xs_normalize arguments: {sorted(kwargs)}")
    result = xs_demean(x)
    if _literal_bool(use_std, "xs_normalize use_std"):
        result = _safe_ratio(result, xs_std(x))
    limit_value = _literal_number(limit, "xs_normalize limit")
    return clamp(result, -limit_value, limit_value) if limit_value > 0.0 else result


@register_dsl_function("xs_one_side")
@register_dsl_function("one_side")
def xs_one_side(x: Expr, side: str = "long") -> Expr:
    name = _literal_text(side, "xs_one_side side").strip().lower()
    if name == "long":
        return sub(x, xs_min(x))
    if name == "short":
        return sub(x, xs_max(x))
    raise ValueError("xs_one_side side must be 'long' or 'short'")


@register_dsl_function("xs_prob_density")
@register_dsl_function("xs_quantile")
def xs_prob_density(
    x: Expr,
    driver: str = "gaussian",
    sigma: float = 1.0,
) -> Expr:
    ranked = xs_pct_rank(x)
    name = _literal_text(driver, "xs_prob_density driver").strip().lower()
    if name in {"gaussian", "normal"}:
        transformed = call("norm_inv", ranked)
    elif name == "uniform":
        transformed = xs_demean(ranked)
    elif name == "cauchy":
        transformed = call("tan", mul(math.pi, sub(ranked, 0.5)))
    else:
        raise ValueError("driver must be gaussian, uniform, or cauchy")
    return mul(transformed, sigma)


xs_quantile = xs_prob_density


@register_dsl_function("xs_scale_down")
def xs_scale_down(x: Expr, constant: float = 0.0) -> Expr:
    low, high = xs_min(x), xs_max(x)
    return sub(_safe_ratio(sub(x, low), sub(high, low)), constant)


@register_dsl_function("xs_winsorize")
def xs_winsorize(x: Expr, std: float = 4.0) -> Expr:
    center = xs_mean(x)
    width = mul(std, xs_std(x))
    return minimum(maximum(x, sub(center, width)), add(center, width))


@register_dsl_function("xs_truncate")
def xs_truncate(x: Expr, max_percent: float = 0.01, **kwargs) -> Expr:
    if "maxPercent" in kwargs:
        max_percent = kwargs.pop("maxPercent")
    if kwargs:
        raise TypeError(f"unknown xs_truncate arguments: {sorted(kwargs)}")
    cap = mul(max_percent, xs_sum(call("abs", x)))
    return minimum(maximum(x, mul(-1.0, cap)), cap)


@register_dsl_function("xs_scale_by_side")
def xs_scale_by_side(x: Expr) -> Expr:
    return xs_scale(x, longscale=1.0, shortscale=1.0)


@register_dsl_function("xs_rank_by_side")
def xs_rank_by_side(x: Expr, rate: float = 2.0, scale: float = 1.0) -> Expr:
    del rate
    positive = where(gt(x, 0.0), x, float("nan"))
    negative = where(lt(x, 0.0), mul(-1.0, x), float("nan"))
    ranked = where(
        gt(x, 0.0),
        xs_pct_rank(positive),
        where(lt(x, 0.0), mul(-1.0, xs_pct_rank(negative)), 0.0),
    )
    scale_value = _literal_number(scale, "xs_rank_by_side scale")
    return xs_scale(ranked, scale=scale_value) if scale_value != 0.0 else ranked


@register_dsl_function("generalized_rank")
def generalized_rank(x: Expr, m: float = 1.0) -> Expr:
    return xs_generalized_rank(x, m)


@register_dsl_function("xs_filter")
def xs_filter(
    x: Expr,
    percentile: float,
    keep_greater: bool = True,
) -> Expr:
    threshold = xs_quantile_value(x, percentile)
    condition = ge(x, threshold) if _literal_bool(
        keep_greater, "xs_filter keep_greater"
    ) else le(x, threshold)
    return where(condition, x, float("nan"))


@register_dsl_function("xs_regression_proj")
@register_dsl_function("regression_proj")
def xs_regression_proj(y: Expr, x: Expr) -> Expr:
    return xs_regression_projection(y, x)


@register_dsl_function("xs_regression_neut")
@register_dsl_function("regression_neut")
def xs_regression_neut(y: Expr, x: Expr) -> Expr:
    return sub(y, xs_regression_proj(y, x))


@register_dsl_function("xs_rank_gmean_amean_diff")
@register_dsl_function("rank_gmean_amean_diff")
def xs_rank_gmean_amean_diff(x: Expr, y: Expr, *rest: Expr) -> Expr:
    ranks = [xs_pct_rank(value) for value in (x, y, *rest)]
    count = float(len(ranks))
    geometric = pow(_product_expr(ranks), 1.0 / count)
    arithmetic = mul(1.0 / count, _sum_expr(ranks))
    return sub(geometric, arithmetic)


def _group_apply(group: Expr, x: Expr, rhs: Expr) -> Expr:
    return call("groupby", group, x, rhs)


@register_dsl_function("group_count")
def group_count(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_count(self_))


@register_dsl_function("group_na_count")
def group_na_count(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_sum(isnan(self_)))


@register_dsl_function("group_mean")
def group_mean(x: Expr, weight: Expr | float, group: Expr) -> Expr:
    if isinstance(weight, Number) and float(weight.value) == 1.0:
        rhs = xs_mean(self_)
    elif isinstance(weight, (int, float)) and float(weight) == 1.0:
        rhs = xs_mean(self_)
    else:
        rhs = xs_weighted_mean(self_, weight)
    return _group_apply(group, x, rhs)


@register_dsl_function("group_extra")
def group_extra(x: Expr, weight: Expr | float, group: Expr) -> Expr:
    replacement = group_mean(x, weight, group)
    return where(isnan(x), replacement, x)


@register_dsl_function("group_max")
def group_max(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_max(self_))


@register_dsl_function("group_median")
def group_median(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, call("xs_median", self_))


@register_dsl_function("group_min")
def group_min(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_min(self_))


@register_dsl_function("group_rank")
def group_rank(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_pct_rank(self_))


@register_dsl_function("group_scale")
def group_scale(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_scale_down(self_))


@register_dsl_function("group_std_dev")
def group_std_dev(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_std(self_))


@register_dsl_function("group_sum")
def group_sum(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_sum(self_))


@register_dsl_function("group_zscore")
def group_zscore(x: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_zscore(self_))


@register_dsl_function("group_percentage")
def group_percentage(
    x: Expr,
    group: Expr,
    percentage: float = 0.5,
) -> Expr:
    return _group_apply(
        group, x, xs_quantile_value(self_, percentage)
    )


@register_dsl_function("group_vector_proj")
def group_vector_proj(x: Expr, y: Expr, group: Expr) -> Expr:
    return _group_apply(group, x, xs_vector_projection(self_, y))


@register_dsl_function("group_vector_neut")
def group_vector_neut(x: Expr, y: Expr, group: Expr) -> Expr:
    return sub(x, group_vector_proj(x, y, group))


@register_dsl_function("group_neutralize")
@register_dsl_function("xs_group_neutralize")
def group_neutralize(x: Expr, group: Expr) -> Expr:
    return sub(x, group_mean(x, 1.0, group))


xs_group_neutralize = group_neutralize


@register_dsl_function("xs_market_neutralize")
def xs_market_neutralize(x: Expr, market: Expr) -> Expr:
    return group_neutralize(x, market)


@register_dsl_function("group_normalize")
def group_normalize(
    x: Expr,
    group: Expr,
    constant_check: bool = False,
    tolerance: float = 0.01,
    scale: float = 1.0,
    **kwargs,
) -> Expr:
    if "constantCheck" in kwargs:
        constant_check = kwargs.pop("constantCheck")
    if kwargs:
        raise TypeError(f"unknown group_normalize arguments: {sorted(kwargs)}")
    result = _group_apply(group, x, xs_scale(self_, scale))
    if _literal_bool(constant_check, "group_normalize constant_check"):
        dispersion = group_std_dev(x, group)
        result = where(le(dispersion, tolerance), float("nan"), result)
    return result


@register_dsl_function("group_backfill")
def group_backfill(
    x: Expr,
    group: Expr,
    periods: int,
    std: float = 4.0,
) -> Expr:
    """Fill missing values from winsorized same-group trailing observations."""

    trailing = rolling_mean(self_, periods, min_periods=1)
    replacement = _group_apply(
        group,
        x,
        xs_mean(xs_winsorize(trailing, std)),
    )
    return where(isnan(x), replacement, x)


@register_dsl_function("periods_from_last_change")
def periods_from_last_change(x: Expr) -> Expr:
    return call("periods_since_last_change", x)


@register_dsl_function("ts_hump_decay")
def ts_hump_decay(
    x: Expr,
    p: float = 0.1,
    relative: bool = False,
) -> Expr:
    return call("hump_decay", x, p=p, relative=relative)


@register_dsl_function("jump_decay")
def jump_decay(
    x: Expr,
    periods: int,
    stddev: bool = True,
    sensitivity: float = 0.5,
    force: float = 0.1,
) -> Expr:
    prior = call("shift", x, 1, 1)
    delta = sub(x, prior)
    threshold = sensitivity
    if _literal_bool(stddev, "jump_decay stddev"):
        threshold = mul(
            sensitivity,
            rolling_std(x, periods, min_periods=2),
        )
    return where(
        gt(call("abs", delta), threshold),
        add(prior, mul(force, delta)),
        x,
    )


@register_dsl_function("keep")
def keep(x: Expr, f: Expr, periods: int = 5) -> Expr:
    age = call("periods_since_last_change", f)
    return where(lt(age, periods), x, float("nan"))


@register_dsl_function("inst_tvr")
@register_dsl_function("ts_inst_tvr")
def ts_inst_tvr(x: Expr, periods: int) -> Expr:
    prior = call("shift", x, 1, 1)
    traded = rolling_sum(
        call("abs", sub(x, prior)), periods, min_periods=1
    )
    held = rolling_sum(call("abs", x), periods, min_periods=1)
    return _safe_ratio(traded, held)


@register_dsl_function("kth_element")
@register_dsl_function("ts_backfill")
def ts_backfill(
    x: Expr,
    periods: int,
    k: int = 1,
    ignore: str = "NAN 0",
) -> Expr:
    return call(
        "rolling_kth",
        x,
        periods,
        k=k,
        ignore=ignore,
        min_periods=k,
    )


@register_dsl_function("last_diff_value")
@register_dsl_function("prev_diff_value")
def prev_diff_value(x: Expr, periods: int) -> Expr:
    return call("rolling_prev_diff", x, periods)


@register_dsl_function("ts_weighted_delay")
def ts_weighted_delay(x: Expr, k: float = 0.5) -> Expr:
    return add(mul(k, x), mul(sub(1.0, k), call("shift", x, 1, 1)))


@register_dsl_function("ts_delay")
@register_dsl_function("ts_shift")
def ts_shift(x: Expr, periods: int) -> Expr:
    return call("shift", x, periods, periods)


@register_dsl_function("ts_delta")
@register_dsl_function("ts_diff")
def ts_diff(x: Expr, periods: int) -> Expr:
    return sub(x, ts_shift(x, periods))


@register_dsl_function("ts_returns")
def ts_returns(x: Expr, periods: int, mode: int = 1) -> Expr:
    lagged = ts_shift(x, periods)
    change = sub(x, lagged)
    mode_value = _literal_int(mode, "ts_returns mode")
    if mode_value == 1:
        denominator = lagged
    elif mode_value == 2:
        denominator = mul(0.5, add(x, lagged))
    else:
        raise ValueError("ts_returns mode must be 1 or 2")
    return div(change, denominator)


@register_dsl_function("ts_pct_change")
def ts_pct_change(x: Expr, periods: int) -> Expr:
    return ts_returns(x, periods, mode=1)


@register_dsl_function("ts_ln_change")
def ts_ln_change(x: Expr, periods: int) -> Expr:
    return call("ln", add(1.0, ts_pct_change(x, periods)))


@register_dsl_function("ts_sum")
def ts_sum(x: Expr, periods: int) -> Expr:
    return rolling_sum(x, periods)


@register_dsl_function("ts_product")
def ts_product(x: Expr, periods: int) -> Expr:
    return call("rolling_product", x, periods)


@register_dsl_function("ts_mean")
def ts_mean(x: Expr, periods: int) -> Expr:
    return rolling_mean(x, periods)


@register_dsl_function("ts_median")
def ts_median(x: Expr, periods: int) -> Expr:
    return call("rolling_median", x, periods)


@register_dsl_function("ts_min")
def ts_min(x: Expr, periods: int) -> Expr:
    return rolling_min(x, periods)


@register_dsl_function("ts_max")
def ts_max(x: Expr, periods: int) -> Expr:
    return rolling_max(x, periods)


@register_dsl_function("ts_std_dev")
@register_dsl_function("ts_std")
def ts_std(x: Expr, periods: int) -> Expr:
    return rolling_std(x, periods)


@register_dsl_function("ts_ir")
def ts_ir(x: Expr, periods: int) -> Expr:
    return _safe_ratio(ts_mean(x, periods), ts_std(x, periods))


@register_dsl_function("ts_rank")
def ts_rank(x: Expr, periods: int) -> Expr:
    return call("rolling_pct_rank", x, periods)


@register_dsl_function("ts_quantile")
@register_dsl_function("ts_prob_density")
def ts_prob_density(
    x: Expr,
    periods: int,
    driver: str = "gaussian",
    sigma: float = 1.0,
) -> Expr:
    ranked = ts_rank(x, periods)
    name = _literal_text(driver, "ts_prob_density driver").strip().lower()
    if name in {"gaussian", "normal"}:
        transformed = call("norm_inv", ranked)
    elif name == "uniform":
        transformed = sub(ranked, 0.5)
    elif name == "cauchy":
        transformed = call("tan", mul(math.pi, sub(ranked, 0.5)))
    else:
        raise ValueError("driver must be gaussian, uniform, or cauchy")
    return mul(transformed, sigma)


@register_dsl_function("ts_percentage")
def ts_percentage(
    x: Expr,
    periods: int,
    percentage: float = 0.5,
) -> Expr:
    return call("rolling_quantile", x, periods, q=percentage)


def _ridge_halflife_from_span(span, name: str = "span") -> float:
    span_value = _numeric_literal(span, name)
    if not math.isfinite(span_value) or span_value < 1.0:
        raise ValueError(f"{name} must be finite and >= 1")
    if span_value == 1.0:
        return 1e-12
    return math.log(0.5) / math.log1p(-2.0 / (span_value + 1.0))


@register_dsl_function("ts_poly_regression")
def ts_poly_regression(
    y: Expr,
    x: Expr,
    periods: int,
    k: int = 1,
    weights: Expr | float = 1.0,
    lambda_: float = 0.0,
) -> Expr:
    degree = _literal_int(k, "ts_poly_regression k")
    if degree < 1:
        raise ValueError("ts_poly_regression k must be >= 1")
    features = [1.0, *(pow(x, value) for value in range(1, degree + 1))]
    model = Ridge(
        *features,
        y=y,
        weights=weights,
        hl=_ridge_halflife_from_span(periods, "ts_poly_regression periods"),
        lambda_=lambda_,
    )
    return get_residuals(model)


@register_dsl_function("ts_decay_linear")
def ts_decay_linear(x: Expr, periods: int) -> Expr:
    return call("rolling_decay_linear", x, periods)


@register_dsl_function("ts_arg_max")
@register_dsl_function("ts_argmax")
def ts_argmax(x: Expr, periods: int) -> Expr:
    return call("rolling_argmax", x, periods)


@register_dsl_function("ts_arg_min")
@register_dsl_function("ts_argmin")
def ts_argmin(x: Expr, periods: int) -> Expr:
    return call("rolling_argmin", x, periods)


@register_dsl_function("ts_av_diff")
@register_dsl_function("ts_mean_diff")
def ts_mean_diff(x: Expr, periods: int) -> Expr:
    return sub(x, ts_mean(x, periods))


@register_dsl_function("ts_max_diff")
def ts_max_diff(x: Expr, periods: int) -> Expr:
    return sub(x, ts_max(x, periods))


@register_dsl_function("ts_min_diff")
def ts_min_diff(x: Expr, periods: int) -> Expr:
    return sub(x, ts_min(x, periods))


@register_dsl_function("ts_min_max_cps")
def ts_min_max_cps(x: Expr, periods: int, f: float = 2.0) -> Expr:
    return sub(add(ts_min(x, periods), ts_max(x, periods)), mul(f, x))


@register_dsl_function("ts_min_max_diff")
def ts_min_max_diff(x: Expr, periods: int, f: float = 0.5) -> Expr:
    return sub(x, mul(f, add(ts_min(x, periods), ts_max(x, periods))))


@register_dsl_function("ts_scale")
def ts_scale(x: Expr, periods: int, constant: float = 0.0) -> Expr:
    low = ts_min(x, periods)
    high = ts_max(x, periods)
    return add(_safe_ratio(sub(x, low), sub(high, low)), constant)


@register_dsl_function("ts_zscore")
def ts_zscore(x: Expr, periods: int) -> Expr:
    return _safe_ratio(sub(x, ts_mean(x, periods)), ts_std(x, periods))


@register_dsl_function("ts_count_nans")
@register_dsl_function("ts_count_nan")
def ts_count_nans(x: Expr, periods: int) -> Expr:
    return rolling_sum(isnan(x), periods)


@register_dsl_function("ts_count_nonnumeric")
def ts_count_nonnumeric(x: Expr, periods: int) -> Expr:
    return rolling_sum(logical_not(isfinite(x)), periods)


@register_dsl_function("ts_entropy")
def ts_entropy(x: Expr, periods: int, buckets: int = 10) -> Expr:
    return call("rolling_entropy", x, periods, buckets=buckets)


@register_dsl_function("ewm_vector_proj")
@register_dsl_function("ts_vector_proj")
def ewm_vector_proj(
    x: Expr,
    y: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    inputs = (x, y)
    numerator = _ewm_raw_moment(
        inputs, (1, 1), span, min_periods, ignore_na, adjust
    )
    denominator = _ewm_raw_moment(
        inputs, (0, 2), span, min_periods, ignore_na, adjust
    )
    return mul(_safe_ratio(numerator, denominator), y)


@register_dsl_function("ewm_vector_neut")
@register_dsl_function("ts_vector_neut")
def ewm_vector_neut(
    x: Expr,
    y: Expr,
    span: float,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
) -> Expr:
    return sub(
        x,
        ewm_vector_proj(
            x, y, span, min_periods, ignore_na, adjust
        ),
    )


ts_vector_proj = ewm_vector_proj
ts_vector_neut = ewm_vector_neut


@register_dsl_function("ts_rank_gmean_amean_diff")
def ts_rank_gmean_amean_diff(
    x: Expr,
    y: Expr,
    *args,
    periods=None,
) -> Expr:
    if periods is None:
        if not args:
            raise TypeError("ts_rank_gmean_amean_diff requires periods")
        *rest, periods = args
    else:
        rest = list(args)
    ranks = [ts_rank(value, periods) for value in (x, y, *rest)]
    count = float(len(ranks))
    geometric = pow(_product_expr(ranks), 1.0 / count)
    arithmetic = mul(1.0 / count, _sum_expr(ranks))
    return sub(geometric, arithmetic)


@register_dsl_function("ts_geomean")
def ts_geomean(
    x: Expr,
    periods: int,
    replace_non_numeric=None,
) -> Expr:
    values = x
    if replace_non_numeric is not None:
        values = where(isfinite(x), x, replace_non_numeric)
    return call("exp", rolling_mean(call("ln", values), periods))


@register_dsl_function("slope")
def slope(x: Expr, periods: int) -> Expr:
    period_count = _literal_int(periods, "slope periods")
    if period_count < 1:
        raise ValueError("slope periods must be >= 1")
    return _sum_expr(
        mul(1.0 / float(lag), ts_diff(x, lag))
        for lag in range(1, period_count + 1)
    )


@register_dsl_function("ts_theilsen")
def ts_theilsen(y: Expr, x: Expr, periods: int) -> Expr:
    return call("rolling_theilsen", y, x, periods)


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

    ``periods`` is the EWM span measured in input rows. The constant feature is
    coefficient 0 and ``x`` is coefficient 1.
    """

    period_value = _numeric_literal(periods, "periods")
    if not math.isfinite(period_value) or period_value < 1.0:
        raise ValueError("ts_regression periods must be finite and >= 1")
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
        hl=_ridge_halflife_from_span(periods, "ts_regression periods"),
        lambda_=lambda_,
    )
    name = _rettype_name(rettype).strip().lower()
    projections = {
        "residual": get_residuals,
        "residuals": get_residuals,
        "prediction": get_preds,
        "predictions": get_preds,
        "alpha": lambda value: get_coefficient(value, 0),
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
    "arc_cos",
    "arc_sin",
    "arc_tan",
    "bucket",
    "clamp",
    "convert_float",
    "elementwise_max",
    "elementwise_min",
    "equal",
    "ewm_co_kurtosis",
    "ewm_co_skewness",
    "ewm_corr",
    "ewm_cov",
    "ewm_kurtosis",
    "ewm_moment",
    "ewm_partial_corr",
    "ewm_skewness",
    "ewm_std",
    "ewm_triple_corr",
    "ewm_var",
    "ewm_vector_neut",
    "ewm_vector_proj",
    "generalized_rank",
    "get_df",
    "group_backfill",
    "group_count",
    "group_extra",
    "group_max",
    "group_mean",
    "group_median",
    "group_min",
    "group_na_count",
    "group_neutralize",
    "group_normalize",
    "group_percentage",
    "group_rank",
    "group_scale",
    "group_std_dev",
    "group_sum",
    "group_vector_neut",
    "group_vector_proj",
    "group_zscore",
    "if_else",
    "inverse",
    "is_finite",
    "is_nan",
    "is_not_finite",
    "is_not_nan",
    "jump_decay",
    "keep",
    "left_right_tail",
    "left_tail",
    "less",
    "log",
    "log_diff",
    "logical_and",
    "logical_or",
    "nan_mask",
    "nan_out",
    "negate",
    "pasteurize",
    "periods_from_last_change",
    "prev_diff_value",
    "replace",
    "reverse",
    "right_tail",
    "rolling_range",
    "rolling_scale",
    "rolling_zscore",
    "round_df",
    "round_down",
    "s_log_1p",
    "sigmoid",
    "signed_power",
    "slope",
    "tail",
    "to_nan",
    "ts_argmax",
    "ts_argmin",
    "ts_backfill",
    "ts_count_nans",
    "ts_count_nonnumeric",
    "ts_decay_linear",
    "ts_diff",
    "ts_entropy",
    "ts_geomean",
    "ts_hump_decay",
    "ts_inst_tvr",
    "ts_ir",
    "ts_ln_change",
    "ts_max",
    "ts_max_diff",
    "ts_mean",
    "ts_mean_diff",
    "ts_median",
    "ts_min",
    "ts_min_diff",
    "ts_min_max_cps",
    "ts_min_max_diff",
    "ts_pct_change",
    "ts_percentage",
    "ts_poly_regression",
    "ts_prob_density",
    "ts_product",
    "ts_rank",
    "ts_rank_gmean_amean_diff",
    "xs_demean",
    "ts_regression",
    "ts_returns",
    "ts_scale",
    "ts_shift",
    "ts_std",
    "ts_sum",
    "ts_theilsen",
    "ts_vector_neut",
    "ts_vector_proj",
    "ts_weighted_delay",
    "ts_zscore",
    "vec_avg",
    "vec_choose",
    "vec_count",
    "vec_ir",
    "vec_kurtosis",
    "vec_max",
    "vec_min",
    "vec_norm",
    "vec_percentage",
    "vec_powersum",
    "vec_range",
    "vec_skewness",
    "vec_stddev",
    "vec_sum",
    "xs_direction",
    "xs_filter",
    "xs_generalized_rank",
    "xs_group_neutralize",
    "xs_market_neutralize",
    "xs_normalize",
    "xs_one_side",
    "xs_prob_density",
    "xs_quantile",
    "xs_rank_by_side",
    "xs_rank_gmean_amean_diff",
    "xs_regression_neut",
    "xs_regression_proj",
    "xs_scale",
    "xs_scale_by_side",
    "xs_scale_down",
    "xs_truncate",
    "xs_vector_proj",
    "xs_vector_neut",
    "xs_winsorize",
    "xs_zscore",
]
