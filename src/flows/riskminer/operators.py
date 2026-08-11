from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import math

from trading_dsl_engine.base.dsl import (
    abs as dsl_abs,
    add,
    arctan,
    div,
    ewm,
    fillna,
    fraction,
    gt,
    isfinite,
    le,
    ln,
    lt,
    maximum,
    minimum,
    mul,
    pow as dsl_pow,
    purify,
    rolling_decay_linear,
    rolling_max,
    rolling_mean,
    rolling_median,
    rolling_min,
    rolling_pct_rank,
    rolling_std,
    rolling_sum,
    shift,
    sign,
    sub,
    where,
    xs_pct_rank,
    xs_rank,
)
from trading_dsl_engine.base.parser import Expr

from .semantics import (
    DEFAULT_TYPE_GRAPH,
    SearchShape,
    SemanticInfo,
    boolean_output,
    broadcast_shape,
    common_output,
    compatible,
    dimensionless_output,
    division_output,
    multiplication_output,
    subtraction_output,
    unary_preserve,
)

Validator = Callable[[Sequence[SemanticInfo]], bool]
Infer = Callable[[Sequence[SemanticInfo]], SemanticInfo]
Builder = Callable[[Sequence[Expr], Sequence[float | None]], Expr]
DEFAULT_DYNAMIC_PERIODS = (5, 60, 1440)


@dataclass(frozen=True)
class OperatorSchema:
    name: str
    arity: int
    validate: Validator
    infer: Infer
    build: Builder
    prior: float = 1.0
    commutative: bool = False
    family: str = "generic"


def _numeric(info: SemanticInfo) -> bool:
    return "numeric" in info.types


def _all_numeric(args: Sequence[SemanticInfo]) -> bool:
    return all(_numeric(arg) for arg in args)


def _row_numeric(info: SemanticInfo) -> bool:
    return _numeric(info) and info.shape in {SearchShape.ROW, SearchShape.BOOLEAN_ROW}


def _unary_numeric(args: Sequence[SemanticInfo]) -> bool:
    return len(args) == 1 and _numeric(args[0]) and args[0].shape in {
        SearchShape.ROW, SearchShape.SCALAR,
    }


def _positive_unary(args: Sequence[SemanticInfo]) -> bool:
    return _unary_numeric(args) and args[0].lower > 0.0


def _cross_sectional_row(args: Sequence[SemanticInfo]) -> bool:
    return len(args) == 1 and _row_numeric(args[0])


def _same_type(args: Sequence[SemanticInfo]) -> bool:
    return len(args) == 2 and _all_numeric(args) and compatible(args[0], args[1])


def _add_same_type(args: Sequence[SemanticInfo]) -> bool:
    if not _same_type(args):
        return False
    left = DEFAULT_TYPE_GRAPH.closure(args[0].types)
    right = DEFAULT_TYPE_GRAPH.closure(args[1].types)
    return not ("timestamp" in left and "timestamp" in right)


def _broadcast_numeric(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 2
        and _all_numeric(args)
        and broadcast_shape(args[0].shape, args[1].shape) is not None
    )


def _where(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 3
        and args[0].boolean
        and compatible(args[1], args[2])
        and broadcast_shape(args[0].shape, args[1].shape) is not None
    )


def _fillna(args: Sequence[SemanticInfo]) -> bool:
    return len(args) == 2 and compatible(args[0], args[1])


def _positive_static_parameter(
    info: SemanticInfo,
    *,
    integer: bool = False,
    maximum_value: float = math.inf,
) -> bool:
    return (
        info.shape is SearchShape.SCALAR
        and info.static
        and info.lower > 0.0
        and info.upper <= maximum_value
        and (not integer or info.integer)
    )


def _temporal(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 2 and _row_numeric(args[0])
        and _positive_static_parameter(args[1])
    )


def _history(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 2 and _row_numeric(args[0])
        and _positive_static_parameter(args[1], integer=True, maximum_value=4096)
    )


def _rolling(args: Sequence[SemanticInfo]) -> bool:
    return _history(args)


def _rolling_pair(args: Sequence[SemanticInfo]) -> bool:
    return (
        len(args) == 3
        and _row_numeric(args[0])
        and _row_numeric(args[1])
        and _positive_static_parameter(args[2], integer=True, maximum_value=4096)
    )


def _dynamic_unary(args: Sequence[SemanticInfo]) -> bool:
    return len(args) == 2 and _row_numeric(args[0]) and _row_numeric(args[1])


def _dynamic_pair(args: Sequence[SemanticInfo]) -> bool:
    return len(args) == 3 and all(_row_numeric(arg) for arg in args)


def _same_infer(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return common_output(args[0], args[1])


def _sub_infer(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return subtraction_output(args[0], args[1])


def _preserve_first(args: Sequence[SemanticInfo]) -> SemanticInfo:
    value = unary_preserve(args[0])
    return SemanticInfo(value.types, SearchShape.ROW, role="value")


def _dimensionless(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return dimensionless_output(args[0])


def _bounded_rank(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return dimensionless_output(args[0], lower=0.0, upper=1.0)


def _std_infer(args: Sequence[SemanticInfo]) -> SemanticInfo:
    value = _preserve_first(args)
    return SemanticInfo(
        value.types, value.shape, lower=0.0, role="value"
    )


def _variance_infer(args: Sequence[SemanticInfo]) -> SemanticInfo:
    value = multiplication_output(args[0], args[0])
    return SemanticInfo(
        value.types, value.shape, lower=0.0, role="value"
    )


def _covariance_infer(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return multiplication_output(args[0], args[1])


def _bool_infer(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return boolean_output(args[0], args[1])


def _where_infer(args: Sequence[SemanticInfo]) -> SemanticInfo:
    return common_output(args[1], args[2])


def _literal_at(values: Sequence[float | None], index: int, name: str) -> float:
    value = values[index]
    if value is None:
        raise ValueError(f"{name} requires a compile-time literal")
    return float(value)


def _rolling_raw_moments(x: Expr, periods: int) -> tuple[Expr, Expr, Expr, Expr]:
    m1 = rolling_mean(x, periods)
    m2 = rolling_mean(mul(x, x), periods)
    m3 = rolling_mean(dsl_pow(x, 3.0), periods)
    m4 = rolling_mean(dsl_pow(x, 4.0), periods)
    return m1, m2, m3, m4


def _rolling_var_expr(x: Expr, periods: int) -> Expr:
    m1 = rolling_mean(x, periods)
    m2 = rolling_mean(mul(x, x), periods)
    return maximum(sub(m2, mul(m1, m1)), 0.0)


def _rolling_skew_expr(x: Expr, periods: int) -> Expr:
    m1, m2, m3, _ = _rolling_raw_moments(x, periods)
    var = maximum(sub(m2, mul(m1, m1)), 0.0)
    central3 = add(sub(m3, mul(3.0, mul(m1, m2))), mul(2.0, dsl_pow(m1, 3.0)))
    return div(central3, dsl_pow(var, 1.5))


def _rolling_kurt_expr(x: Expr, periods: int) -> Expr:
    m1, m2, m3, m4 = _rolling_raw_moments(x, periods)
    var = maximum(sub(m2, mul(m1, m1)), 0.0)
    central4 = add(
        sub(add(m4, mul(6.0, mul(mul(m1, m1), m2))), mul(4.0, mul(m1, m3))),
        mul(-3.0, dsl_pow(m1, 4.0)),
    )
    return sub(div(central4, mul(var, var)), 3.0)


def _pairwise_observations(x: Expr, y: Expr) -> tuple[Expr, Expr]:
    # Covariance/correlation must use the same observations for x, y and xy.
    # Masking them together preserves pairwise-missing semantics when one input
    # is absent independently of the other.
    valid = isfinite(x) & isfinite(y)
    nan = float("nan")
    return where(valid, x, nan), where(valid, y, nan)


def _rolling_cov_pair_expr(
    x_pair: Expr, y_pair: Expr, periods: int
) -> Expr:
    return sub(
        rolling_mean(mul(x_pair, y_pair), periods),
        mul(
            rolling_mean(x_pair, periods),
            rolling_mean(y_pair, periods),
        ),
    )


def _rolling_cov_expr(x: Expr, y: Expr, periods: int) -> Expr:
    x_pair, y_pair = _pairwise_observations(x, y)
    return _rolling_cov_pair_expr(x_pair, y_pair, periods)


def _rolling_corr_expr(x: Expr, y: Expr, periods: int) -> Expr:
    x_pair, y_pair = _pairwise_observations(x, y)
    return div(
        _rolling_cov_pair_expr(x_pair, y_pair, periods),
        mul(
            rolling_std(x_pair, periods, ddof=0),
            rolling_std(y_pair, periods, ddof=0),
        ),
    )


def _dynamic_bank(
    selector: Expr,
    expressions: Sequence[Expr],
) -> Expr:
    if not expressions:
        raise ValueError("dynamic operator requires at least one static branch")
    if len(expressions) == 1:
        return expressions[0]
    rank = xs_pct_rank(selector)
    result = expressions[-1]
    count = len(expressions)
    # Nested conditions are ordered from the largest threshold down so the
    # smallest matching quantile wins.
    for index in range(count - 2, -1, -1):
        threshold = float(index + 1) / float(count)
        result = where(le(rank, threshold), expressions[index], result)
    return result


def _dynamic_unary_builder(
    fn: Callable[[Expr, int], Expr],
    periods: Sequence[int],
) -> Builder:
    values = tuple(int(period) for period in periods)
    return lambda exprs, literals: _dynamic_bank(
        exprs[1], tuple(fn(exprs[0], period) for period in values)
    )


def _dynamic_pair_builder(
    fn: Callable[[Expr, Expr, int], Expr],
    periods: Sequence[int],
) -> Builder:
    values = tuple(int(period) for period in periods)
    return lambda exprs, literals: _dynamic_bank(
        exprs[2], tuple(fn(exprs[0], exprs[1], period) for period in values)
    )


def default_operator_catalog(
    *,
    dynamic_periods: Sequence[int] = DEFAULT_DYNAMIC_PERIODS,
) -> tuple[OperatorSchema, ...]:
    """Typed catalog covering the paper inventory and native-safe extensions."""

    periods = tuple(sorted({int(value) for value in dynamic_periods if int(value) > 0}))
    if not periods:
        raise ValueError("dynamic_periods must contain a positive integer")
    out: list[OperatorSchema] = []

    for name, fn, prior in (
        ("abs", dsl_abs, 0.8), ("purify", purify, 0.7),
    ):
        out.append(OperatorSchema(
            name, 1, _unary_numeric, _preserve_first,
            lambda exprs, literals, fn=fn: fn(exprs[0]),
            prior=prior, family="unary_preserving",
        ))
    for name, fn, validator, prior in (
        ("sign", sign, _unary_numeric, 0.8),
        ("fraction", fraction, _unary_numeric, 0.35),
        ("arctan", arctan, _unary_numeric, 0.55),
        ("log", ln, _positive_unary, 0.7),
    ):
        out.append(OperatorSchema(
            name, 1, validator, _dimensionless,
            lambda exprs, literals, fn=fn: fn(exprs[0]),
            prior=prior, family="normalization",
        ))
    for name, fn, prior in (
        ("xs_rank", xs_rank, 1.8),
        ("xs_pct_rank", xs_pct_rank, 1.6),
    ):
        out.append(OperatorSchema(
            name, 1, _cross_sectional_row, _bounded_rank,
            lambda exprs, literals, fn=fn: fn(exprs[0]),
            prior=prior, family="cross_sectional",
        ))

    out.extend((
        OperatorSchema("add", 2, _add_same_type, _same_infer,
                       lambda e, l: add(e[0], e[1]), 1.25, True, "compatible_binary"),
        OperatorSchema("sub", 2, _same_type, _sub_infer,
                       lambda e, l: sub(e[0], e[1]), 1.35, False, "compatible_binary"),
        OperatorSchema("minimum", 2, _same_type, _same_infer,
                       lambda e, l: minimum(e[0], e[1]), 0.5, True, "compatible_binary"),
        OperatorSchema("maximum", 2, _same_type, _same_infer,
                       lambda e, l: maximum(e[0], e[1]), 0.5, True, "compatible_binary"),
        OperatorSchema("mul", 2, _broadcast_numeric,
                       lambda a: multiplication_output(a[0], a[1]),
                       lambda e, l: mul(e[0], e[1]), 1.05, True, "numeric_binary"),
        OperatorSchema("div", 2, _broadcast_numeric,
                       lambda a: division_output(a[0], a[1]),
                       lambda e, l: div(e[0], e[1]), 1.3, False, "numeric_binary"),
        OperatorSchema("greater", 2, _same_type, _bool_infer,
                       lambda e, l: gt(e[0], e[1]), 0.65, False, "comparison"),
        OperatorSchema("less", 2, _same_type, _bool_infer,
                       lambda e, l: lt(e[0], e[1]), 0.65, False, "comparison"),
        OperatorSchema("fillna", 2, _fillna, _same_infer,
                       lambda e, l: fillna(e[0], e[1]), 0.35, False, "compatible_binary"),
        OperatorSchema("where", 3, _where, _where_infer,
                       lambda e, l: where(e[0], e[1], e[2]), 0.35, False, "conditional"),
    ))

    static_specs: tuple[tuple[str, Callable, Infer, float], ...] = (
        ("ewm", lambda x, p: ewm(x, span=p), _preserve_first, 1.45),
        ("shift", lambda x, p: shift(x, p, p), _preserve_first, 1.05),
        ("rolling_rank", lambda x, p: rolling_pct_rank(x, p), _bounded_rank, 0.9),
        ("rolling_skew", _rolling_skew_expr, _dimensionless, 0.55),
        ("rolling_kurt", _rolling_kurt_expr, _dimensionless, 0.45),
        ("rolling_mean", lambda x, p: rolling_mean(x, p), _preserve_first, 0.9),
        ("rolling_median", lambda x, p: rolling_median(x, p), _preserve_first, 0.55),
        ("rolling_sum", lambda x, p: rolling_sum(x, p), _preserve_first, 0.65),
        ("rolling_std", lambda x, p: rolling_std(x, p, ddof=0), _std_infer, 0.8),
        ("rolling_var", _rolling_var_expr, _variance_infer, 0.7),
        ("rolling_max", lambda x, p: rolling_max(x, p), _preserve_first, 0.55),
        ("rolling_min", lambda x, p: rolling_min(x, p), _preserve_first, 0.55),
        ("rolling_wma", lambda x, p: rolling_decay_linear(x, p), _preserve_first, 0.65),
    )
    for name, fn, infer, prior in static_specs:
        validator = _temporal if name == "ewm" else (_history if name == "shift" else _rolling)
        out.append(OperatorSchema(
            name, 2, validator, infer,
            lambda e, l, fn=fn, name=name: fn(e[0], int(_literal_at(l, 1, name))),
            prior=prior, family="temporal",
        ))

    for name, fn, infer, prior in (
        ("rolling_cov", _rolling_cov_expr, _covariance_infer, 0.65),
        ("rolling_corr", _rolling_corr_expr, _dimensionless, 0.75),
    ):
        out.append(OperatorSchema(
            name, 3, _rolling_pair, infer,
            lambda e, l, fn=fn, name=name: fn(
                e[0], e[1], int(_literal_at(l, 2, name))
            ),
            prior=prior, family="rolling_pair",
        ))

    dynamic_unary = (
        ("dynamic_ewm", lambda x, p: ewm(x, span=p), _preserve_first, 0.8),
        ("dynamic_shift", lambda x, p: shift(x, p, p), _preserve_first, 0.6),
        ("dynamic_rolling_rank", lambda x, p: rolling_pct_rank(x, p), _bounded_rank, 0.55),
        ("dynamic_rolling_mean", lambda x, p: rolling_mean(x, p), _preserve_first, 0.55),
        ("dynamic_rolling_std", lambda x, p: rolling_std(x, p, ddof=0), _std_infer, 0.5),
    )
    for name, fn, infer, prior in dynamic_unary:
        out.append(OperatorSchema(
            name, 2, _dynamic_unary, infer,
            _dynamic_unary_builder(fn, periods),
            prior=prior, family="dynamic_temporal",
        ))
    for name, fn, infer, prior in (
        ("dynamic_rolling_cov", _rolling_cov_expr, _covariance_infer, 0.45),
        ("dynamic_rolling_corr", _rolling_corr_expr, _dimensionless, 0.5),
    ):
        out.append(OperatorSchema(
            name, 3, _dynamic_pair, infer,
            _dynamic_pair_builder(fn, periods),
            prior=prior, family="dynamic_temporal_pair",
        ))
    return tuple(out)


def catalog_by_name(
    schemas: Sequence[OperatorSchema] | None = None,
) -> dict[str, OperatorSchema]:
    values = tuple(default_operator_catalog() if schemas is None else schemas)
    out: dict[str, OperatorSchema] = {}
    for schema in values:
        if schema.name in out:
            raise ValueError(f"duplicate operator schema {schema.name!r}")
        out[schema.name] = schema
    return out


__all__ = [
    "DEFAULT_DYNAMIC_PERIODS", "OperatorSchema", "catalog_by_name",
    "default_operator_catalog",
]
