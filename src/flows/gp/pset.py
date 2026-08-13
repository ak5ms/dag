from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import re
from typing import Mapping, Sequence

from deap import gp

from flows.gp.regression import (
    REGRESSION_PROJECTIONS,
    rowwise_ridge_projection,
    temporal_poly_regression_residual,
    temporal_ridge_projection,
    xs_regression_neutralize,
)
from flows.gp.types import (
    AxisSpec,
    BoolParam,
    BoolRow,
    CountRow,
    DatetimeUnit,
    DerivedNumericRow,
    DimensionlessRow,
    DurationRow,
    ExprValue,
    FilterHSpec,
    FilterTSpec,
    FrequencySpec,
    KthIgnoreSpec,
    NumericRow,
    PeriodAtLeastTwo,
    PositiveFloat,
    PositiveInt,
    PositiveNumber,
    PriceRow,
    QuantileParam,
    QuantityRow,
    TimestampRow,
    TradingDayHorizonRow,
    VALUE_TYPES,
    unwrap,
)
from flows.gp.utils_primitives import (
    NON_ROW_CPP_STREAM_UTIL_NAMES,
    ROW_SHAPED_CPP_STREAM_UTIL_NAMES,
    register_cpp_stream_utils,
)
from flows.gp.wrappers import broadcast_reduction, broadcast_xs_unary
from flows.riskminer.semantics import (
    DEFAULT_TYPE_GRAPH,
    SemanticInfo,
    inputdata_alpha_terminal_metadata,
)
from trading_dsl_engine.base import dsl


_BUILTIN_DSL_OPERATOR_NAMES = frozenset(dsl._DSL_OP_SIGNATURES)
_BASE_DSL_REGISTERED_NAMES = frozenset({
    "to_dt",
    "timeofday",
    "hour",
    "minute",
    "second",
    "year",
    "month",
    "day",
    "dayofweek",
    "dayofyear",
    "shift",
    "floor",
    "ceil",
    "round",
    "ratio",
    "diff",
})
_DIRECT_DSL_OPERATOR_NAMES = frozenset({"le", "ge", "Ridge"})
ALL_DSL_OPERATOR_NAMES = (
    _BUILTIN_DSL_OPERATOR_NAMES
    | _BASE_DSL_REGISTERED_NAMES
    | _DIRECT_DSL_OPERATOR_NAMES
)

EXCLUDED_DSL_OPERATOR_NAMES = frozenset({
    "emit", "einsum", "groupby", "cat", "cache", "buffer", "outer",
    "bspline", "rbf_basis", "future_rbf_basis_sum", "col", "Ridge",
    "InstrumentBasisMean", "get_beta", "get_preds", "get_residuals", "get_sse",
    "get_sst", "get_r2", "get_residual_variance", "get_standard_errors",
    "get_standard_error", "get_tstats", "get_tstat", "get_effective_df",
    "get_effective_n", "get_coefficient", "xstd", "xs_sort", "xs_norm",
})
EXPECTED_DSL_OPERATOR_NAMES = ALL_DSL_OPERATOR_NAMES - EXCLUDED_DSL_OPERATOR_NAMES
EXTRA_DSL_OPERATOR_NAMES = EXPECTED_DSL_OPERATOR_NAMES - _BUILTIN_DSL_OPERATOR_NAMES

ROWWISE_RIDGE_COMPOSITE_NAMES = frozenset(
    f"ridge_{name}" for name in REGRESSION_PROJECTIONS
)
GP_COMPOSITE_OPERATOR_NAMES = ROWWISE_RIDGE_COMPOSITE_NAMES | frozenset({
    "ts_poly_regression",
    "xs_regression_neut",
})
GP_CPP_STREAM_UTILITY_OPERATOR_NAMES = ROW_SHAPED_CPP_STREAM_UTIL_NAMES
EXPECTED_GP_OPERATOR_NAMES = (
    EXPECTED_DSL_OPERATOR_NAMES
    | GP_COMPOSITE_OPERATOR_NAMES
    | GP_CPP_STREAM_UTILITY_OPERATOR_NAMES
)


@dataclass(frozen=True)
class GPConfig:
    fields: Mapping[str, SemanticInfo] | None = None
    positive_ints: tuple[int, ...] = (1, 2, 3, 5, 10, 20, 60, 120, 240, 1440)
    positive_floats: tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)
    quantiles: tuple[float, ...] = (0.1, 0.25, 0.5, 0.75, 0.9)
    bools: tuple[bool, ...] = (False, True)
    axes: tuple[int | tuple[int, ...], ...] = (1,)
    datetime_units: tuple[str, ...] = ("us", "ms", "s", "T", "H", "D")
    frequencies: tuple[str | int | float, ...] = ("1s", "1min", "5min", "1H", "1D")
    filter_h: tuple[str, ...] = ("1,2,3,4",)
    filter_t: tuple[str, ...] = ("0.5",)
    kth_ignore: tuple[str, ...] = ("NAN 0", "NAN")
    replace_specs: tuple[tuple[str, str], ...] = (("NAN", "0"), ("0", "NAN"))
    bucket_specs: tuple[tuple[str, str], ...] = (
        ("buckets", "0,0.25,0.5,0.75,1"),
        ("range", "0,1,0.1"),
    )

    def __post_init__(self) -> None:
        if not self.positive_ints or any(
            isinstance(v, bool) or int(v) <= 0 or int(v) != v
            for v in self.positive_ints
        ):
            raise ValueError("positive_ints must contain only positive integers")
        if not self.positive_floats or any(float(v) <= 0.0 for v in self.positive_floats):
            raise ValueError("positive_floats must contain only positive floats")
        if not self.quantiles:
            raise ValueError("quantiles cannot be empty")
        for value in self.quantiles:
            QuantileParam(value)
        if not self.bools:
            raise ValueError("bools cannot be empty")
        if not self.axes:
            raise ValueError("axes cannot be empty")
        for axis in self.axes:
            AxisSpec(axis)
        if not self.replace_specs:
            raise ValueError("replace_specs cannot be empty")
        if not self.bucket_specs:
            raise ValueError("bucket_specs cannot be empty")
        for mode, text in self.bucket_specs:
            if mode not in {"buckets", "range"} or not str(text).strip():
                raise ValueError("bucket_specs entries must be ('buckets'|'range', nonempty_text)")


def _semantic_expr_type(info: SemanticInfo) -> type[NumericRow]:
    closure = DEFAULT_TYPE_GRAPH.closure(info.types)
    if info.boolean:
        return BoolRow
    if "timestamp" in closure:
        return TimestampRow
    if "trading_day_horizon" in closure:
        return TradingDayHorizonRow
    if "duration" in closure:
        return DurationRow
    if "price" in closure:
        return PriceRow
    if "quantity" in closure:
        return QuantityRow
    if "count" in closure:
        return CountRow
    if "dimensionless" in closure:
        return DimensionlessRow
    return DerivedNumericRow


def _safe_name(value: object) -> str:
    text = str(value).replace("-", "neg_").replace(".", "p").replace("+", "pos_")
    text = re.sub(r"[^0-9A-Za-z_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_") or "value"
    return f"v_{text}" if text[0].isdigit() else text


def _type_tag(type_: type) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", type_.__name__).lower().removesuffix("_row")


def _core_call(name: str, ret: type[ExprValue], *values):
    return ret(dsl.call(name, *(unwrap(value) for value in values)))


def _public_call(name: str, ret: type[ExprValue], *values):
    return ret(getattr(dsl, name)(*(unwrap(value) for value in values)))


def _floordiv_call(ret: type[ExprValue], left, right):
    return ret(dsl.floor(dsl.div(unwrap(left), unwrap(right))))


def _clip_call(ret: type[ExprValue], value, lower, upper):
    return ret(dsl.minimum(dsl.maximum(unwrap(value), unwrap(lower)), unwrap(upper)))


def _round_decimals(ret: type[ExprValue], value: ExprValue, decimals: PositiveInt):
    factor = dsl.pow(10.0, decimals.value)
    return ret(dsl.div(dsl.call("round", dsl.mul(value.expr, factor)), factor))


def _floor_ceil_freq_call(name: str, value: TimestampRow, frequency: FrequencySpec):
    return TimestampRow(getattr(dsl, name)(value.expr, freq=frequency.value))


def _round_freq_call(value: TimestampRow, frequency: FrequencySpec):
    return TimestampRow(dsl.round(value.expr, freq=frequency.value))


def _to_dt_call(value: ExprValue, unit: DatetimeUnit):
    return TimestampRow(dsl.to_dt(value.expr, unit=unit.value))


def _safe_shift(ret: type[ExprValue], value: ExprValue, lag: PositiveInt):
    return ret(dsl.call("shift", value.expr, lag.value, lag.value))


def _safe_diff(ret: type[ExprValue], value: ExprValue, lag: PositiveInt):
    return ret(dsl.sub(value.expr, dsl.call("shift", value.expr, lag.value, lag.value)))


def _ewm_call(
    name: str,
    ret: type[ExprValue],
    values: Sequence[ExprValue],
    period: PositiveInt,
    *,
    k: int | None = None,
):
    kwargs: dict[str, object] = {
        "min_periods": period.value,
        "ignore_na": True,
        "adjust": False,
    }
    if k is not None:
        kwargs["k"] = k
    return ret(
        dsl.call(
            name,
            *(value.expr for value in values),
            period.value,
            **kwargs,
        )
    )


def _ewm1(name: str, ret: type[ExprValue], value: ExprValue, period: PositiveInt):
    return _ewm_call(name, ret, (value,), period)


def _ewm_moment(ret: type[ExprValue], value: ExprValue, period: PositiveInt):
    return _ewm_call("ewm_moment", ret, (value,), period, k=2)


def _ewm2(name: str, ret: type[ExprValue], x: ExprValue, y: ExprValue, period: PositiveInt):
    return _ewm_call(name, ret, (x, y), period)


def _ewm3(name: str, ret: type[ExprValue], x: ExprValue, y: ExprValue, z: ExprValue, period: PositiveInt):
    return _ewm_call(name, ret, (x, y, z), period)


def _rolling_simple(name: str, ret: type[ExprValue], value: ExprValue, periods: PositiveInt):
    return ret(dsl.call(name, value.expr, periods.value, min_periods=periods.value))


def _rolling_std(ret: type[ExprValue], value: ExprValue, periods: PositiveInt):
    return ret(
        dsl.call(
            "rolling_std",
            value.expr,
            periods.value,
            min_periods=periods.value,
            ddof=0,
        )
    )


def _rolling_quantile(ret: type[ExprValue], value: ExprValue, periods: PositiveInt, q: QuantileParam):
    return ret(
        dsl.call(
            "rolling_quantile",
            value.expr,
            periods.value,
            q=q.value,
            min_periods=periods.value,
        )
    )


def _rolling_kth(ret: type[ExprValue], value: ExprValue, periods: PositiveInt, ignore: KthIgnoreSpec):
    return ret(
        dsl.call(
            "rolling_kth",
            value.expr,
            periods.value,
            k=1,
            ignore=ignore.value,
            min_periods=periods.value,
        )
    )


def _rolling_entropy(value: ExprValue, periods: PositiveInt):
    return DimensionlessRow(
        dsl.call(
            "rolling_entropy",
            value.expr,
            periods.value,
            buckets=10,
            min_periods=periods.value,
        )
    )


def _rolling_scale(ret: type[ExprValue], value: ExprValue, periods: PositiveInt):
    return ret(
        dsl.call(
            "rolling_scale",
            value.expr,
            periods.value,
            constant=0.0,
            min_periods=periods.value,
        )
    )


def _rolling_theilsen(y: ExprValue, x: ExprValue, periods: PeriodAtLeastTwo):
    return DerivedNumericRow(
        dsl.call(
            "rolling_theilsen",
            y.expr,
            x.expr,
            periods.value,
            min_periods=periods.value,
        )
    )


def _reduction_call(name: str, ret: type[ExprValue], value: ExprValue, axis: AxisSpec, ignore_na: BoolParam):
    return ret(
        broadcast_reduction(
            name,
            value.expr,
            axis=axis.value,
            ignore_na=ignore_na.value,
        )
    )


def _std_reduction_call(ret: type[ExprValue], value: ExprValue, axis: AxisSpec, ignore_na: BoolParam):
    return ret(
        broadcast_reduction(
            "std",
            value.expr,
            axis=axis.value,
            ddof=0,
            ignore_na=ignore_na.value,
        )
    )


def _broadcast_xs_call(name: str, ret: type[ExprValue], value: ExprValue, q: QuantileParam):
    return ret(broadcast_xs_unary(name, value.expr, q.value))


def _regression_ret_type(name: str) -> type[NumericRow]:
    if name in {"r2", "effective_df", "effective_n"}:
        return DimensionlessRow
    return DerivedNumericRow


def _temporal_regression_call(name: str, y: ExprValue, x: ExprValue, periods: PositiveInt):
    ret = _regression_ret_type(name)
    return ret(temporal_ridge_projection(name, y.expr, x.expr, periods.value))


def _rowwise_regression_call(name: str, *values: ExprValue):
    ret = _regression_ret_type(name)
    y, *regressors = values
    return ret(rowwise_ridge_projection(name, y.expr, tuple(value.expr for value in regressors)))


def _poly_regression_call(degree: int, y: ExprValue, x: ExprValue, periods: PositiveInt):
    return DerivedNumericRow(
        temporal_poly_regression_residual(y.expr, x.expr, periods.value, degree)
    )


def _xs_regression_neut_call(y: ExprValue, x: ExprValue):
    return DerivedNumericRow(xs_regression_neutralize(y.expr, x.expr))


class _Registrar:
    def __init__(self, pset: gp.PrimitiveSetTyped) -> None:
        self.pset = pset
        self.families: set[str] = set()
        self.primitive_family: dict[str, str] = {}
        self._names: set[str] = set()

    def add(self, family: str, fn, args: Sequence[type], ret: type, *, variant: str) -> None:
        base = {"and": "and_op", "or": "or_op"}.get(family, family)
        name = _safe_name(f"{base}__{variant}")
        if name in self._names:
            raise ValueError(f"duplicate GP primitive name {name!r}")
        self.pset.addPrimitive(fn, list(args), ret, name=name)
        self._names.add(name)
        self.families.add(family)
        self.primitive_family[name] = family


def _core(reg: _Registrar, family: str, args: Sequence[type], ret: type, variant: str, *, op: str | None = None) -> None:
    reg.add(family, partial(_core_call, op or family, ret), args, ret, variant=variant)


def _public(reg: _Registrar, family: str, args: Sequence[type], ret: type, variant: str) -> None:
    reg.add(family, partial(_public_call, family, ret), args, ret, variant=variant)


def _add_terminal(pset, value, ret_type: type, name: str) -> None:
    pset.addTerminal(value, ret_type, name=_safe_name(name))


def _add_terminals(pset: gp.PrimitiveSetTyped, config: GPConfig) -> dict[str, str]:
    metadata = dict(
        inputdata_alpha_terminal_metadata()
        if config.fields is None
        else config.fields
    )
    fields: dict[str, str] = {}
    for field_name, info in metadata.items():
        row_type = _semantic_expr_type(info)
        terminal_name = _safe_name(f"field_{field_name}")
        _add_terminal(pset, row_type(dsl.var(field_name)), row_type, terminal_name)
        fields[field_name] = terminal_name
    positive_int_values = sorted({int(v) for v in config.positive_ints})
    for value in positive_int_values:
        _add_terminal(pset, PositiveInt(value), PositiveInt, f"positive_int_{value}")
        if value >= 2:
            _add_terminal(pset, PeriodAtLeastTwo(value), PeriodAtLeastTwo, f"period_ge_2_{value}")
    for value in sorted({float(v) for v in config.positive_floats}):
        _add_terminal(
            pset,
            PositiveFloat(value),
            PositiveFloat,
            f"positive_float_{_safe_name(f'{value:g}')}",
        )
    for value in sorted({float(v) for v in config.quantiles}):
        _add_terminal(
            pset,
            QuantileParam(value),
            QuantileParam,
            f"quantile_{_safe_name(f'{value:g}')}",
        )
    for value in config.bools:
        _add_terminal(pset, BoolParam(value), BoolParam, f"bool_param_{str(value).lower()}")
    for i, value in enumerate(config.axes):
        _add_terminal(pset, AxisSpec(value), AxisSpec, f"axis_{i}_{value}")
    for value in config.datetime_units:
        _add_terminal(pset, DatetimeUnit(value), DatetimeUnit, f"dt_unit_{value}")
    for i, value in enumerate(config.frequencies):
        _add_terminal(pset, FrequencySpec(value), FrequencySpec, f"frequency_{i}_{value}")
    for i, value in enumerate(config.filter_h):
        _add_terminal(pset, FilterHSpec(value), FilterHSpec, f"filter_h_{i}")
    for i, value in enumerate(config.filter_t):
        _add_terminal(pset, FilterTSpec(value), FilterTSpec, f"filter_t_{i}")
    for i, value in enumerate(config.kth_ignore):
        _add_terminal(pset, KthIgnoreSpec(value), KthIgnoreSpec, f"kth_ignore_{i}")
    return fields


def _same_type_ops(reg: _Registrar) -> None:
    for row_type in VALUE_TYPES:
        tag = _type_tag(row_type)
        if row_type is not TimestampRow:
            _core(reg, "add", (row_type, row_type), row_type, tag)
        for name in ("minimum", "maximum", "fillna"):
            _core(reg, name, (row_type, row_type), row_type, tag)
        sub_ret = DurationRow if row_type is TimestampRow else row_type
        _core(reg, "sub", (row_type, row_type), sub_ret, tag)
        for name in ("eq", "ne", "lt", "gt", "le", "ge"):
            _core(reg, name, (row_type, row_type), BoolRow, tag)
        _core(reg, "where", (BoolRow, row_type, row_type), row_type, tag)
        reg.add("clip", partial(_clip_call, row_type), (row_type, row_type, row_type), row_type, variant=tag)


def _scalar_broadcast_ops(reg: _Registrar) -> None:
    for row_type in VALUE_TYPES:
        tag = _type_tag(row_type)
        if row_type is not TimestampRow:
            _core(reg, "add", (row_type, PositiveNumber), row_type, f"{tag}_scalar")
            _core(reg, "add", (PositiveNumber, row_type), row_type, f"scalar_{tag}")
            _core(reg, "sub", (row_type, PositiveNumber), row_type, f"{tag}_scalar")
        for name in ("minimum", "maximum"):
            _core(reg, name, (row_type, PositiveNumber), row_type, f"{tag}_scalar")
            _core(reg, name, (PositiveNumber, row_type), row_type, f"scalar_{tag}")
        _core(reg, "fillna", (row_type, PositiveNumber), row_type, f"{tag}_scalar")
        reg.add("clip", partial(_clip_call, row_type), (row_type, PositiveNumber, PositiveNumber), row_type, variant=f"{tag}_scalar_bounds")
        for name in ("eq", "ne", "lt", "gt", "le", "ge"):
            _core(reg, name, (row_type, PositiveNumber), BoolRow, f"{tag}_scalar")
            _core(reg, name, (PositiveNumber, row_type), BoolRow, f"scalar_{tag}")
        _core(reg, "where", (BoolRow, row_type, PositiveNumber), row_type, f"{tag}_scalar_false")
        _core(reg, "where", (BoolRow, PositiveNumber, row_type), row_type, f"scalar_true_{tag}")
        _core(reg, "mul", (row_type, PositiveNumber), row_type, f"{tag}_scalar")
        _core(reg, "mul", (PositiveNumber, row_type), row_type, f"scalar_{tag}")
        _core(reg, "div", (row_type, PositiveNumber), row_type, f"{tag}_scalar")
        inv_ret = DimensionlessRow if issubclass(row_type, DimensionlessRow) else DerivedNumericRow
        _core(reg, "div", (PositiveNumber, row_type), inv_ret, f"scalar_{tag}")
        reg.add("floordiv", partial(_floordiv_call, row_type), (row_type, PositiveNumber), row_type, variant=f"{tag}_scalar")
        reg.add("floordiv", partial(_floordiv_call, inv_ret), (PositiveNumber, row_type), inv_ret, variant=f"scalar_{tag}")
        _core(reg, "mod", (row_type, PositiveNumber), row_type, f"{tag}_scalar")
        _core(reg, "mod", (PositiveNumber, row_type), DerivedNumericRow, f"scalar_{tag}")
        pow_ret = DimensionlessRow if issubclass(row_type, DimensionlessRow) else DerivedNumericRow
        _core(reg, "pow", (row_type, PositiveNumber), pow_ret, f"{tag}_scalar")
    for family in ("and", "and_", "or", "or_", "xor"):
        op = {"and": "and_", "or": "or_"}.get(family, family)
        _core(reg, family, (BoolRow, BoolParam), BoolRow, "row_scalar", op=op)
        _core(reg, family, (BoolParam, BoolRow), BoolRow, "scalar_row", op=op)


def _binary_numeric_ops(reg: _Registrar) -> None:
    for left in VALUE_TYPES:
        for right in VALUE_TYPES:
            variant = f"{_type_tag(left)}_{_type_tag(right)}"
            if issubclass(left, DimensionlessRow) and not issubclass(right, DimensionlessRow):
                mul_ret = right
            elif issubclass(right, DimensionlessRow) and not issubclass(left, DimensionlessRow):
                mul_ret = left
            elif issubclass(left, DimensionlessRow) and issubclass(right, DimensionlessRow):
                mul_ret = DimensionlessRow
            else:
                mul_ret = DerivedNumericRow
            _core(reg, "mul", (left, right), mul_ret, variant)
            if left is right and left not in {TimestampRow, TradingDayHorizonRow}:
                div_ret = DimensionlessRow
            elif issubclass(right, DimensionlessRow):
                div_ret = left
            else:
                div_ret = DerivedNumericRow
            _core(reg, "div", (left, right), div_ret, variant)
            reg.add("floordiv", partial(_floordiv_call, div_ret), (left, right), div_ret, variant=variant)
            _core(reg, "mod", (left, right), left if left is right else DerivedNumericRow, variant)


def _logical_ops(reg: _Registrar) -> None:
    for family in ("and", "and_", "or", "or_", "xor"):
        op = {"and": "and_", "or": "or_"}.get(family, family)
        _core(reg, family, (BoolRow, BoolRow), BoolRow, "bool", op=op)
    _core(reg, "logical_not", (BoolRow,), BoolRow, "bool")


def _unary_ops(reg: _Registrar) -> None:
    for row_type in VALUE_TYPES:
        tag = _type_tag(row_type)
        for name in ("abs", "purify", "cumsum"):
            _core(reg, name, (row_type,), row_type, tag)
        for name in ("isnan", "isfinite"):
            _core(reg, name, (row_type,), BoolRow, tag)
        for name in ("floor", "ceil"):
            _core(reg, name, (row_type,), row_type, f"{tag}_scalar")
    for name in (
        "ln", "exp", "sign", "fraction", "arctan", "acos", "asin", "sin",
        "cos", "tan", "tanh", "norm_inv", "xs_rank", "xs_pct_rank",
    ):
        _core(reg, name, (NumericRow,), DimensionlessRow, "numeric")
    _core(reg, "sqrt", (NumericRow,), DerivedNumericRow, "numeric")


def _ewm_ops(reg: _Registrar) -> None:
    for row_type in VALUE_TYPES:
        tag = _type_tag(row_type)
        reg.add("ewm", partial(_ewm1, "ewm", row_type), (row_type, PositiveInt), row_type, variant=tag)
        reg.add("ewm_std", partial(_ewm1, "ewm_std", row_type), (row_type, PositiveInt), row_type, variant=tag)
        reg.add("ewm_var", partial(_ewm1, "ewm_var", DerivedNumericRow), (row_type, PositiveInt), DerivedNumericRow, variant=tag)
        reg.add("ewm_moment", partial(_ewm_moment, DerivedNumericRow), (row_type, PositiveInt), DerivedNumericRow, variant=tag)
        for name in ("ewm_skewness", "ewm_kurtosis"):
            reg.add(name, partial(_ewm1, name, DimensionlessRow), (row_type, PositiveInt), DimensionlessRow, variant=tag)
    for name, ret in (("ewm_cov", DerivedNumericRow), ("ewm_corr", DimensionlessRow)):
        reg.add(name, partial(_ewm2, name, ret), (NumericRow, NumericRow, PositiveInt), ret, variant="numeric")
    for name in ("ewm_co_skewness", "ewm_co_kurtosis"):
        reg.add(name, partial(_ewm2, name, DerivedNumericRow), (NumericRow, NumericRow, PositiveInt), DerivedNumericRow, variant="numeric")
    for name in ("ewm_triple_corr", "ewm_partial_corr"):
        reg.add(name, partial(_ewm3, name, DimensionlessRow), (NumericRow, NumericRow, NumericRow, PositiveInt), DimensionlessRow, variant="numeric")


def _rolling_ops(reg: _Registrar) -> None:
    preserving = (
        "roll_mean", "rolling_sum", "rolling_mean", "rolling_min", "rolling_max",
        "rolling_median", "rolling_product", "rolling_decay_linear", "rolling_range",
        "rolling_zscore",
    )
    for row_type in VALUE_TYPES:
        tag = _type_tag(row_type)
        for name in preserving:
            ret = DimensionlessRow if name == "rolling_zscore" else row_type
            reg.add(name, partial(_rolling_simple, name, ret), (row_type, PositiveInt), ret, variant=tag)
        reg.add("rolling_std", partial(_rolling_std, row_type), (row_type, PositiveInt), row_type, variant=tag)
        reg.add("rolling_quantile", partial(_rolling_quantile, row_type), (row_type, PositiveInt, QuantileParam), row_type, variant=tag)
        reg.add("rolling_prev_diff", partial(_core_call, "rolling_prev_diff", row_type), (row_type, PeriodAtLeastTwo), row_type, variant=tag)
        reg.add("rolling_kth", partial(_rolling_kth, row_type), (row_type, PositiveInt, KthIgnoreSpec), row_type, variant=tag)
        reg.add("rolling_scale", partial(_rolling_scale, row_type), (row_type, PositiveInt), row_type, variant=tag)
    for name, ret in (
        ("rolling_pct_rank", DimensionlessRow),
        ("rolling_argmin", CountRow),
        ("rolling_argmax", CountRow),
    ):
        reg.add(name, partial(_rolling_simple, name, ret), (NumericRow, PositiveInt), ret, variant="numeric")
    reg.add("rolling_entropy", _rolling_entropy, (NumericRow, PositiveInt), DimensionlessRow, variant="numeric")
    reg.add("rolling_theilsen", _rolling_theilsen, (NumericRow, NumericRow, PeriodAtLeastTwo), DerivedNumericRow, variant="numeric")


def _alpha_ops(reg: _Registrar) -> None:
    _core(reg, "periods_since_last_change", (NumericRow,), CountRow, "numeric")
    for row_type in VALUE_TYPES:
        tag = _type_tag(row_type)
        _core(reg, "hump", (row_type,), row_type, f"{tag}_default")
        _core(reg, "hump", (row_type, PositiveFloat), row_type, tag)
        _core(reg, "hump_decay", (row_type,), row_type, f"{tag}_default")
        _core(reg, "hump_decay", (row_type, PositiveFloat, BoolParam), row_type, f"{tag}_full")
        _core(reg, "trade_when", (BoolRow, row_type, BoolRow), row_type, tag)
        _core(reg, "filter", (row_type,), row_type, f"{tag}_default")
        _core(reg, "filter", (row_type, FilterHSpec, FilterTSpec), row_type, f"{tag}_full")


def _cross_sectional_ops(reg: _Registrar) -> None:
    aggregates = ("xs_sum", "xs_mean", "xs_std", "xs_min", "xs_max", "xs_median")
    for row_type in VALUE_TYPES:
        tag = _type_tag(row_type)
        for name in aggregates:
            _core(reg, name, (row_type,), row_type, tag)
        _core(reg, "xs_count", (row_type,), CountRow, tag)
        reg.add("xs_quantile_value", partial(_core_call, "xs_quantile_value", row_type), (row_type, QuantileParam), row_type, variant=tag)
        _core(reg, "xs_weighted_mean", (row_type, DimensionlessRow), row_type, tag)
        reg.add("vec_quantile", partial(_broadcast_xs_call, "vec_quantile", row_type), (row_type, QuantileParam), row_type, variant=tag)
        _core(reg, "xs_demean", (row_type,), row_type, tag)
        for name in ("xs_vector_projection", "xs_regression_projection"):
            _core(reg, name, (row_type, NumericRow), row_type, tag)
        _core(reg, "xs_scale", (row_type,), row_type, f"{tag}_default")
        _core(reg, "xs_scale", (row_type, PositiveFloat), row_type, f"{tag}_scale")
        for name in ("xs_vector_proj", "xs_vector_neut"):
            _core(reg, name, (row_type, NumericRow), row_type, tag)
    _core(reg, "densify", (NumericRow,), CountRow, "numeric")
    _core(reg, "xs_generalized_rank", (NumericRow,), DimensionlessRow, "default")
    _core(reg, "xs_generalized_rank", (NumericRow, PositiveFloat), DimensionlessRow, "m")
    for name in ("xs_zscore", "xs_direction"):
        _core(reg, name, (NumericRow,), DimensionlessRow, "numeric")


def _time_and_history_ops(reg: _Registrar) -> None:
    for row_type in VALUE_TYPES:
        tag = _type_tag(row_type)
        _core(reg, "ffill", (row_type, PositiveInt), row_type, tag)
        reg.add("shift", partial(_safe_shift, row_type), (row_type, PositiveInt), row_type, variant=tag)
        diff_ret = DurationRow if row_type is TimestampRow else row_type
        reg.add("diff", partial(_safe_diff, diff_ret), (row_type, PositiveInt), diff_ret, variant=tag)
    _public(reg, "ratio", (NumericRow, NumericRow), DimensionlessRow, "numeric")
    for name in ("floor", "ceil"):
        reg.add(name, partial(_floor_ceil_freq_call, name), (TimestampRow, FrequencySpec), TimestampRow, variant="timestamp_frequency")
    _public(reg, "round", (NumericRow,), DerivedNumericRow, "default")
    reg.add("round", partial(_round_decimals, DerivedNumericRow), (NumericRow, PositiveInt), DerivedNumericRow, variant="decimals")
    reg.add("round", _round_freq_call, (TimestampRow, FrequencySpec), TimestampRow, variant="timestamp_frequency")
    reg.add("to_dt", _to_dt_call, (NumericRow, DatetimeUnit), TimestampRow, variant="unit")
    for name in (
        "timeofday", "hour", "minute", "second", "year", "month", "day",
        "dayofweek", "dayofyear",
    ):
        _public(reg, name, (TimestampRow,), DurationRow if name == "timeofday" else CountRow, "timestamp")


def _regression_ops(reg: _Registrar) -> None:
    for name in REGRESSION_PROJECTIONS:
        ret = _regression_ret_type(name)
        reg.add(
            "ts_regression",
            partial(_temporal_regression_call, name),
            (NumericRow, NumericRow, PositiveInt),
            ret,
            variant=name,
        )
        family = f"ridge_{name}"
        for regressors in (1, 2, 3):
            reg.add(
                family,
                partial(_rowwise_regression_call, name),
                (NumericRow,) * (regressors + 1),
                ret,
                variant=f"{regressors}x",
            )
    for degree in (1, 2, 3):
        reg.add(
            "ts_poly_regression",
            partial(_poly_regression_call, degree),
            (NumericRow, NumericRow, PositiveInt),
            DerivedNumericRow,
            variant=f"degree_{degree}",
        )
    reg.add(
        "xs_regression_neut",
        _xs_regression_neut_call,
        (NumericRow, NumericRow),
        DerivedNumericRow,
        variant="numeric",
    )


def _reductions(reg: _Registrar) -> None:
    for row_type in VALUE_TYPES:
        tag = _type_tag(row_type)
        for name in ("sum", "mean", "reduce_min", "reduce_max"):
            if name == "sum" and row_type is BoolRow:
                ret = CountRow
            elif name == "mean" and row_type is BoolRow:
                ret = DimensionlessRow
            else:
                ret = row_type
            reg.add(name, partial(_reduction_call, name, ret), (row_type, AxisSpec, BoolParam), ret, variant=tag)
        std_ret = DimensionlessRow if row_type is BoolRow else row_type
        reg.add("std", partial(_std_reduction_call, std_ret), (row_type, AxisSpec, BoolParam), std_ret, variant=tag)


def make_pset(config: GPConfig | None = None) -> gp.PrimitiveSetTyped:
    """Build the full strongly typed GP grammar.

    Precedence is direct/core DSL first, then canonical row-shaped cpp_stream
    utilities, then only the GP-specific composites needed for otherwise
    unrepresentable intermediate shapes.
    """

    config = config or GPConfig()
    pset = gp.PrimitiveSetTyped("alpha", [], NumericRow)
    field_terminal_names = _add_terminals(pset, config)
    reg = _Registrar(pset)
    for register in (
        _same_type_ops,
        _scalar_broadcast_ops,
        _binary_numeric_ops,
        _logical_ops,
        _unary_ops,
        _ewm_ops,
        _rolling_ops,
        _alpha_ops,
        _cross_sectional_ops,
        _time_and_history_ops,
        _regression_ops,
        _reductions,
    ):
        register(reg)

    added_utility_families = register_cpp_stream_utils(
        reg,
        config,
        skip_names=reg.families,
    )

    missing = EXPECTED_GP_OPERATOR_NAMES - reg.families
    unexpected = reg.families - EXPECTED_GP_OPERATOR_NAMES
    if missing or unexpected:
        raise AssertionError(
            f"GP primitive coverage mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    pset.gp_operator_families = frozenset(reg.families)
    pset.gp_dsl_operator_families = EXPECTED_DSL_OPERATOR_NAMES
    pset.gp_composite_operator_families = GP_COMPOSITE_OPERATOR_NAMES
    pset.gp_cpp_stream_utility_families = GP_CPP_STREAM_UTILITY_OPERATOR_NAMES
    pset.gp_added_cpp_stream_utility_families = added_utility_families
    pset.gp_non_row_cpp_stream_utility_families = NON_ROW_CPP_STREAM_UTIL_NAMES
    pset.gp_excluded_operator_families = EXCLUDED_DSL_OPERATOR_NAMES
    pset.gp_primitive_family = dict(reg.primitive_family)
    pset.gp_field_terminals = dict(field_terminal_names)
    pset.gp_config = config
    return pset


def primitive_names_for_operator(pset: gp.PrimitiveSetTyped, operator: str) -> tuple[str, ...]:
    families = getattr(pset, "gp_primitive_family", {})
    return tuple(sorted(name for name, family in families.items() if family == operator))


__all__ = [
    "ALL_DSL_OPERATOR_NAMES",
    "EXCLUDED_DSL_OPERATOR_NAMES",
    "EXPECTED_DSL_OPERATOR_NAMES",
    "EXPECTED_GP_OPERATOR_NAMES",
    "EXTRA_DSL_OPERATOR_NAMES",
    "GP_COMPOSITE_OPERATOR_NAMES",
    "GP_CPP_STREAM_UTILITY_OPERATOR_NAMES",
    "GPConfig",
    "NON_ROW_CPP_STREAM_UTIL_NAMES",
    "ROWWISE_RIDGE_COMPOSITE_NAMES",
    "ROW_SHAPED_CPP_STREAM_UTIL_NAMES",
    "make_pset",
    "primitive_names_for_operator",
]
