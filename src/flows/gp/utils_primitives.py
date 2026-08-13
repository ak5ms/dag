from __future__ import annotations

from functools import partial
from typing import Iterable, Sequence

from flows.gp.types import (
    BoolParam,
    BoolRow,
    CountRow,
    DerivedNumericRow,
    DimensionlessRow,
    ExprValue,
    KthIgnoreSpec,
    NumericRow,
    PeriodAtLeastTwo,
    PositiveFloat,
    PositiveInt,
    PositiveNumber,
    TimestampRow,
    DurationRow,
    TradingDayHorizonRow,
    VALUE_TYPES,
    unwrap,
)
from trading_dsl_engine.cpp_stream.python import utils as cpp_stream_utils


# These utilities genuinely leave lane-shaped GP expression space. They are the
# only canonical cpp_stream utility functions intentionally unavailable to GP.
NON_ROW_CPP_STREAM_UTIL_NAMES = frozenset({
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
})

ALL_CPP_STREAM_UTIL_NAMES = frozenset(cpp_stream_utils.__all__)
ROW_SHAPED_CPP_STREAM_UTIL_NAMES = ALL_CPP_STREAM_UTIL_NAMES - NON_ROW_CPP_STREAM_UTIL_NAMES

_GROUP_KEY_TYPES = (CountRow, BoolRow, TradingDayHorizonRow)


def _call(fn, ret: type[ExprValue], *values, **kwargs):
    return ret(
        fn(
            *(unwrap(value) for value in values),
            **{name: unwrap(value) for name, value in kwargs.items()},
        )
    )


def _preserve(reg, name: str, *, fn=None, tails: Sequence[tuple[Sequence[type], str]] = ((), "default")) -> None:
    fn = fn or getattr(cpp_stream_utils, name)
    for row_type in VALUE_TYPES:
        for tail, variant in tails:
            reg.add(
                name,
                partial(_call, fn, row_type),
                (row_type, *tail),
                row_type,
                variant=f"{row_type.__name__}_{variant}",
            )


def _numeric(reg, name: str, ret: type[NumericRow], args: Sequence[type], *, fn=None, variant: str = "numeric") -> None:
    reg.add(
        name,
        partial(_call, fn or getattr(cpp_stream_utils, name), ret),
        args,
        ret,
        variant=variant,
    )


def _nan_out(mode: str, ret: type[ExprValue], value: ExprValue, *bounds):
    kwargs = {}
    if mode in {"lower", "both"}:
        kwargs["lower"] = unwrap(bounds[0])
    if mode == "upper":
        kwargs["upper"] = unwrap(bounds[0])
    elif mode == "both":
        kwargs["upper"] = unwrap(bounds[1])
    return ret(cpp_stream_utils.nan_out(value.expr, **kwargs))


def _replace(spec: tuple[str, str], ret: type[ExprValue], value: ExprValue):
    target, dest = spec
    return ret(cpp_stream_utils.replace(value.expr, target, dest))


def _bucket(spec: tuple[str, str], value: ExprValue):
    mode, text = spec
    kwargs = {"buckets": text} if mode == "buckets" else {"range_": text}
    return CountRow(cpp_stream_utils.bucket(value.expr, **kwargs))


def _one_side(side: str, ret: type[ExprValue], value: ExprValue):
    return ret(cpp_stream_utils.xs_one_side(value.expr, side=side))


def _density(fn, driver: str, value: ExprValue, *args):
    raw = [unwrap(arg) for arg in args]
    if fn is cpp_stream_utils.ts_prob_density:
        periods = raw[0]
        sigma = raw[1] if len(raw) > 1 else 1.0
        expr = fn(value.expr, periods, driver=driver, sigma=sigma)
    else:
        sigma = raw[0] if raw else 1.0
        expr = fn(value.expr, driver=driver, sigma=sigma)
    return DimensionlessRow(expr)


def _ts_returns(mode: int, value: ExprValue, periods: PositiveInt):
    return DimensionlessRow(
        cpp_stream_utils.ts_returns(value.expr, periods.value, mode=mode)
    )


def _ts_backfill(ret: type[ExprValue], value: ExprValue, periods: PositiveInt, ignore: KthIgnoreSpec | None = None):
    kwargs = {"k": 1}
    if ignore is not None:
        kwargs["ignore"] = ignore.value
    return ret(cpp_stream_utils.ts_backfill(value.expr, periods.value, **kwargs))


def _ewm_vector(fn, ret: type[ExprValue], x: ExprValue, y: ExprValue, span: PositiveInt):
    return ret(
        fn(
            x.expr,
            y.expr,
            span.value,
            min_periods=span.value,
            ignore_na=True,
            adjust=False,
        )
    )


def _rank_gmean_time(arity: int, *values):
    rows = values[:arity]
    periods = values[arity]
    return DimensionlessRow(
        cpp_stream_utils.ts_rank_gmean_amean_diff(
            *(value.expr for value in rows),
            periods=periods.value,
        )
    )


def _group_mean(ret: type[ExprValue], x: ExprValue, group: ExprValue):
    return ret(cpp_stream_utils.group_mean(x.expr, 1.0, group.expr))


def _group_extra(ret: type[ExprValue], x: ExprValue, group: ExprValue):
    return ret(cpp_stream_utils.group_extra(x.expr, 1.0, group.expr))


def _group_normalize(ret: type[ExprValue], x: ExprValue, group: ExprValue, *params):
    kwargs = {}
    if params:
        kwargs = {
            "constant_check": params[0].value,
            "tolerance": params[1].value,
            "scale": params[2].value,
        }
    return ret(cpp_stream_utils.group_normalize(x.expr, group.expr, **kwargs))


def _register_group_ops(reg) -> set[str]:
    families: set[str] = set()
    preserving = {
        "group_extra": _group_extra,
        "group_max": None,
        "group_mean": _group_mean,
        "group_median": None,
        "group_min": None,
        "group_sum": None,
        "group_vector_neut": None,
        "group_vector_proj": None,
        "group_neutralize": None,
        "xs_group_neutralize": None,
        "xs_market_neutralize": None,
        "group_backfill": None,
    }
    for group_type in _GROUP_KEY_TYPES:
        for row_type in VALUE_TYPES:
            tag = f"{row_type.__name__}_{group_type.__name__}"
            for name, adapter in preserving.items():
                if name in {"group_vector_neut", "group_vector_proj"}:
                    reg.add(
                        name,
                        partial(_call, getattr(cpp_stream_utils, name), row_type),
                        (row_type, NumericRow, group_type),
                        row_type,
                        variant=tag,
                    )
                elif name == "xs_market_neutralize":
                    reg.add(
                        name,
                        partial(_call, cpp_stream_utils.xs_market_neutralize, row_type),
                        (row_type, group_type),
                        row_type,
                        variant=tag,
                    )
                elif name == "group_backfill":
                    reg.add(
                        name,
                        partial(_call, cpp_stream_utils.group_backfill, row_type),
                        (row_type, group_type, PositiveInt),
                        row_type,
                        variant=f"{tag}_default",
                    )
                    reg.add(
                        name,
                        partial(_call, cpp_stream_utils.group_backfill, row_type),
                        (row_type, group_type, PositiveInt, PositiveFloat),
                        row_type,
                        variant=f"{tag}_std",
                    )
                elif adapter is not None:
                    reg.add(
                        name,
                        partial(adapter, row_type),
                        (row_type, group_type),
                        row_type,
                        variant=tag,
                    )
                else:
                    reg.add(
                        name,
                        partial(_call, getattr(cpp_stream_utils, name), row_type),
                        (row_type, group_type),
                        row_type,
                        variant=tag,
                    )
                families.add(name)

            for name, ret in (
                ("group_count", CountRow),
                ("group_na_count", CountRow),
                ("group_rank", DimensionlessRow),
                ("group_scale", DimensionlessRow),
                ("group_std_dev", DerivedNumericRow),
                ("group_zscore", DimensionlessRow),
            ):
                reg.add(
                    name,
                    partial(_call, getattr(cpp_stream_utils, name), ret),
                    (row_type, group_type),
                    ret,
                    variant=tag,
                )
                families.add(name)

            reg.add(
                "group_percentage",
                partial(_call, cpp_stream_utils.group_percentage, row_type),
                (row_type, group_type, PositiveFloat),
                row_type,
                variant=tag,
            )
            families.add("group_percentage")

            reg.add(
                "group_normalize",
                partial(_group_normalize, DerivedNumericRow),
                (row_type, group_type),
                DerivedNumericRow,
                variant=f"{tag}_default",
            )
            reg.add(
                "group_normalize",
                partial(_group_normalize, DerivedNumericRow),
                (row_type, group_type, BoolParam, PositiveFloat, PositiveFloat),
                DerivedNumericRow,
                variant=f"{tag}_full",
            )
            families.add("group_normalize")
    return families


def register_cpp_stream_utils(reg, config, *, skip_names: Iterable[str] = ()) -> frozenset[str]:
    """Register every canonical row-shaped cpp_stream utility not already exposed."""

    skip = set(skip_names)
    added: set[str] = set()

    def mark(name: str) -> bool:
        if name in skip:
            return False
        added.add(name)
        return True

    # Straightforward scalar/row compositions.
    for name, ret in (
        ("log", DimensionlessRow),
        ("inverse", DerivedNumericRow),
        ("log_diff", DimensionlessRow),
        ("s_log_1p", DimensionlessRow),
        ("sigmoid", DimensionlessRow),
        ("arc_cos", DimensionlessRow),
        ("arc_sin", DimensionlessRow),
        ("arc_tan", DimensionlessRow),
    ):
        if mark(name):
            _numeric(reg, name, ret, (NumericRow,))

    for name in ("elementwise_max", "elementwise_min"):
        if mark(name):
            fn = getattr(cpp_stream_utils, name)
            for row_type in VALUE_TYPES:
                for arity in (2, 3, 4):
                    reg.add(
                        name,
                        partial(_call, fn, row_type),
                        (row_type,) * arity,
                        row_type,
                        variant=f"{row_type.__name__}_{arity}",
                    )

    if mark("clamp"):
        for row_type in VALUE_TYPES:
            reg.add("clamp", partial(_call, cpp_stream_utils.clamp, row_type), (row_type,), row_type, variant=f"{row_type.__name__}_default")
            reg.add("clamp", partial(_call, cpp_stream_utils.clamp, row_type), (row_type, PositiveNumber, PositiveNumber), row_type, variant=f"{row_type.__name__}_bounds")
            reg.add("clamp", partial(_call, cpp_stream_utils.clamp, row_type), (row_type, PositiveNumber, PositiveNumber, BoolParam), row_type, variant=f"{row_type.__name__}_inverse")

    if mark("nan_mask"):
        for row_type in VALUE_TYPES:
            reg.add("nan_mask", partial(_call, cpp_stream_utils.nan_mask, row_type), (row_type, NumericRow), row_type, variant=row_type.__name__)

    if mark("nan_out"):
        for row_type in VALUE_TYPES:
            reg.add("nan_out", partial(_nan_out, "lower", row_type), (row_type, PositiveNumber), row_type, variant=f"{row_type.__name__}_lower")
            reg.add("nan_out", partial(_nan_out, "upper", row_type), (row_type, PositiveNumber), row_type, variant=f"{row_type.__name__}_upper")
            reg.add("nan_out", partial(_nan_out, "both", row_type), (row_type, PositiveNumber, PositiveNumber), row_type, variant=f"{row_type.__name__}_both")

    if mark("replace"):
        specs = tuple(getattr(config, "replace_specs", (("NAN", "0"), ("0", "NAN"))))
        for row_type in VALUE_TYPES:
            for index, spec in enumerate(specs):
                reg.add("replace", partial(_replace, tuple(spec), row_type), (row_type,), row_type, variant=f"{row_type.__name__}_{index}")

    for name, tails in (
        ("reverse", (((), "default"),)),
        ("round_down", (((), "default"), ((PositiveFloat,), "factor"))),
        ("to_nan", (((), "default"), ((PositiveNumber, BoolParam), "full"))),
        ("left_tail", (((), "default"), ((PositiveNumber,), "bound"))),
        ("right_tail", (((), "default"), ((PositiveNumber,), "bound"))),
        ("tail", (((), "default"), ((PositiveNumber, PositiveNumber, PositiveNumber), "full"))),
        ("left_right_tail", (((PositiveNumber, PositiveNumber), "bounds"),)),
        ("pasteurize", (((), "default"),)),
        ("convert_float", (((), "default"),)),
    ):
        if mark(name):
            _preserve(reg, name, tails=tails)

    if mark("round_df"):
        _preserve(reg, "round_df", tails=(((PositiveInt,), "decimals"),))

    if mark("signed_power"):
        _numeric(reg, "signed_power", DerivedNumericRow, (NumericRow, NumericRow), variant="row")
        _numeric(reg, "signed_power", DerivedNumericRow, (NumericRow, PositiveNumber), variant="scalar")

    for name in ("negate", "logical_and", "logical_or"):
        if mark(name):
            args = (BoolRow,) if name == "negate" else (BoolRow, BoolRow)
            _numeric(reg, name, BoolRow, args, variant="bool")

    for name in ("is_not_nan", "is_nan", "is_finite", "is_not_finite"):
        if mark(name):
            _numeric(reg, name, BoolRow, (NumericRow,))

    for name in ("equal", "less"):
        if mark(name):
            _numeric(reg, name, BoolRow, (NumericRow, NumericRow))

    if mark("if_else"):
        for row_type in VALUE_TYPES:
            reg.add("if_else", partial(_call, cpp_stream_utils.if_else, row_type), (BoolRow, row_type, row_type), row_type, variant=row_type.__name__)

    if mark("get_df"):
        _numeric(reg, "get_df", DerivedNumericRow, (NumericRow, PositiveNumber))

    if mark("bucket"):
        specs = tuple(getattr(config, "bucket_specs", (("buckets", "0,0.25,0.5,0.75,1"), ("range", "0,1,0.1"))))
        for index, spec in enumerate(specs):
            reg.add("bucket", partial(_bucket, tuple(spec)), (NumericRow,), CountRow, variant=f"spec_{index}")

    # Cross-sectional row utilities.
    if mark("xs_normalize"):
        _numeric(reg, "xs_normalize", DerivedNumericRow, (NumericRow,), variant="default")
        _numeric(reg, "xs_normalize", DerivedNumericRow, (NumericRow, BoolParam, PositiveFloat), variant="full")

    if mark("xs_one_side"):
        for row_type in VALUE_TYPES:
            for side in ("long", "short"):
                reg.add("xs_one_side", partial(_one_side, side, row_type), (row_type,), row_type, variant=f"{row_type.__name__}_{side}")

    if mark("xs_prob_density"):
        for driver in ("gaussian", "uniform", "cauchy"):
            reg.add("xs_prob_density", partial(_density, cpp_stream_utils.xs_prob_density, driver), (NumericRow,), DimensionlessRow, variant=f"{driver}_default")
            reg.add("xs_prob_density", partial(_density, cpp_stream_utils.xs_prob_density, driver), (NumericRow, PositiveFloat), DimensionlessRow, variant=f"{driver}_sigma")

    for name, ret, tails in (
        ("xs_scale_down", DimensionlessRow, ((NumericRow,), (NumericRow, PositiveFloat))),
        ("xs_scale_by_side", DimensionlessRow, ((NumericRow,),)),
        ("xs_rank_by_side", DimensionlessRow, ((NumericRow,), (NumericRow, PositiveFloat, PositiveFloat))),
        ("generalized_rank", DimensionlessRow, ((NumericRow,), (NumericRow, PositiveFloat))),
        ("xs_regression_proj", DerivedNumericRow, ((NumericRow, NumericRow),)),
    ):
        if mark(name):
            for index, args in enumerate(tails):
                _numeric(reg, name, ret, args, variant=str(index))

    for name in ("xs_winsorize", "xs_truncate"):
        if mark(name):
            _preserve(reg, name, tails=(((), "default"), ((PositiveFloat,), "param")))

    if mark("xs_filter"):
        for row_type in VALUE_TYPES:
            reg.add("xs_filter", partial(_call, cpp_stream_utils.xs_filter, row_type), (row_type, PositiveFloat), row_type, variant=f"{row_type.__name__}_default")
            reg.add("xs_filter", partial(_call, cpp_stream_utils.xs_filter, row_type), (row_type, PositiveFloat, BoolParam), row_type, variant=f"{row_type.__name__}_full")

    if mark("xs_rank_gmean_amean_diff"):
        for arity in (2, 3, 4):
            reg.add("xs_rank_gmean_amean_diff", partial(_call, cpp_stream_utils.xs_rank_gmean_amean_diff, DimensionlessRow), (NumericRow,) * arity, DimensionlessRow, variant=f"arity_{arity}")

    # Group utilities are row-shaped compositions. Restrict generated group keys
    # to discrete semantic row types to keep cardinality manageable.
    group_names = _register_group_ops(reg)
    for name in group_names:
        if name in skip:
            # Registration above is intentionally deterministic; skipped names are
            # removed from expected accounting only when already directly exposed.
            continue
        added.add(name)

    # Time-series row utilities.
    if mark("periods_from_last_change"):
        _numeric(reg, "periods_from_last_change", CountRow, (NumericRow,))

    if mark("ts_hump_decay"):
        _preserve(reg, "ts_hump_decay", tails=(((), "default"), ((PositiveFloat, BoolParam), "full")))

    if mark("jump_decay"):
        _preserve(reg, "jump_decay", tails=(((PositiveInt,), "default"), ((PositiveInt, BoolParam, PositiveFloat, PositiveFloat), "full")))

    if mark("keep"):
        for row_type in VALUE_TYPES:
            reg.add("keep", partial(_call, cpp_stream_utils.keep, row_type), (row_type, NumericRow), row_type, variant=f"{row_type.__name__}_default")
            reg.add("keep", partial(_call, cpp_stream_utils.keep, row_type), (row_type, NumericRow, PositiveInt), row_type, variant=f"{row_type.__name__}_period")

    if mark("ts_inst_tvr"):
        _numeric(reg, "ts_inst_tvr", DimensionlessRow, (NumericRow, PositiveInt))

    if mark("ts_backfill"):
        for row_type in VALUE_TYPES:
            reg.add("ts_backfill", partial(_ts_backfill, row_type), (row_type, PositiveInt), row_type, variant=f"{row_type.__name__}_default")
            reg.add("ts_backfill", partial(_ts_backfill, row_type), (row_type, PositiveInt, KthIgnoreSpec), row_type, variant=f"{row_type.__name__}_ignore")

    if mark("prev_diff_value"):
        _preserve(reg, "prev_diff_value", tails=(((PeriodAtLeastTwo,), "period"),))

    if mark("ts_weighted_delay"):
        _preserve(reg, "ts_weighted_delay", tails=(((), "default"), ((PositiveFloat,), "weight")))

    for name in ("ts_shift", "ts_sum", "ts_product", "ts_mean", "ts_median", "ts_min", "ts_max", "ts_std", "ts_decay_linear"):
        if mark(name):
            _preserve(reg, name, tails=(((PositiveInt,), "period"),))

    if mark("ts_diff"):
        for row_type in VALUE_TYPES:
            ret = DurationRow if row_type is TimestampRow else row_type
            reg.add("ts_diff", partial(_call, cpp_stream_utils.ts_diff, ret), (row_type, PositiveInt), ret, variant=row_type.__name__)

    if mark("ts_returns"):
        for mode in (1, 2):
            reg.add("ts_returns", partial(_ts_returns, mode), (NumericRow, PositiveInt), DimensionlessRow, variant=f"mode_{mode}")

    for name in ("ts_pct_change", "ts_ln_change", "ts_ir", "ts_rank", "ts_zscore", "ts_inst_tvr"):
        if mark(name):
            _numeric(reg, name, DimensionlessRow, (NumericRow, PositiveInt))

    if mark("ts_prob_density"):
        for driver in ("gaussian", "uniform", "cauchy"):
            reg.add("ts_prob_density", partial(_density, cpp_stream_utils.ts_prob_density, driver), (NumericRow, PositiveInt), DimensionlessRow, variant=f"{driver}_default")
            reg.add("ts_prob_density", partial(_density, cpp_stream_utils.ts_prob_density, driver), (NumericRow, PositiveInt, PositiveFloat), DimensionlessRow, variant=f"{driver}_sigma")

    if mark("ts_percentage"):
        for row_type in VALUE_TYPES:
            reg.add("ts_percentage", partial(_call, cpp_stream_utils.ts_percentage, row_type), (row_type, PositiveInt, PositiveFloat), row_type, variant=row_type.__name__)

    for name, ret in (("ts_argmax", CountRow), ("ts_argmin", CountRow)):
        if mark(name):
            _numeric(reg, name, ret, (NumericRow, PositiveInt))

    for name in ("ts_mean_diff", "ts_max_diff", "ts_min_diff"):
        if mark(name):
            _preserve(reg, name, tails=(((PositiveInt,), "period"),))

    for name in ("ts_min_max_cps", "ts_min_max_diff"):
        if mark(name):
            _preserve(reg, name, tails=(((PositiveInt,), "default"), ((PositiveInt, PositiveFloat), "factor")))

    if mark("ts_scale"):
        _numeric(reg, "ts_scale", DimensionlessRow, (NumericRow, PositiveInt), variant="default")
        _numeric(reg, "ts_scale", DimensionlessRow, (NumericRow, PositiveInt, PositiveFloat), variant="constant")

    for name in ("ts_count_nans", "ts_count_nonnumeric"):
        if mark(name):
            _numeric(reg, name, CountRow, (NumericRow, PositiveInt))

    if mark("ts_entropy"):
        _numeric(reg, "ts_entropy", DimensionlessRow, (NumericRow, PositiveInt), variant="default")
        _numeric(reg, "ts_entropy", DimensionlessRow, (NumericRow, PositiveInt, PositiveInt), variant="buckets")

    for name in ("ewm_vector_proj", "ewm_vector_neut", "ts_vector_proj", "ts_vector_neut"):
        if mark(name):
            fn = getattr(cpp_stream_utils, name)
            for row_type in VALUE_TYPES:
                reg.add(name, partial(_ewm_vector, fn, row_type), (row_type, NumericRow, PositiveInt), row_type, variant=row_type.__name__)

    if mark("ts_rank_gmean_amean_diff"):
        for arity in (2, 3, 4):
            reg.add("ts_rank_gmean_amean_diff", partial(_rank_gmean_time, arity), (NumericRow,) * arity + (PositiveInt,), DimensionlessRow, variant=f"arity_{arity}")

    if mark("ts_geomean"):
        _numeric(reg, "ts_geomean", DerivedNumericRow, (NumericRow, PositiveInt), variant="default")
        _numeric(reg, "ts_geomean", DerivedNumericRow, (NumericRow, PositiveInt, PositiveNumber), variant="replacement")

    if mark("slope"):
        _numeric(reg, "slope", DerivedNumericRow, (NumericRow, PositiveInt))

    if mark("ts_theilsen"):
        _numeric(reg, "ts_theilsen", DerivedNumericRow, (NumericRow, NumericRow, PeriodAtLeastTwo))

    expected = ROW_SHAPED_CPP_STREAM_UTIL_NAMES - skip
    missing = expected - added
    unexpected = added - expected
    if missing or unexpected:
        raise AssertionError(
            f"cpp_stream utility GP coverage mismatch: missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )
    return frozenset(added)


__all__ = [
    "ALL_CPP_STREAM_UTIL_NAMES",
    "NON_ROW_CPP_STREAM_UTIL_NAMES",
    "ROW_SHAPED_CPP_STREAM_UTIL_NAMES",
    "register_cpp_stream_utils",
]
