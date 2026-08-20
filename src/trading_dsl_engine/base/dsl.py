from __future__ import annotations

import builtins
from collections.abc import Callable, Sequence
from inspect import Parameter, Signature, signature

from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, String, Universe, UniverseItem


class DSLFunctionRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, list[tuple[Signature, Callable[..., Expr]]]] = {}
        self._annotations: dict[str, dict[str, object]] = {}

    def register(self, name: str, fn: Callable[..., Expr], overwrite: bool = True) -> None:
        if not overwrite and name in self._fns:
            raise ValueError(f"DSL function already registered: {name}")
        overloads = self._fns.get(name, [])
        sig = signature(fn)
        for param_name, param in sig.parameters.items():
            if param.annotation is Parameter.empty:
                continue
            annotations = self._annotations.setdefault(name, {})
            previous = annotations.get(param_name, Parameter.empty)
            if previous is not Parameter.empty and previous != param.annotation:
                raise TypeError(
                    f"Conflicting annotation for DSL function {name!r} parameter {param_name!r}: "
                    f"{previous!r} != {param.annotation!r}"
                )
            annotations[param_name] = param.annotation
        if any(_same_signature(existing, sig) for existing, _ in overloads):
            raise ValueError(f"DSL function overload already registered: {name}{sig}")
        overloads.append((sig, fn))
        self._fns[name] = overloads

    def get(self, name: str) -> Callable[..., Expr] | None:
        overloads = self._fns.get(name)
        if not overloads:
            return None

        def _dispatch(*args, **kwargs) -> Expr:
            op_sig = _DSL_OP_SIGNATURES.get(name)
            op_matches = _signature_matches(op_sig, *args, **kwargs)
            explicit_arg_count = len(args) + len(kwargs)
            op_param_count = len(op_sig.parameters) if op_sig is not None else 0
            for sig, fn in overloads:
                match = _signature_match(sig, *args, **kwargs)
                if match is None:
                    continue
                if op_matches and _uses_only_defaults(match):
                    if explicit_arg_count < op_param_count:
                        return ensure_expr(fn(*args, **kwargs))
                    continue
                return ensure_expr(fn(*args, **kwargs))
            if op_matches:
                return call(name, *args, **kwargs)
            raise TypeError(f"No matching DSL function overload for {name}{_call_shape(args, kwargs)}")

        return _dispatch


def _same_signature(left: Signature, right: Signature) -> bool:
    return tuple(left.parameters.values()) == tuple(right.parameters.values())


def _signature_match(sig: Signature | None, *args, **kwargs):
    if sig is None:
        return None
    try:
        return sig.bind(*args, **kwargs)
    except TypeError:
        return None


def _signature_matches(sig: Signature | None, *args, **kwargs) -> bool:
    return _signature_match(sig, *args, **kwargs) is not None


def _uses_only_defaults(bound) -> bool:
    return len(bound.arguments) < len(bound.signature.parameters)


def _call_shape(args, kwargs) -> str:
    suffix = ", " if args and kwargs else ""
    return f"({', '.join(type(arg).__name__ for arg in args)}{suffix}{', '.join(kwargs)})"


DEFAULT_DSL_REGISTRY = DSLFunctionRegistry()


def ensure_expr(value) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, (tuple, list)):
        if len(value) == 0:
            raise TypeError("Key tuples cannot be empty")
        return KeyTuple(tuple(ensure_expr(item) for item in value))
    if isinstance(value, bool):
        return Number(float(value))
    if isinstance(value, int):
        return Number(value)
    if isinstance(value, float):
        return Number(value)
    if isinstance(value, str):
        return String(value)
    raise TypeError(f"Expected Expr|int|float, got {type(value).__name__}")


def univ(*groups: Sequence[UniverseItem] | UniverseItem) -> Universe:
    normalized: list[tuple[UniverseItem, ...]] = []
    for group in groups:
        if isinstance(group, (str, int)):
            members = (group,)
        else:
            members = tuple(group)
        if len(members) == 0:
            raise ValueError("Universe groups cannot be empty")
        for member in members:
            if not isinstance(member, (str, int)):
                raise TypeError(f"Universe members must be str or int, got {type(member).__name__}")
        normalized.append(members)
    if len(normalized) == 0:
        raise ValueError("univ expects at least one group")
    return Universe(tuple(normalized))


def var(name: str) -> Identifier:
    return Identifier(name)


def call(name: str, *args, **kwargs) -> Expr:
    return Call(
        name,
        tuple(ensure_expr(a) for a in args),
        tuple((key, ensure_expr(value)) for key, value in kwargs.items()),
    )


def _axis_expr(axis) -> Expr:
    if isinstance(axis, Expr):
        return axis
    if isinstance(axis, bool):
        return Number(float(axis))
    if isinstance(axis, int):
        return Number(axis)
    if isinstance(axis, float):
        return Number(axis)
    if isinstance(axis, (tuple, list)):
        if not axis:
            raise ValueError("axis cannot be empty")
        return KeyTuple(tuple(_axis_expr(item) for item in axis))
    raise TypeError("axis must be an int or a non-empty list/tuple of ints")


def reduction(name: str, x, *, axis=None, ddof=0, ignore_na=True) -> Expr:
    if name not in {"sum", "mean", "std"}:
        raise ValueError(f"unsupported reduction {name!r}")
    kwargs = []
    if axis is not None:
        kwargs.append(("axis", _axis_expr(axis)))
    if name == "std" and ddof != 0:
        kwargs.append(("ddof", ensure_expr(ddof)))
    if not ignore_na:
        kwargs.append(("ignore_na", ensure_expr(ignore_na)))
    return Call(name, (ensure_expr(x),), tuple(kwargs))


def emit(x, *, mode="last") -> Expr:
    if mode != "last":
        raise ValueError("emit currently supports only mode='last'")
    return Call("emit", (ensure_expr(x),), (("mode", String(mode)),))


GROUPBY_VALUE_PLACEHOLDER = "self_"
self_ = var(GROUPBY_VALUE_PLACEHOLDER)


class GroupedExpr:
    def __init__(self, lhs, key) -> None:
        self.lhs = ensure_expr(lhs)
        key_expr = ensure_expr(key)
        if not isinstance(key_expr, KeyTuple):
            key_expr = KeyTuple((key_expr,))
        if builtins.sum(1 for item in key_expr.items if isinstance(item, Universe)) > 1:
            raise TypeError("groupby key tuple may contain at most one univ(...) element")
        self.key = key_expr

    def apply(self, rhs, *args) -> Expr:
        if callable(rhs):
            rhs_expr = rhs(var(GROUPBY_VALUE_PLACEHOLDER), *args)
        elif args:
            raise TypeError("GroupedExpr.apply args require a callable rhs")
        else:
            rhs_expr = rhs
        return call("groupby", self.key, self.lhs, rhs_expr)

    def __getattr__(self, name: str):
        def _grouped_op(*args) -> Expr:
            return self.apply(lambda value: call(name, value, *args))

        return _grouped_op


def grouped(lhs, key) -> GroupedExpr:
    return GroupedExpr(lhs, key)


def _dsl_signature(*names: str, variadic: str | None = None, defaults: dict[str, object] | None = None) -> Signature:
    defaults = defaults or {}
    params = [Parameter(name, Parameter.POSITIONAL_OR_KEYWORD, default=defaults.get(name, Parameter.empty)) for name in names]
    if variadic is not None:
        params.append(Parameter(variadic, Parameter.VAR_POSITIONAL))
    return Signature(params)


_DSL_OP_SIGNATURES: dict[str, Signature] = {
    **{
        name: _dsl_signature("x")
        for name in {
            "abs",
            "ln",
            "ceil",
            "floor",
            "exp",
            "sign",
            "arctan",
            "acos",
            "asin",
            "sin",
            "cos",
            "tan",
            "tanh",
            "sqrt",
            "isnan",
            "isfinite",
            "logical_not",
            "purify",
            "fraction",
            "norm_inv",
            "xs_norm",
            "xs_rank",
            "xs_pct_rank",
            "xs_sort",
            "xstd",
            "outer",
            "cumsum",
        }
    },
    **{
        name: _dsl_signature("x")
        for name in {
            "get_beta",
            "get_preds",
            "get_residuals",
            "get_sse",
            "get_sst",
            "get_r2",
            "get_residual_variance",
            "get_standard_errors",
            "get_tstats",
            "get_effective_df",
            "get_effective_n",
        }
    },
    **{
        name: _dsl_signature("x", "y")
        for name in {
            "add",
            "sub",
            "mul",
            "mod",
            "pow",
            "div",
            "floordiv",
            "eq",
            "ne",
            "lt",
            "gt",
            "and",
            "and_",
            "or",
            "or_",
            "xor",
            "fillna",
            "minimum",
            "maximum",
        }
    },
    "where": _dsl_signature("condition", "true", "false"),
    "clip": _dsl_signature("x", "lo", "hi"),
    "round": _dsl_signature("x", "decimals"),
    "ewm": _dsl_signature("x", "span", "min_periods", "ignore_na", "adjust", defaults={"min_periods": 0, "ignore_na": True, "adjust": False}),
    "ewm_moment": _dsl_signature(
        "x",
        "span",
        "k",
        "min_periods",
        "ignore_na",
        "adjust",
        defaults={
            "k": 2,
            "min_periods": 0,
            "ignore_na": True,
            "adjust": False,
        },
    ),
    **{
        name: _dsl_signature(
            "x",
            "span",
            "min_periods",
            "ignore_na",
            "adjust",
            defaults={"min_periods": 0, "ignore_na": True, "adjust": False},
        )
        for name in {"ewm_var", "ewm_std", "ewm_skewness", "ewm_kurtosis"}
    },
    **{
        name: _dsl_signature(
            "x",
            "y",
            "span",
            "min_periods",
            "ignore_na",
            "adjust",
            defaults={"min_periods": 0, "ignore_na": True, "adjust": False},
        )
        for name in {"ewm_cov", "ewm_corr"}
    },
    **{
        name: _dsl_signature(
            "y",
            "x",
            "span",
            "min_periods",
            "ignore_na",
            "adjust",
            defaults={"min_periods": 0, "ignore_na": True, "adjust": False},
        )
        for name in {"ewm_co_skewness", "ewm_co_kurtosis"}
    },
    **{
        name: _dsl_signature(
            "x",
            "y",
            "z",
            "span",
            "min_periods",
            "ignore_na",
            "adjust",
            defaults={"min_periods": 0, "ignore_na": True, "adjust": False},
        )
        for name in {"ewm_triple_corr", "ewm_partial_corr"}
    },
    "roll_mean": _dsl_signature(
        "x", "periods", "min_periods", defaults={"min_periods": None}
    ),
    **{
        name: _dsl_signature(
            "x", "periods", "min_periods", defaults={"min_periods": None}
        )
        for name in {
            "rolling_sum",
            "rolling_mean",
            "rolling_min",
            "rolling_max",
            "rolling_median",
            "rolling_pct_rank",
            "rolling_argmin",
            "rolling_argmax",
        }
    },
    "rolling_std": _dsl_signature(
        "x",
        "periods",
        "min_periods",
        "ddof",
        defaults={"min_periods": None, "ddof": 0},
    ),
    "rolling_quantile": _dsl_signature(
        "x",
        "periods",
        "q",
        "min_periods",
        defaults={"q": 0.5, "min_periods": None},
    ),
    "rolling_theilsen": _dsl_signature(
        "y", "x", "periods", "min_periods", defaults={"min_periods": None}
    ),
    "periods_since_last_change": _dsl_signature("x"),
    "hump": _dsl_signature("x", "hump", defaults={"hump": 0.01}),
    "hump_decay": _dsl_signature(
        "x", "p", "relative", defaults={"p": 0.1, "relative": False}
    ),
    "trade_when": _dsl_signature("trigger", "alpha", "exit"),
    "filter": _dsl_signature(
        "x", "h", "t", defaults={"h": "1,2,3,4", "t": "0.5"}
    ),
    **{
        name: _dsl_signature(
            "x", "periods", "min_periods", defaults={"min_periods": None}
        )
        for name in {
            "rolling_product",
            "rolling_decay_linear",
        }
    },
    "rolling_prev_diff": _dsl_signature("x", "periods"),
    "rolling_kth": _dsl_signature(
        "x",
        "periods",
        "k",
        "ignore",
        "min_periods",
        defaults={"k": 1, "ignore": "NAN 0", "min_periods": None},
    ),
    "rolling_entropy": _dsl_signature(
        "x",
        "periods",
        "buckets",
        "min_periods",
        defaults={"buckets": 10, "min_periods": None},
    ),
    "vec_quantile": _dsl_signature("x", "q", defaults={"q": 0.5}),
    **{
        name: _dsl_signature("x")
        for name in {
            "xs_count",
            "xs_sum",
            "xs_mean",
            "xs_std",
            "xs_min",
            "xs_max",
            "xs_median",
            "densify",
        }
    },
    "xs_quantile_value": _dsl_signature("x", "q", defaults={"q": 0.5}),
    "xs_weighted_mean": _dsl_signature("x", "weight"),
    "xs_vector_projection": _dsl_signature("target", "regressor"),
    "xs_regression_projection": _dsl_signature("target", "regressor"),
    "xs_generalized_rank": _dsl_signature("x", "m", defaults={"m": 1.0}),
    **{
        name: _dsl_signature("x")
        for name in {"xs_demean", "xs_zscore", "xs_direction"}
    },
    "xs_scale": _dsl_signature(
        "x",
        "scale",
        "longscale",
        "shortscale",
        defaults={
            "scale": 1.0,
            "longscale": None,
            "shortscale": None,
        },
    ),
    **{
        name: _dsl_signature("x", "y")
        for name in {"xs_vector_proj", "xs_vector_neut"}
    },
    **{
        name: _dsl_signature(
            "x", "periods", "min_periods", defaults={"min_periods": None}
        )
        for name in {"rolling_range", "rolling_zscore"}
    },
    "rolling_scale": _dsl_signature(
        "x",
        "periods",
        "constant",
        "min_periods",
        defaults={"constant": 0.0, "min_periods": None},
    ),
    "ts_regression": _dsl_signature(
        "y",
        "x",
        "periods",
        "lag",
        "rettype",
        "weights",
        "lambda_",
        defaults={
            "lag": 0,
            "rettype": "residual",
            "weights": 1.0,
            "lambda_": 0.0,
        },
    ),
    **{
        name: _dsl_signature("x", "component")
        for name in {"get_coefficient", "get_standard_error", "get_tstat"}
    },
    "ffill": _dsl_signature("x", "limit"),
    "shift": _dsl_signature("x", "lag", "max_lag", defaults={"lag": 1, "max_lag": None}),
    "buffer": _dsl_signature("shift_expr", "min", "max"),
    "cache": _dsl_signature("x", "where"),
    "bspline": _dsl_signature("x", "n_basis"),
    "rbf_basis": _dsl_signature("ev_ts", "session_start", "session_end", "n_basis"),
    "future_rbf_basis_sum": _dsl_signature("ev_ts", "session_start", "session_end", "n_basis", "n_steps"),
    "col": _dsl_signature("matrix", "index"),
    "InstrumentBasisMean": _dsl_signature("features", "y", "weights", "hl"),
    "cat": _dsl_signature(variadic="args"),
    "einsum": _dsl_signature(variadic="args"),
    "groupby": _dsl_signature("key_tuple", "lhs", "op_using_self_"),
    "sum": _dsl_signature(
        "x", "axis", "ignore_na", defaults={"axis": None, "ignore_na": True}
    ),
    "mean": _dsl_signature(
        "x", "axis", "ignore_na", defaults={"axis": None, "ignore_na": True}
    ),
    "std": _dsl_signature(
        "x",
        "axis",
        "ddof",
        "ignore_na",
        defaults={"axis": None, "ddof": 0, "ignore_na": True},
    ),
    "reduce_min": _dsl_signature(
        "x", "axis", "ignore_na", defaults={"axis": None, "ignore_na": True}
    ),
    "reduce_max": _dsl_signature(
        "x", "axis", "ignore_na", defaults={"axis": None, "ignore_na": True}
    ),
    "emit": _dsl_signature("x", "mode", defaults={"mode": "last"}),
}


def get_dsl_op_signature(name: str) -> Signature | None:
    return _DSL_OP_SIGNATURES.get(name)


def op(name: str) -> Callable[..., Expr]:
    def _op(*args, **kwargs) -> Expr:
        return call(name, *args, **kwargs)

    _op.__name__ = name
    _op.__signature__ = _DSL_OP_SIGNATURES.get(name, Signature())
    return _op


def register_dsl_function(name: str | None = None, registry: DSLFunctionRegistry | None = None):
    target = registry or DEFAULT_DSL_REGISTRY

    def _decorator(fn: Callable[..., Expr]) -> Callable[..., Expr]:
        fn_name = name or fn.__name__

        target.register(fn_name, fn)
        dispatch = target.get(fn_name) or fn
        op_signature = _DSL_OP_SIGNATURES.get(fn_name)
        if op_signature is not None:
            dispatch.__signature__ = op_signature
        return dispatch

    return _decorator


add = op("add")
sub = op("sub")
mul = op("mul")
div = op("div")
floordiv = op("floordiv")
mod = op("mod")
eq = op("eq")
ne = op("ne")
and_ = op("and_")
or_ = op("or_")
xor = op("xor")
where = op("where")
lt = op("lt")
gt = op("gt")
le = op("le")
ge = op("ge")
abs = op("abs")
isnan = op("isnan")
fillna = op("fillna")
ffill = op("ffill")
ln = op("ln")
ceil = op("ceil")
floor = op("floor")
exp = op("exp")
sign = op("sign")
fraction = op("fraction")
purify = op("purify")
arctan = op("arctan")
acos = op("acos")
asin = op("asin")
sin = op("sin")
cos = op("cos")
tan = op("tan")
tanh = op("tanh")
sqrt = op("sqrt")
isfinite = op("isfinite")
logical_not = op("logical_not")
minimum = op("minimum")
maximum = op("maximum")
pow = op("pow")
cumsum = op("cumsum")
shift = op("shift")
buffer = op("buffer")
ewm = op("ewm")
roll_mean = op("roll_mean")
xs_rank = op("xs_rank")
xs_pct_rank = op("xs_pct_rank")
norm_inv = op("norm_inv")
xs_norm = op("xs_norm")
clip = op("clip")
cache = op("cache")
outer = op("outer")
bspline = op("bspline")
rbf_basis = op("rbf_basis")
future_rbf_basis_sum = op("future_rbf_basis_sum")
col = op("col")
einsum = op("einsum")
ewm_moment = op("ewm_moment")
ewm_var = op("ewm_var")
ewm_std = op("ewm_std")
ewm_skewness = op("ewm_skewness")
ewm_kurtosis = op("ewm_kurtosis")
ewm_cov = op("ewm_cov")
ewm_corr = op("ewm_corr")
ewm_co_skewness = op("ewm_co_skewness")
ewm_co_kurtosis = op("ewm_co_kurtosis")
ewm_triple_corr = op("ewm_triple_corr")
ewm_partial_corr = op("ewm_partial_corr")
rolling_sum = op("rolling_sum")
rolling_mean = op("rolling_mean")
rolling_std = op("rolling_std")
rolling_min = op("rolling_min")
rolling_max = op("rolling_max")
rolling_median = op("rolling_median")
rolling_quantile = op("rolling_quantile")
rolling_pct_rank = op("rolling_pct_rank")
rolling_argmin = op("rolling_argmin")
rolling_argmax = op("rolling_argmax")
rolling_theilsen = op("rolling_theilsen")
periods_since_last_change = op("periods_since_last_change")
hump = op("hump")
hump_decay = op("hump_decay")
trade_when = op("trade_when")
filter = op("filter")
rolling_product = op("rolling_product")
rolling_kth = op("rolling_kth")
rolling_prev_diff = op("rolling_prev_diff")
rolling_decay_linear = op("rolling_decay_linear")
rolling_entropy = op("rolling_entropy")
vec_quantile = op("vec_quantile")
xs_count = op("xs_count")
xs_sum = op("xs_sum")
xs_mean = op("xs_mean")
xs_std = op("xs_std")
xs_min = op("xs_min")
xs_max = op("xs_max")
xs_median = op("xs_median")
xs_quantile_value = op("xs_quantile_value")
xs_weighted_mean = op("xs_weighted_mean")
xs_vector_projection = op("xs_vector_projection")
xs_regression_projection = op("xs_regression_projection")
xs_generalized_rank = op("xs_generalized_rank")
densify = op("densify")

cat = op("cat")
groupby = op("groupby")
sum = op("sum")
mean = op("mean")
std = op("std")
reduce_min = op("reduce_min")
reduce_max = op("reduce_max")


def _literal_string(value, name: str) -> str:
    if isinstance(value, String):
        return value.value
    if isinstance(value, str):
        return value
    raise TypeError(f"{name} must be a string, got {type(value).__name__}")


def _unit_microseconds(unit: str) -> float:
    unit = _literal_string(unit, "Datetime unit")
    unit_microseconds = {
        "ns": 0.001,
        "nanosecond": 0.001,
        "nanoseconds": 0.001,
        "us": 1.0,
        "microsecond": 1.0,
        "microseconds": 1.0,
        "ms": 1_000.0,
        "millisecond": 1_000.0,
        "milliseconds": 1_000.0,
        "s": 1_000_000.0,
        "sec": 1_000_000.0,
        "second": 1_000_000.0,
        "seconds": 1_000_000.0,
        "T": 60_000_000.0,
        "min": 60_000_000.0,
        "minute": 60_000_000.0,
        "minutes": 60_000_000.0,
        "H": 3_600_000_000.0,
        "h": 3_600_000_000.0,
        "hour": 3_600_000_000.0,
        "hours": 3_600_000_000.0,
        "D": 86_400_000_000.0,
        "d": 86_400_000_000.0,
        "day": 86_400_000_000.0,
        "days": 86_400_000_000.0,
    }
    try:
        return unit_microseconds[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported datetime unit {unit!r}") from exc


def _duration_microseconds(value: str | int | float, default_unit: str = "us") -> float:
    if isinstance(value, Number):
        return float(value.value) * _unit_microseconds(default_unit)
    if isinstance(value, (int, float)):
        return float(value) * _unit_microseconds(default_unit)
    text = _literal_string(value, "Duration").strip()
    if not text:
        raise ValueError("Duration cannot be empty")
    idx = 0
    while idx < len(text) and (text[idx].isdigit() or text[idx] in "+-."):
        idx += 1
    number = float(text[:idx]) if idx else 1.0
    return number * _unit_microseconds(text[idx:] or default_unit)


def _epoch_days(x: Expr) -> Expr:
    return floor(div(x, 86_400_000_000.0))


def _civil_parts(x: Expr) -> tuple[Expr, Expr, Expr, Expr]:
    z = add(_epoch_days(x), 719468.0)
    era = floor(div(where(lt(z, 0.0), sub(z, 146096.0), z), 146097.0))
    doe = sub(z, mul(era, 146097.0))
    yoe = floor(
        div(
            add(
                sub(doe, floor(div(doe, 1460.0))),
                sub(floor(div(doe, 36524.0)), floor(div(doe, 146096.0))),
            ),
            365.0,
        )
    )
    year_march = add(yoe, mul(era, 400.0))
    doy_march = sub(
        doe,
        add(sub(mul(365.0, yoe), floor(div(yoe, 100.0))), floor(div(yoe, 4.0))),
    )
    mp = floor(div(add(mul(5.0, doy_march), 2.0), 153.0))
    day_value = add(sub(doy_march, floor(div(add(mul(153.0, mp), 2.0), 5.0))), 1.0)
    month_value = add(mp, where(lt(mp, 10.0), 3.0, -9.0))
    year_value = add(year_march, where(lt(month_value, 3.0), 1.0, 0.0))
    return year_value, month_value, day_value, doy_march


def _is_leap_year_expr(year_expr: Expr) -> Expr:
    return and_(
        eq(mod(year_expr, 4.0), 0.0),
        or_(ne(mod(year_expr, 100.0), 0.0), eq(mod(year_expr, 400.0), 0.0)),
    )


@register_dsl_function("to_dt")
def to_dt(x: Expr, unit: str = "us") -> Expr:
    return mul(x, _unit_microseconds(unit))


@register_dsl_function("timeofday")
def timeofday(x: Expr) -> Expr:
    return mod(x, 86_400_000_000.0)


@register_dsl_function("hour")
def hour(x: Expr) -> Expr:
    return floor(div(timeofday(x), 3_600_000_000.0))


@register_dsl_function("minute")
def minute(x: Expr) -> Expr:
    return mod(floor(div(timeofday(x), 60_000_000.0)), 60.0)


@register_dsl_function("second")
def second(x: Expr) -> Expr:
    return mod(floor(div(timeofday(x), 1_000_000.0)), 60.0)


@register_dsl_function("year")
def year(x: Expr) -> Expr:
    year_value, _, _, _ = _civil_parts(x)
    return year_value


@register_dsl_function("month")
def month(x: Expr) -> Expr:
    _, month_value, _, _ = _civil_parts(x)
    return month_value


@register_dsl_function("day")
def day(x: Expr) -> Expr:
    _, _, day_value, _ = _civil_parts(x)
    return day_value


@register_dsl_function("dayofweek")
def dayofweek(x: Expr) -> Expr:
    return mod(add(_epoch_days(x), 3.0), 7.0)


@register_dsl_function("dayofyear")
def dayofyear(x: Expr) -> Expr:
    year_value, month_value, _, doy_march = _civil_parts(x)
    return where(
        gt(month_value, 2.0),
        add(add(doy_march, 60.0), _is_leap_year_expr(year_value)),
        sub(doy_march, 305.0),
    )


@register_dsl_function("shift")
def shift(x: Expr, nlag: int | float = 1) -> Expr:
    return shift(x, nlag, nlag)


@register_dsl_function("floor")
def floor(x: Expr, freq: str | int | float | None = None) -> Expr:
    micros = _duration_microseconds(freq)
    return mul(floor(div(x, micros)), micros)


@register_dsl_function("ceil")
def ceil(x: Expr, freq: str | int | float | None = None) -> Expr:
    micros = _duration_microseconds(freq)
    return mul(ceil(div(x, micros)), micros)


@register_dsl_function("round")
def round(x: Expr, *args, freq: str | int | float | None = None) -> Expr:
    if freq is None:
        if len(args) > 1:
            raise TypeError("round accepts at most one decimals argument")
        if not args:
            return call("round", x)
        factor = pow(10.0, args[0])
        return div(call("round", mul(x, factor)), factor)
    if args:
        raise TypeError("round cannot combine decimals with freq")
    micros = _duration_microseconds(freq)
    return mul(floor(add(div(x, micros), 0.5)), micros)


def InstrumentBasisMean(features, y=None, weights=None, hl=None) -> Expr:  # noqa: N802
    if y is None or hl is None:
        if weights is not None:
            raise TypeError("InstrumentBasisMean positional form cannot combine positional y/hl with keyword weights")
        return call("InstrumentBasisMean", features)
    if weights is None:
        return call("InstrumentBasisMean", features, y, 1.0, hl)
    return call("InstrumentBasisMean", features, y, weights, hl)


def Ridge(  # noqa: N802
    *features,
    y=None,
    weights=None,
    hl=None,
    lambda_=None,
    lam=None,
    nonneg=False,
    recompute_every=1,
) -> Expr:
    ridge_lambda = lambda_ if lambda_ is not None else lam
    recompute_kwargs = (
        {}
        if recompute_every == 1
        else {"recompute_every": recompute_every}
    )
    if y is None or hl is None or ridge_lambda is None:
        if weights is not None:
            raise TypeError(
                "Ridge positional form cannot combine positional "
                "y/hl/lambda with keyword weights"
            )
        return call(
            "Ridge",
            *features,
            3.0 if nonneg else 2.0,
            **recompute_kwargs,
        )
    if weights is None:
        return call(
            "Ridge",
            *features,
            y,
            1.0,
            hl,
            ridge_lambda,
            3.0 if nonneg else 2.0,
            **recompute_kwargs,
        )
    return call(
        "Ridge",
        *features,
        y,
        weights,
        hl,
        ridge_lambda,
        3.0 if nonneg else 2.0,
        **recompute_kwargs,
    )


get_beta = op("get_beta")
get_preds = op("get_preds")
get_residuals = op("get_residuals")
get_coefficient = op("get_coefficient")
get_sse = op("get_sse")
get_sst = op("get_sst")
get_r2 = op("get_r2")
get_residual_variance = op("get_residual_variance")
get_standard_errors = op("get_standard_errors")
get_standard_error = op("get_standard_error")
get_tstats = op("get_tstats")
get_tstat = op("get_tstat")
get_effective_df = op("get_effective_df")
get_effective_n = op("get_effective_n")
mean = op("mean")


@register_dsl_function("ratio")
def ratio(a: Expr, b: Expr) -> Expr:
    return div(a, b)


@register_dsl_function("diff")
def diff(x: Expr, nlag=1.0, max_size=1.0) -> Expr:
    return sub(x, shift(x, nlag, max_size))
