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
    if isinstance(value, (int, float)):
        return Number(float(value))
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
    if isinstance(axis, (int, float)):
        return Number(float(axis))
    if isinstance(axis, (tuple, list)):
        if not axis:
            raise ValueError("axis cannot be empty")
        return KeyTuple(tuple(_axis_expr(item) for item in axis))
    raise TypeError("axis must be an int or a non-empty list/tuple of ints")


def reduction(name: str, x, *, axis=None, ddof=0) -> Expr:
    if name not in {"sum", "mean", "std"}:
        raise ValueError(f"unsupported reduction {name!r}")
    kwargs = []
    if axis is not None:
        kwargs.append(("axis", _axis_expr(axis)))
    if name == "std" and ddof != 0:
        kwargs.append(("ddof", ensure_expr(ddof)))
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
            "isnan",
            "purify",
            "fraction",
            "norm_inv",
            "xs_norm",
            "xs_rank",
            "get_beta",
            "get_preds",
            "xs_sort",
            "xstd",
            "outer",
            "cumsum",
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
        }
    },
    "where": _dsl_signature("condition", "true", "false"),
    "clip": _dsl_signature("x", "lo", "hi"),
    "round": _dsl_signature("x", "decimals"),
    "ewm": _dsl_signature("x", "span", "min_periods", "ignore_na", "adjust", defaults={"min_periods": 0, "ignore_na": True, "adjust": False}),
    "roll_mean": _dsl_signature("x", "lookback", "min_periods"),
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
    "sum": _dsl_signature("x", "axis", defaults={"axis": None}),
    "mean": _dsl_signature("x", "axis", defaults={"axis": None}),
    "std": _dsl_signature("x", "axis", "ddof", defaults={"axis": None, "ddof": 0}),
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
pow = op("pow")
cumsum = op("cumsum")
shift = op("shift")
buffer = op("buffer")
ewm = op("ewm")
roll_mean = op("roll_mean")
xs_rank = op("xs_rank")
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

cat = op("cat")
groupby = op("groupby")
sum = op("sum")
mean = op("mean")
std = op("std")


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
    while idx < len(text) and (text[idx].isdigit() or text[idx] in ".+-"):
        idx += 1
    if idx == 0 or idx == len(text):
        raise ValueError(f"Invalid duration {text!r}")
    return float(text[:idx]) * _unit_microseconds(text[idx:].strip())


def datetime_floor(ts, interval, unit="us") -> Expr:
    step = _duration_microseconds(interval, _literal_string(unit, "Datetime unit"))
    return floor(ensure_expr(ts) / step) * step


def datetime_ceil(ts, interval, unit="us") -> Expr:
    step = _duration_microseconds(interval, _literal_string(unit, "Datetime unit"))
    return ceil(ensure_expr(ts) / step) * step


def datetime_round(ts, interval, unit="us") -> Expr:
    step = _duration_microseconds(interval, _literal_string(unit, "Datetime unit"))
    return floor((ensure_expr(ts) + 0.5 * step) / step) * step


def calendar_feature(ts, field="weekday", unit="us") -> Expr:
    field = _literal_string(field, "Calendar field")
    unit = _literal_string(unit, "Datetime unit")
    micros = ensure_expr(ts) * _unit_microseconds(unit)
    day_us = _unit_microseconds("D")
    hour_us = _unit_microseconds("H")
    minute_us = _unit_microseconds("T")
    second_us = _unit_microseconds("s")
    day = floor(micros / day_us)
    time_us = micros - day * day_us
    fields = {
        "weekday": mod(day + 3.0, 7.0),
        "hour": floor(time_us / hour_us),
        "minute": floor(mod(time_us, hour_us) / minute_us),
        "second": floor(mod(time_us, minute_us) / second_us),
        "day": day,
    }
    try:
        return fields[field]
    except KeyError as exc:
        raise ValueError(f"Unsupported calendar field {field!r}") from exc


for _name, _fn in {
    "datetime_floor": datetime_floor,
    "datetime_ceil": datetime_ceil,
    "datetime_round": datetime_round,
    "calendar_feature": calendar_feature,
}.items():
    DEFAULT_DSL_REGISTRY.register(_name, _fn)


__all__ = [
    "DSLFunctionRegistry",
    "DEFAULT_DSL_REGISTRY",
    "ensure_expr",
    "var",
    "call",
    "op",
    "register_dsl_function",
    "get_dsl_op_signature",
    "univ",
    "grouped",
    "self_",
    "GROUPBY_VALUE_PLACEHOLDER",
]
