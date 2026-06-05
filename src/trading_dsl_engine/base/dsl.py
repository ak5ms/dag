from __future__ import annotations

from collections.abc import Callable, Sequence

from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, String, Universe, UniverseItem


class DSLFunctionRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, Callable[..., Expr]] = {}

    def register(self, name: str, fn: Callable[..., Expr], overwrite: bool = True) -> None:
        if not overwrite and name in self._fns:
            raise ValueError(f"DSL function already registered: {name}")
        self._fns[name] = fn

    def get(self, name: str) -> Callable[..., Expr] | None:
        return self._fns.get(name)


DEFAULT_DSL_REGISTRY = DSLFunctionRegistry()


def ensure_expr(value) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, tuple):
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


GROUPBY_VALUE_PLACEHOLDER = "self_"
self_ = var(GROUPBY_VALUE_PLACEHOLDER)


class GroupedExpr:
    def __init__(self, lhs, key) -> None:
        self.lhs = ensure_expr(lhs)
        key_expr = ensure_expr(key)
        if not isinstance(key_expr, KeyTuple):
            key_expr = KeyTuple((key_expr,))
        if sum(1 for item in key_expr.items if isinstance(item, Universe)) > 1:
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


def op(name: str) -> Callable[..., Expr]:
    def _op(*args, **kwargs) -> Expr:
        return call(name, *args, **kwargs)

    _op.__name__ = name
    return _op


def register_dsl_function(name: str | None = None, registry: DSLFunctionRegistry | None = None):
    target = registry or DEFAULT_DSL_REGISTRY

    def _decorator(fn: Callable[..., Expr]) -> Callable[..., Expr]:
        fn_name = name or fn.__name__

        def _wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            return ensure_expr(out)

        target.register(fn_name, _wrapped)
        return fn

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
segmented_cumsum = op("segmented_cumsum")
shift = op("shift")
buffer = op("buffer")
ewm = op("ewm")
xs_rank = op("xs_rank")
outer = op("outer")
bspline = op("bspline")
session_rbf_basis = op("session_rbf_basis")
future_session_rbf_basis_sum = op("future_session_rbf_basis_sum")
col = op("col")
einsum = op("einsum")

cat = op("cat")
groupby = op("groupby")


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


def _floor_expr(x: Expr) -> Expr:
    return call("floor", x)


def _ceil_expr(x: Expr) -> Expr:
    return call("ceil", x)


def _round_expr(x: Expr, *args) -> Expr:
    return call("round", x, *args)


def _epoch_days(x: Expr) -> Expr:
    return _floor_expr(div(x, 86_400_000_000.0))


def _civil_parts(x: Expr) -> tuple[Expr, Expr, Expr, Expr]:
    z = add(_epoch_days(x), 719468.0)
    era = _floor_expr(div(where(lt(z, 0.0), sub(z, 146096.0), z), 146097.0))
    doe = sub(z, mul(era, 146097.0))
    yoe = _floor_expr(
        div(
            add(
                sub(doe, _floor_expr(div(doe, 1460.0))),
                sub(_floor_expr(div(doe, 36524.0)), _floor_expr(div(doe, 146096.0))),
            ),
            365.0,
        )
    )
    year_march = add(yoe, mul(era, 400.0))
    doy_march = sub(
        doe,
        add(sub(mul(365.0, yoe), _floor_expr(div(yoe, 100.0))), _floor_expr(div(yoe, 4.0))),
    )
    mp = _floor_expr(div(add(mul(5.0, doy_march), 2.0), 153.0))
    day_value = add(sub(doy_march, _floor_expr(div(add(mul(153.0, mp), 2.0), 5.0))), 1.0)
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
    return _floor_expr(div(timeofday(x), 3_600_000_000.0))


@register_dsl_function("minute")
def minute(x: Expr) -> Expr:
    return mod(_floor_expr(div(timeofday(x), 60_000_000.0)), 60.0)


@register_dsl_function("second")
def second(x: Expr) -> Expr:
    return mod(_floor_expr(div(timeofday(x), 1_000_000.0)), 60.0)


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


@register_dsl_function("floor")
def floor(x: Expr, freq: str | int | float | None = None) -> Expr:
    if freq is None:
        return _floor_expr(x)
    micros = _duration_microseconds(freq)
    return mul(_floor_expr(div(x, micros)), micros)


@register_dsl_function("ceil")
def ceil(x: Expr, freq: str | int | float | None = None) -> Expr:
    if freq is None:
        return _ceil_expr(x)
    micros = _duration_microseconds(freq)
    return mul(_ceil_expr(div(x, micros)), micros)


@register_dsl_function("round")
def round(x: Expr, *args, freq: str | int | float | None = None) -> Expr:
    if freq is None:
        return _round_expr(x, *args)
    if args:
        raise TypeError("round cannot combine decimals with freq")
    micros = _duration_microseconds(freq)
    return mul(_floor_expr(add(div(x, micros), 0.5)), micros)


def InstrumentBasisMean(features, y=None, weights=None, hl=None) -> Expr:  # noqa: N802
    if y is None or hl is None:
        if weights is not None:
            raise TypeError("InstrumentBasisMean positional form cannot combine positional y/hl with keyword weights")
        return call("InstrumentBasisMean", features)
    if weights is None:
        return call("InstrumentBasisMean", features, y, 1.0, hl)
    return call("InstrumentBasisMean", features, y, weights, hl)


def Ridge(*features, y=None, weights=None, hl=None, lambda_=None, lam=None) -> Expr:  # noqa: N802
    ridge_lambda = lambda_ if lambda_ is not None else lam
    if y is None or hl is None or ridge_lambda is None:
        if weights is not None:
            raise TypeError("Ridge positional form cannot combine positional y/hl/lambda with keyword weights")
        return call("Ridge", *features)
    if weights is None:
        return call("Ridge", *features, y, 1.0, hl, ridge_lambda)
    return call("Ridge", *features, y, weights, hl, ridge_lambda)


get_beta = op("get_beta")
get_preds = op("get_preds")
rolling_quantile = op("rolling_quantile")
mean = op("mean")


@register_dsl_function("ratio")
def ratio(a: Expr, b: Expr) -> Expr:
    return div(a, b)


@register_dsl_function("diff")
def diff(x: Expr, nlag=1.0, max_size=1.0) -> Expr:
    return sub(x, shift(x, nlag, max_size))
