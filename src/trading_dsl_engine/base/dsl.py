from __future__ import annotations

from collections.abc import Callable, Sequence

from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, Universe, UniverseItem


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


def call(name: str, *args) -> Expr:
    return Call(name, tuple(ensure_expr(a) for a in args))


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
    def _op(*args) -> Expr:
        return call(name, *args)

    _op.__name__ = name
    return _op


def register_dsl_function(name: str | None = None, registry: DSLFunctionRegistry | None = None):
    target = registry or DEFAULT_DSL_REGISTRY

    def _decorator(fn: Callable[..., Expr]) -> Callable[..., Expr]:
        fn_name = name or fn.__name__

        def _wrapped(*args):
            out = fn(*args)
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
shift = op("shift")
ewm = op("ewm")
xs_rank = op("xs_rank")
outer = op("outer")
bspline = op("bspline")
col = op("col")
groupby = op("groupby")


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
