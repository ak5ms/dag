from __future__ import annotations

from collections.abc import Callable

from trading_dsl_engine.parser import Call, Expr, Number


class DSLFunctionRegistry:
    def __init__(self) -> None:
        self._fns: dict[str, Callable[..., Expr]] = {}

    def register(self, name: str, fn: Callable[..., Expr]) -> None:
        if name in self._fns:
            raise ValueError(f"DSL function already registered: {name}")
        self._fns[name] = fn

    def get(self, name: str) -> Callable[..., Expr] | None:
        return self._fns.get(name)


DEFAULT_DSL_REGISTRY = DSLFunctionRegistry()


def ensure_expr(value) -> Expr:
    if isinstance(value, Expr):
        return value
    if isinstance(value, (int, float)):
        return Number(float(value))
    raise TypeError(f"Expected Expr|int|float, got {type(value).__name__}")


def call(name: str, *args) -> Expr:
    return Call(name, tuple(ensure_expr(a) for a in args))


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
div = op("div")
ewm = op("ewm")
xs_rank = op("xs_rank")
outer = op("outer")


@register_dsl_function("ratio")
def ratio(a: Expr, b: Expr) -> Expr:
    return div(a, b)
