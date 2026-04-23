from __future__ import annotations

from collections.abc import Callable

from .parser import Call, Expr, Number


DSL_FUNCTIONS: dict[str, Callable[..., Expr]] = {}


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


def register_dsl_function(name: str | None = None):
    def _decorator(fn: Callable[..., Expr]) -> Callable[..., Expr]:
        fn_name = name or fn.__name__
        if fn_name in DSL_FUNCTIONS:
            raise ValueError(f"DSL function already registered: {fn_name}")

        def _wrapped(*args):
            out = fn(*args)
            return ensure_expr(out)

        DSL_FUNCTIONS[fn_name] = _wrapped
        return fn

    return _decorator


# Builtin operation constructors to support python-level composition.
add = op("add")
div = op("div")
ewm = op("ewm")
xs_rank = op("xs_rank")
outer = op("outer")


@register_dsl_function("ratio")
def ratio(a: Expr, b: Expr) -> Expr:
    return div(a, b)
