from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax

from trading_dsl_engine.base.custom import (
    OutputKind,
    StatelessCall,
    StatelessFunction,
)
from trading_dsl_engine.base.dsl import ensure_expr
from trading_dsl_engine.base.parser import Expr

# Backward-compatible public names. The expression itself is backend-neutral so
# cpp_stream can lower a named native implementation without importing JAX types
# into the shared IR package.
StatelessJaxCall = StatelessCall
StatelessJaxFunction = StatelessFunction


@dataclass(frozen=True, eq=False)
class RollingJaxCall(Expr):
    """Expression node for experimental rolling-window JAX callables."""

    fn: Callable[[jax.Array], jax.Array]
    args: tuple[Expr, ...]
    lookback: int
    min_periods: int
    output_kind: OutputKind | None = None
    output_width: int | None = None
    name: str | None = None


def rolling(
    x,
    lookback: int,
    min_periods: int | None,
    fn: Callable[[jax.Array], jax.Array],
    *,
    output_kind: OutputKind | None = None,
    output_width: int | None = None,
    name: str | None = None,
) -> RollingJaxCall:
    lookback_i = int(lookback)
    min_periods_i = lookback_i if min_periods is None else int(min_periods)
    if lookback_i <= 0 or min_periods_i <= 0 or min_periods_i > lookback_i:
        raise ValueError("rolling expects 0 < min_periods <= lookback")
    return RollingJaxCall(
        fn=fn,
        args=(ensure_expr(x),),
        lookback=lookback_i,
        min_periods=min_periods_i,
        output_kind=output_kind,
        output_width=output_width,
        name=name or getattr(fn, "__name__", None),
    )


def stateless(
    fn: Callable[..., jax.Array] | None = None,
    *,
    output_kind: OutputKind | None = None,
    output_width: int | None = None,
    name: str | None = None,
    cpp_name: str | None = None,
):
    """Build a backend-neutral named stateless expression.

    JAX backends execute ``fn``. Native backends use ``cpp_name`` and explicit
    output metadata to select a compiled policy.
    """

    def wrap(target: Callable[..., jax.Array]) -> StatelessFunction:
        return StatelessFunction(
            fn=target,
            output_kind=output_kind,
            output_width=output_width,
            name=name or getattr(target, "__name__", None),
            cpp_name=cpp_name,
        )

    if fn is None:
        return wrap
    return wrap(fn)


__all__ = [
    "OutputKind",
    "StatelessJaxCall",
    "StatelessJaxFunction",
    "RollingJaxCall",
    "rolling",
    "stateless",
]
