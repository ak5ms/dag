from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import jax

from trading_dsl_engine.base.dsl import ensure_expr
from trading_dsl_engine.base.parser import Expr

OutputKind = Literal["scalar", "vector", "matrix", "object"]


@dataclass(frozen=True, eq=False)
class StatelessJaxCall(Expr):
    """Expression node for user-supplied stateless JAX callables in jax_flat."""

    fn: Callable[..., jax.Array]
    args: tuple[Expr, ...]
    output_kind: OutputKind | None = None
    output_width: int | None = None
    name: str | None = None


@dataclass(frozen=True)
class StatelessJaxFunction:
    """Callable expression builder for stateless, variadic JAX functions."""

    fn: Callable[..., jax.Array]
    output_kind: OutputKind | None = None
    output_width: int | None = None
    name: str | None = None

    def __call__(self, *args) -> StatelessJaxCall:
        if not args:
            raise TypeError("stateless JAX functions expect at least one argument")
        return StatelessJaxCall(
            fn=self.fn,
            args=tuple(ensure_expr(arg) for arg in args),
            output_kind=self.output_kind,
            output_width=self.output_width,
            name=self.name or getattr(self.fn, "__name__", None),
        )


def stateless(
    fn: Callable[..., jax.Array] | None = None,
    *,
    output_kind: OutputKind | None = None,
    output_width: int | None = None,
    name: str | None = None,
):
    """Build a jax_flat Expr wrapper around a stateless JAX callable.

    The callable is executed inside the jax_flat tick and batch JIT paths and may
    accept any number of compiled child values. If output metadata is omitted,
    jax_flat infers the output kind and width from the first child, which is
    suitable for shape-preserving transforms such as ``jnp.flip``.
    """

    def wrap(target: Callable[..., jax.Array]) -> StatelessJaxFunction:
        return StatelessJaxFunction(
            fn=target,
            output_kind=output_kind,
            output_width=output_width,
            name=name or getattr(target, "__name__", None),
        )

    if fn is None:
        return wrap
    return wrap(fn)
