from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from trading_dsl_engine.base.dsl import ensure_expr
from trading_dsl_engine.base.parser import Expr

OutputKind = Literal["scalar", "vector", "matrix", "object"]


@dataclass(frozen=True, eq=False)
class StatelessCall(Expr):
    """Backend-neutral expression for a named stateless callable.

    ``fn`` remains available to Python/JAX backends. Native backends select an
    implementation from ``cpp_name``; they never inspect or execute the Python
    callable in the row loop.
    """

    fn: Callable[..., Any]
    args: tuple[Expr, ...]
    output_kind: OutputKind | None = None
    output_width: int | None = None
    name: str | None = None
    cpp_name: str | None = None


@dataclass(frozen=True)
class StatelessFunction:
    fn: Callable[..., Any]
    output_kind: OutputKind | None = None
    output_width: int | None = None
    name: str | None = None
    cpp_name: str | None = None

    def __call__(self, *args) -> StatelessCall:
        if not args:
            raise TypeError("stateless functions expect at least one argument")
        return StatelessCall(
            fn=self.fn,
            args=tuple(ensure_expr(arg) for arg in args),
            output_kind=self.output_kind,
            output_width=self.output_width,
            name=self.name or getattr(self.fn, "__name__", None),
            cpp_name=self.cpp_name,
        )


__all__ = ["OutputKind", "StatelessCall", "StatelessFunction"]
