from __future__ import annotations

from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.parser import Expr


def broadcast_like(x: Expr, value: Expr) -> Expr:
    """Broadcast a row-scalar expression back across the lanes of ``x``.

    ``where`` broadcasts scalar branches against a row condition, avoiding an
    artificial dimension-changing node in the GP grammar while keeping the
    underlying DSL reduction explicit.
    """

    condition = dsl.isfinite(x)
    return dsl.where(condition, value, value)


def broadcast_reduction(
    name: str,
    x: Expr,
    *,
    axis: int | tuple[int, ...],
    ignore_na: bool = True,
    ddof: int | None = None,
) -> Expr:
    """Apply a non-temporal DSL reduction and broadcast it back to a row."""

    axes = axis if isinstance(axis, tuple) else (axis,)
    if not axes or any(int(value) <= 0 for value in axes):
        raise ValueError("GP reductions must use only axes > 0")
    kwargs: dict[str, object] = {"axis": axis, "ignore_na": ignore_na}
    if ddof is not None:
        kwargs["ddof"] = ddof
    reduced = dsl.call(name, x, **kwargs)
    return broadcast_like(x, reduced)


def broadcast_xs_unary(name: str, x: Expr, *args) -> Expr:
    """Broadcast a genuinely dimension-reducing unary DSL op back to the row."""

    return broadcast_like(x, dsl.call(name, x, *args))


__all__ = [
    "broadcast_like",
    "broadcast_reduction",
    "broadcast_xs_unary",
]
