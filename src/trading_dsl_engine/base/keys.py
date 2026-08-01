from __future__ import annotations

from dataclasses import dataclass

from trading_dsl_engine.base.dsl import ensure_expr
from trading_dsl_engine.base.parser import Expr


_SUPPORTED_DTYPES = {
    "float32",
    "float64",
    "int32",
    "int64",
    "uint32",
    "uint64",
}


@dataclass(frozen=True, eq=False)
class Key(Expr):
    """A group-key expression plus backend-neutral physical hints.

    The hints are assertions supplied by the caller. Backends may use them to
    select a dense resolver, evaluate a lane-invariant expression once per row,
    and preserve integer/categorical typing. They do not change key semantics.

    ``num_keys`` describes the number of consecutive valid integer values
    beginning at ``offset``. NaN remains a valid additional category.
    """

    expr: Expr
    num_keys: int | None = None
    offset: int = 0
    row_scalar: bool | None = None
    dtype: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "expr", ensure_expr(self.expr))
        if self.num_keys is not None:
            num_keys = int(self.num_keys)
            if num_keys <= 0:
                raise ValueError("Key.num_keys must be > 0")
            object.__setattr__(self, "num_keys", num_keys)
        object.__setattr__(self, "offset", int(self.offset))
        if self.row_scalar is not None:
            object.__setattr__(self, "row_scalar", bool(self.row_scalar))
        if self.dtype is not None:
            dtype = str(self.dtype).lower()
            if dtype not in _SUPPORTED_DTYPES:
                raise ValueError(
                    f"unsupported Key.dtype {self.dtype!r}; expected one of "
                    f"{sorted(_SUPPORTED_DTYPES)}"
                )
            object.__setattr__(self, "dtype", dtype)


def key(
    expr,
    *,
    num_keys: int | None = None,
    offset: int = 0,
    row_scalar: bool | None = None,
    dtype: str | None = None,
) -> Key:
    return Key(
        expr=ensure_expr(expr),
        num_keys=num_keys,
        offset=offset,
        row_scalar=row_scalar,
        dtype=dtype,
    )


__all__ = ["Key", "key"]
