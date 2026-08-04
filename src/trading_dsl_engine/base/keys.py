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
    """Attach backend-neutral metadata to one dynamic group-key expression.

    ``expr``
        The expression whose value identifies the group. Wrapping an expression
        in ``Key`` does not otherwise change its mathematical semantics.

    ``num_keys``
        Number of consecutive non-NaN integer categories in the key's bounded
        domain. When supplied, the valid values are exactly
        ``offset, offset + 1, ..., offset + num_keys - 1``. A backend may use
        this finite domain to replace hashing with direct dense state indexing.
        NaN remains a separate additional category for floating-point keys.

        Examples: ``Key(month, num_keys=12, offset=1)`` describes months 1..12;
        ``Key(minute, num_keys=60)`` describes minute values 0..59.

    ``offset``
        First value in the bounded domain described by ``num_keys``. Dense
        routing maps a valid key value ``v`` to the zero-based digit
        ``v - offset``. ``offset`` has no effect unless ``num_keys`` is set.

    ``row_scalar``
        Whether the expression is lane invariant: one key value applies to all
        instruments in a row. ``True`` permits evaluating and resolving the key
        once per row and broadcasting the resulting group slot. ``False`` means
        each lane may have a different key. ``None`` asks the compiler to infer
        this from input shapes and expression dependencies. This is an assertion;
        incorrectly marking a lane-varying expression row-scalar changes results.

    ``dtype``
        Expected scalar value type of the completed key expression. Supported
        values are float32/float64/int32/int64/uint32/uint64. For direct mmap
        inputs the compiler verifies this against the file dtype. For derived
        expressions it verifies the inferred native result type. The hint never
        authorizes an implicit conversion of the input or expression.

    A tuple may contain independently described ``Key`` objects. If every dynamic
    key has ``num_keys``, a backend may use mixed-radix dense routing with capacity
    ``product(num_keys_i + 1)``; the extra digit preserves each floating key's NaN
    category. If any key is unbounded, the tuple is resolved by exact hashing.
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
    """Construct :class:`Key`; see ``Key`` for exact hint semantics."""
    return Key(
        expr=ensure_expr(expr),
        num_keys=num_keys,
        offset=offset,
        row_scalar=row_scalar,
        dtype=dtype,
    )


__all__ = ["Key", "key"]
