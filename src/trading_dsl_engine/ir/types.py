from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ValueKind = Literal["scalar", "vector", "matrix", "fixed", "object"]


@dataclass(frozen=True, slots=True)
class ValueType:
    """Backend-neutral value shape for one formula timestep.

    ``scalar``
        One value for the complete timestep.

    ``vector``
        One value per instrument. ``width`` is always one.

    ``matrix``
        One fixed-width feature row per instrument. A value with width ``K`` has
        logical shape ``(n_instruments, K)`` at each timestep.

    ``fixed``
        A fixed-width value not indexed by instrument. Ridge coefficients use
        this shape outside groupby.

    ``object``
        A structured intermediate that must be projected before file output.
        ``width`` carries the compile-time coefficient width for Ridge values.
    """

    kind: ValueKind
    width: int = 1
    dtype: str = "float64"

    def __post_init__(self) -> None:
        if int(self.width) <= 0:
            raise ValueError("ValueType.width must be > 0")
        if self.kind in {"scalar", "vector"} and self.width != 1:
            raise ValueError(f"{self.kind} values must have width=1")


def matrix(width: int, dtype: str = "float64") -> ValueType:
    return ValueType("matrix", int(width), dtype)


def fixed(width: int, dtype: str = "float64") -> ValueType:
    return ValueType("fixed", int(width), dtype)


def object_value(width: int, dtype: str = "float64") -> ValueType:
    return ValueType("object", int(width), dtype)


SCALAR = ValueType("scalar")
VECTOR = ValueType("vector")


__all__ = [
    "ValueKind",
    "ValueType",
    "SCALAR",
    "VECTOR",
    "matrix",
    "fixed",
    "object_value",
]
