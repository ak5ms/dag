from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal


ValueKind = Literal["scalar", "vector", "matrix", "fixed", "tensor", "object"]
Dimension = int | None


@dataclass(frozen=True, slots=True)
class ValueType:
    """Backend-neutral value shape for one formula timestep.

    ``None`` in ``shape`` denotes the instrument dimension. Existing scalar,
    vector, matrix, and fixed-width kinds retain their compatibility fields;
    arbitrary einsum outputs use ``tensor`` with an explicit logical shape.
    """

    kind: ValueKind
    width: int = 1
    dtype: str = "float64"
    shape: tuple[Dimension, ...] | None = None

    def __post_init__(self) -> None:
        if int(self.width) <= 0:
            raise ValueError("ValueType.width must be > 0")
        if self.kind in {"scalar", "vector"} and self.width != 1:
            raise ValueError(f"{self.kind} values must have width=1")

        shape = self.shape
        if shape is None:
            if self.kind == "scalar":
                shape = ()
            elif self.kind == "vector":
                shape = (None,)
            elif self.kind == "matrix":
                shape = (None, int(self.width))
            elif self.kind == "fixed":
                shape = (int(self.width),)
            elif self.kind == "tensor":
                raise ValueError("tensor ValueType requires an explicit shape")
        if shape is not None:
            normalized = tuple(shape)
            for extent in normalized:
                if extent is not None and (not isinstance(extent, int) or extent < 0):
                    raise ValueError(f"invalid ValueType extent {extent!r}")
            object.__setattr__(self, "shape", normalized)

    @property
    def logical_shape(self) -> tuple[Dimension, ...]:
        if self.shape is None:
            raise ValueError(f"{self.kind} values do not have a tensor shape")
        return self.shape


def matrix(width: int, dtype: str = "float64") -> ValueType:
    width = int(width)
    return ValueType("matrix", width, dtype, (None, width))


def fixed(width: int, dtype: str = "float64") -> ValueType:
    width = int(width)
    return ValueType("fixed", width, dtype, (width,))


def tensor(shape: tuple[Dimension, ...] | list[Dimension], dtype: str = "float64") -> ValueType:
    normalized = tuple(shape)
    if normalized == ():
        return ValueType("scalar", 1, dtype, ())
    if normalized == (None,):
        return ValueType("vector", 1, dtype, normalized)
    if len(normalized) == 2 and normalized[0] is None and isinstance(normalized[1], int):
        return matrix(normalized[1], dtype)
    if len(normalized) == 1 and isinstance(normalized[0], int):
        return fixed(normalized[0], dtype)
    fixed_product = prod(extent for extent in normalized if extent is not None)
    return ValueType("tensor", max(1, int(fixed_product)), dtype, normalized)


def object_value(width: int, dtype: str = "float64") -> ValueType:
    return ValueType("object", int(width), dtype, None)


def resolve_shape(value_type: ValueType, n_instruments: int) -> tuple[int, ...]:
    if n_instruments <= 0:
        raise ValueError("n_instruments must be > 0")
    return tuple(
        n_instruments if extent is None else int(extent)
        for extent in value_type.logical_shape
    )


def shape_size(shape: tuple[int, ...] | list[int]) -> int:
    return prod(shape) if shape else 1


SCALAR = ValueType("scalar", shape=())
VECTOR = ValueType("vector", shape=(None,))


__all__ = [
    "ValueKind",
    "Dimension",
    "ValueType",
    "SCALAR",
    "VECTOR",
    "matrix",
    "fixed",
    "tensor",
    "object_value",
    "resolve_shape",
    "shape_size",
]
