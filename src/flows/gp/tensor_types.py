from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import ClassVar, Literal

from flows.gp.types import BoolRow, CountRow, DerivedNumericRow, DimensionlessRow, ExprValue, PriceRow, QuantityRow, StaticValue
from trading_dsl_engine.cpp_stream.python.source_types import InputTypeSpec

TensorSemantic = Literal["derived", "dimensionless", "bool", "count", "price", "volume"]


class NumericTensor(ExprValue):
    tensor_rank: ClassVar[int]
    tensor_semantic: ClassVar[TensorSemantic]


_CLASSES: dict[tuple[int, TensorSemantic], type[NumericTensor]] = {}
_PARENTS = {
    "derived": NumericTensor,
    "dimensionless": "derived",
    "bool": "dimensionless",
    "count": "dimensionless",
    "price": NumericTensor,
    "volume": NumericTensor,
}


def tensor_type(rank: int, semantic: TensorSemantic) -> type[NumericTensor]:
    rank = int(rank)
    if rank < 2:
        raise ValueError("tensor rank must be >= 2")
    key = (rank, semantic)
    if key in _CLASSES:
        return _CLASSES[key]
    parent = _PARENTS.get(semantic)
    if parent is None:
        raise ValueError(f"unknown tensor semantic {semantic!r}")
    if isinstance(parent, str):
        parent = tensor_type(rank, parent)
    name = {
        "derived": "DerivedNumericTensor",
        "dimensionless": "DimensionlessTensor",
        "bool": "BoolTensor",
        "count": "CountTensor",
        "price": "BookPriceTensor",
        "volume": "BookVolumeTensor",
    }[semantic] + str(rank)
    result = type(name, (parent,), {"tensor_rank": rank, "tensor_semantic": semantic})
    _CLASSES[key] = result
    return result


def tensor_rank(type_: type[NumericTensor]) -> int:
    return int(type_.tensor_rank)


def tensor_semantic(type_: type[NumericTensor]) -> TensorSemantic:
    return type_.tensor_semantic


def tensor_types_for_rank(rank: int) -> tuple[type[NumericTensor], ...]:
    return tuple(tensor_type(rank, semantic) for semantic in _PARENTS)


def reduced_type(type_: type[NumericTensor], semantic: TensorSemantic | None = None):
    rank = tensor_rank(type_)
    semantic = semantic or tensor_semantic(type_)
    if rank > 2:
        return tensor_type(rank - 1, semantic)
    return {
        "derived": DerivedNumericRow,
        "dimensionless": DimensionlessRow,
        "bool": BoolRow,
        "count": CountRow,
        "price": PriceRow,
        "volume": QuantityRow,
    }[semantic]


@dataclass(frozen=True)
class TensorIndex(StaticValue):
    value: int

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or int(self.value) != self.value or self.value < 0:
            raise ValueError("TensorIndex must be a nonnegative integer")
        object.__setattr__(self, "value", int(self.value))


@dataclass(frozen=True)
class TensorFieldSpec:
    """Source stored as (rows, instruments, *feature_shape)."""

    name: str
    semantic: TensorSemantic
    feature_shape: tuple[int, ...] = (10,)
    dtype: str = "float64"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        shape = tuple(int(value) for value in self.feature_shape)
        if not name or self.semantic not in _PARENTS or not shape or any(value <= 0 for value in shape):
            raise ValueError("invalid tensor field specification")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "feature_shape", shape)

    @property
    def logical_rank(self) -> int:
        return len(self.feature_shape) + 1

    def gp_type(self) -> type[NumericTensor]:
        return tensor_type(self.logical_rank, self.semantic)

    def input_type(self, n_instruments: int) -> InputTypeSpec:
        shape = (int(n_instruments), *self.feature_shape)
        return InputTypeSpec(dtype=self.dtype, row_width=prod(shape), row_shape=shape)


DEFAULT_TENSOR_FIELDS = (
    TensorFieldSpec("book_price", "price"),
    TensorFieldSpec("book_volume", "volume"),
)


def tensor_input_types(fields, n_instruments: int) -> dict[str, InputTypeSpec]:
    return {field.name: field.input_type(n_instruments) for field in fields}


DerivedNumericMatrix = tensor_type(2, "derived")
DimensionlessMatrix = tensor_type(2, "dimensionless")
BoolMatrix = tensor_type(2, "bool")
CountMatrix = tensor_type(2, "count")
BookPriceMatrix = tensor_type(2, "price")
BookVolumeMatrix = tensor_type(2, "volume")

__all__ = [name for name in globals() if not name.startswith("_")]
