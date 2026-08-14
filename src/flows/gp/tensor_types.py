from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import ClassVar, Literal

from flows.gp.types import BoolRow, CountRow, DerivedNumericRow, DimensionlessRow, ExprValue, PriceRow, QuantityRow, StaticValue
from trading_dsl_engine.cpp_stream.python.source_types import InputTypeSpec

TensorSemantic = Literal["numeric", "derived", "dimensionless", "bool", "count", "price", "volume"]


class NumericTensor(ExprValue):
    tensor_rank: ClassVar[int]
    tensor_semantic: ClassVar[TensorSemantic]


_CLASSES: dict[tuple[int, TensorSemantic], type[NumericTensor]] = {}
_PARENTS = {
    "numeric": NumericTensor,
    "derived": "numeric",
    "dimensionless": "derived",
    "bool": "dimensionless",
    "count": "dimensionless",
    "price": "numeric",
    "volume": "numeric",
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
    prefix = {
        "numeric": "NumericTensor",
        "derived": "DerivedNumericTensor",
        "dimensionless": "DimensionlessTensor",
        "bool": "BoolTensor",
        "count": "CountTensor",
        "price": "BookPriceTensor",
        "volume": "BookVolumeTensor",
    }[semantic]
    result = type(prefix + str(rank), (parent,), {"tensor_rank": rank, "tensor_semantic": semantic})
    _CLASSES[key] = result
    return result


def tensor_rank(type_: type[NumericTensor]) -> int:
    return int(type_.tensor_rank)


def tensor_semantic(type_: type[NumericTensor]) -> TensorSemantic:
    return type_.tensor_semantic


def tensor_types_for_rank(rank: int) -> tuple[type[NumericTensor], ...]:
    return tuple(tensor_type(rank, semantic) for semantic in _PARENTS if semantic != "numeric")


def reduced_type(type_: type[NumericTensor], semantic: TensorSemantic | None = None):
    rank = tensor_rank(type_)
    semantic = semantic or tensor_semantic(type_)
    if rank > 2:
        return tensor_type(rank - 1, semantic)
    return {
        "numeric": DerivedNumericRow,
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
    """Tensor terminal, either external or composed from existing row columns."""

    name: str
    semantic: TensorSemantic
    feature_shape: tuple[int, ...]
    columns: tuple[str, ...] = ()
    dtype: str = "float64"

    def __post_init__(self) -> None:
        name = str(self.name).strip()
        shape = tuple(int(value) for value in self.feature_shape)
        columns = tuple(str(value) for value in self.columns)
        if not name or self.semantic not in _PARENTS or self.semantic == "numeric" or not shape or any(value <= 0 for value in shape):
            raise ValueError("invalid tensor field specification")
        if columns and (len(shape) != 1 or len(columns) != shape[0]):
            raise ValueError("composed tensor columns must equal its single feature extent")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "feature_shape", shape)
        object.__setattr__(self, "columns", columns)

    @property
    def logical_rank(self) -> int:
        return len(self.feature_shape) + 1

    @property
    def external(self) -> bool:
        return not self.columns

    def gp_type(self) -> type[NumericTensor]:
        return tensor_type(self.logical_rank, self.semantic)

    def input_type(self, n_instruments: int) -> InputTypeSpec:
        shape = (int(n_instruments), *self.feature_shape)
        return InputTypeSpec(dtype=self.dtype, row_width=prod(shape), row_shape=shape)


def _levels(prefixes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(f"{prefix}{level}_out0" for prefix in prefixes for level in range(10))


DEFAULT_TENSOR_FIELDS = (
    TensorFieldSpec("book_price", "price", (20,), _levels(("ap", "bp"))),
    TensorFieldSpec("book_volume", "volume", (20,), _levels(("volume_a", "volume_b"))),
)


def tensor_input_types(fields, n_instruments: int) -> dict[str, InputTypeSpec]:
    return {field.name: field.input_type(n_instruments) for field in fields if field.external}


NumericMatrix = tensor_type(2, "numeric")
DerivedNumericMatrix = tensor_type(2, "derived")
DimensionlessMatrix = tensor_type(2, "dimensionless")
BoolMatrix = tensor_type(2, "bool")
CountMatrix = tensor_type(2, "count")
BookPriceMatrix = tensor_type(2, "price")
BookVolumeMatrix = tensor_type(2, "volume")

__all__ = [name for name in globals() if not name.startswith("_")]
