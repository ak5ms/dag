from __future__ import annotations

from dataclasses import dataclass
from math import prod


SUPPORTED_DTYPES: dict[str, str] = {
    "float32": "float",
    "float64": "double",
    "int32": "std::int32_t",
    "int64": "std::int64_t",
    "uint32": "std::uint32_t",
    "uint64": "std::uint64_t",
}


@dataclass(frozen=True, slots=True)
class InputTypeSpec:
    dtype: str
    row_width: int
    row_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        dtype = str(self.dtype).lower()
        if dtype not in SUPPORTED_DTYPES:
            raise TypeError(
                f"unsupported cpp_stream input dtype {dtype!r}; expected one of "
                f"{sorted(SUPPORTED_DTYPES)}"
            )
        row_width = int(self.row_width)
        if row_width <= 0:
            raise ValueError("input row_width must be > 0")
        if self.row_shape is None:
            row_shape = () if row_width == 1 else (row_width,)
        else:
            row_shape = tuple(int(extent) for extent in self.row_shape)
            if any(extent <= 0 for extent in row_shape):
                raise ValueError("input row_shape extents must be > 0")
            shape_width = prod(row_shape) if row_shape else 1
            if shape_width != row_width:
                raise ValueError(
                    f"row_shape {row_shape} has width {shape_width}, "
                    f"expected row_width={row_width}"
                )
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "row_width", row_width)
        object.__setattr__(self, "row_shape", row_shape)

    @property
    def cpp_type(self) -> str:
        return SUPPORTED_DTYPES[self.dtype]

    @property
    def row_scalar(self) -> bool:
        return self.row_shape == ()


__all__ = ["InputTypeSpec", "SUPPORTED_DTYPES"]
