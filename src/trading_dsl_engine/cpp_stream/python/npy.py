from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np


_SUPPORTED_DTYPES: dict[str, str] = {
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

    def __post_init__(self) -> None:
        dtype = str(self.dtype).lower()
        if dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"unsupported cpp_stream input dtype {dtype!r}; expected one of "
                f"{sorted(_SUPPORTED_DTYPES)}"
            )
        if int(self.row_width) <= 0:
            raise ValueError("input row_width must be > 0")
        object.__setattr__(self, "dtype", dtype)
        object.__setattr__(self, "row_width", int(self.row_width))

    @property
    def cpp_type(self) -> str:
        return _SUPPORTED_DTYPES[self.dtype]

    @property
    def row_scalar(self) -> bool:
        return self.row_width == 1


@dataclass(frozen=True, slots=True)
class NpyArrayInfo:
    path: Path
    dtype: str
    shape: tuple[int, ...]
    fortran_order: bool
    data_offset: int
    rows: int
    row_width: int

    @property
    def row_scalar(self) -> bool:
        return self.row_width == 1

    @property
    def cpp_type(self) -> str:
        return _SUPPORTED_DTYPES[self.dtype]

    @property
    def input_type(self) -> InputTypeSpec:
        return InputTypeSpec(dtype=self.dtype, row_width=self.row_width)


@dataclass(slots=True)
class NpyMMap:
    info: NpyArrayInfo
    array: np.memmap

    @property
    def data_pointer(self) -> int:
        return int(self.array.ctypes.data)


def _open_public_memmap(path: Path, mode: str) -> np.memmap:
    array = np.load(path, mmap_mode=mode, allow_pickle=False)
    if not isinstance(array, np.memmap):
        raise TypeError(f"expected a .npy memory map for {path}")
    return array


def _info_from_memmap(path: Path, array: np.memmap) -> NpyArrayInfo:
    dtype = np.dtype(array.dtype)
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise TypeError(f"cpp_stream does not support object/structured .npy dtype {dtype}")
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        raise TypeError("cpp_stream currently requires native/little-endian .npy data")
    dtype_name = dtype.name
    if dtype_name not in _SUPPORTED_DTYPES:
        raise TypeError(
            f"unsupported cpp_stream .npy dtype {dtype_name!r}; expected one of "
            f"{sorted(_SUPPORTED_DTYPES)}"
        )
    if not array.flags.c_contiguous:
        raise ValueError("cpp_stream requires C-order .npy arrays")
    shape = tuple(int(value) for value in array.shape)
    if len(shape) == 1:
        rows, row_width = shape[0], 1
    elif len(shape) == 2:
        rows, row_width = shape
    else:
        raise ValueError(f"cpp_stream expects a 1D or 2D .npy array, got shape={shape}")
    if rows < 0 or row_width <= 0:
        raise ValueError(f"invalid .npy shape {shape}")
    data_offset = int(array.offset)
    expected_bytes = data_offset + rows * row_width * dtype.itemsize
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f".npy payload size mismatch for {path}: expected {expected_bytes} bytes, "
            f"found {actual_bytes}"
        )
    return NpyArrayInfo(
        path=path,
        dtype=dtype_name,
        shape=shape,
        fortran_order=False,
        data_offset=data_offset,
        rows=rows,
        row_width=row_width,
    )


def inspect_npy(path: str | Path) -> NpyArrayInfo:
    resolved = Path(path)
    array = _open_public_memmap(resolved, "r")
    try:
        return _info_from_memmap(resolved, array)
    finally:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()


def mmap_npy(path: str | Path, *, mode: str = "r") -> NpyMMap:
    resolved = Path(path)
    array = _open_public_memmap(resolved, mode)
    try:
        info = _info_from_memmap(resolved, array)
    except Exception:
        mapping = getattr(array, "_mmap", None)
        if mapping is not None:
            mapping.close()
        raise
    return NpyMMap(info=info, array=array)


def inspect_npy_mapping(data: Mapping[str, str | Path]) -> dict[str, NpyArrayInfo]:
    return {name: inspect_npy(path) for name, path in data.items()}


__all__ = [
    "InputTypeSpec",
    "NpyArrayInfo",
    "NpyMMap",
    "inspect_npy",
    "inspect_npy_mapping",
    "mmap_npy",
]
