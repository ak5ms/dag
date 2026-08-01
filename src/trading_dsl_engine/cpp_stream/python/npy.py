from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
from numpy.lib import format as npy_format


_SUPPORTED_DTYPES: dict[str, str] = {
    "float32": "float",
    "float64": "double",
    "int32": "std::int32_t",
    "int64": "std::int64_t",
    "uint32": "std::uint32_t",
    "uint64": "std::uint64_t",
}


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


@dataclass(slots=True)
class NpyMMap:
    info: NpyArrayInfo
    array: np.memmap

    @property
    def data_pointer(self) -> int:
        return int(self.array.ctypes.data)


def inspect_npy(path: str | Path) -> NpyArrayInfo:
    resolved = Path(path)
    with resolved.open("rb") as handle:
        version = npy_format.read_magic(handle)
        # NumPy's private helper is the single implementation used by the public
        # v1/v2 wrappers and correctly handles v3 UTF-8 headers as well.
        shape, fortran_order, dtype = npy_format._read_array_header(handle, version)  # type: ignore[attr-defined]
        data_offset = handle.tell()

    dtype = np.dtype(dtype)
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
    if fortran_order:
        raise ValueError("cpp_stream requires C-order .npy arrays")
    shape = tuple(int(value) for value in shape)
    if len(shape) == 1:
        rows, row_width = shape[0], 1
    elif len(shape) == 2:
        rows, row_width = shape
    else:
        raise ValueError(f"cpp_stream expects a 1D or 2D .npy array, got shape={shape}")
    if rows < 0 or row_width <= 0:
        raise ValueError(f"invalid .npy shape {shape}")
    expected_bytes = data_offset + rows * row_width * dtype.itemsize
    actual_bytes = resolved.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f".npy payload size mismatch for {resolved}: expected {expected_bytes} bytes, "
            f"found {actual_bytes}"
        )
    return NpyArrayInfo(
        path=resolved,
        dtype=dtype_name,
        shape=shape,
        fortran_order=False,
        data_offset=data_offset,
        rows=rows,
        row_width=row_width,
    )


def mmap_npy(path: str | Path, *, mode: str = "r") -> NpyMMap:
    info = inspect_npy(path)
    array = np.memmap(
        info.path,
        dtype=np.dtype(info.dtype),
        mode=mode,
        offset=info.data_offset,
        shape=info.shape,
        order="C",
    )
    return NpyMMap(info=info, array=array)


def inspect_npy_mapping(data: Mapping[str, str | Path]) -> dict[str, NpyArrayInfo]:
    return {name: inspect_npy(path) for name, path in data.items()}


__all__ = ["NpyArrayInfo", "NpyMMap", "inspect_npy", "inspect_npy_mapping", "mmap_npy"]
