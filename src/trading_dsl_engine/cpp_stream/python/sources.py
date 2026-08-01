from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Mapping, Protocol, TypeAlias
from urllib.parse import unquote, urlparse

import numpy as np

from trading_dsl_engine.cpp_stream.python.npy import (
    InputTypeSpec,
    inspect_npy,
    mmap_npy,
)


SourceValue: TypeAlias = str | Path | np.ndarray | "InputSource"


@dataclass(frozen=True, slots=True)
class InputSource:
    """One formula input and optional source-specific metadata.

    The adapter is normally inferred independently for every input from the object
    type, URI scheme, or file extension. ``adapter`` is an escape hatch for formats
    whose location is ambiguous. Raw/headerless sources require ``input_type``.
    """

    value: object
    input_type: InputTypeSpec | None = None
    adapter: str | None = None
    options: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "options", MappingProxyType(dict(self.options)))


def source(
    value: object,
    *,
    input_type: InputTypeSpec | None = None,
    adapter: str | None = None,
    **options: object,
) -> InputSource:
    return InputSource(value, input_type=input_type, adapter=adapter, options=options)


@dataclass(frozen=True, slots=True)
class SourceInfo:
    adapter: str
    input_type: InputTypeSpec
    rows: int
    description: str


@dataclass(slots=True)
class PreparedSource:
    info: SourceInfo
    data_pointer: int
    owner: object
    _close: Callable[[], None]

    def close(self) -> None:
        self._close()


class SourceAdapter(Protocol):
    name: str

    def accepts(self, item: InputSource) -> bool: ...

    def inspect(
        self,
        item: InputSource,
        *,
        expected: InputTypeSpec | None,
    ) -> SourceInfo: ...

    def open(
        self,
        item: InputSource,
        *,
        expected: InputTypeSpec | None,
    ) -> PreparedSource: ...


_ADAPTERS: dict[str, SourceAdapter] = {}
_ADAPTER_ORDER: list[str] = []


def register_source_adapter(
    adapter: SourceAdapter,
    *,
    replace: bool = False,
    prepend: bool = False,
) -> None:
    name = str(adapter.name)
    if not name:
        raise ValueError("source adapter name must be nonempty")
    if name in _ADAPTERS and not replace:
        raise ValueError(f"source adapter {name!r} is already registered")
    if name in _ADAPTER_ORDER:
        _ADAPTER_ORDER.remove(name)
    _ADAPTERS[name] = adapter
    if prepend:
        _ADAPTER_ORDER.insert(0, name)
    else:
        _ADAPTER_ORDER.append(name)


def _as_source(value: SourceValue) -> InputSource:
    return value if isinstance(value, InputSource) else InputSource(value)


def _local_path(value: object) -> Path | None:
    if isinstance(value, Path):
        return value
    if not isinstance(value, str):
        return None
    parsed = urlparse(value)
    if parsed.scheme == "":
        return Path(value)
    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(f"unsupported non-local file URI {value!r}")
        return Path(unquote(parsed.path))
    return None


def _expected_type(
    item: InputSource,
    expected: InputTypeSpec | None,
) -> InputTypeSpec | None:
    declared = item.input_type
    if declared is not None and expected is not None and declared != expected:
        raise TypeError(
            f"source declares {declared}, but compilation expects {expected}"
        )
    return declared if declared is not None else expected


def _validate_actual(
    actual: InputTypeSpec,
    item: InputSource,
    expected: InputTypeSpec | None,
    description: str,
) -> InputTypeSpec:
    required = _expected_type(item, expected)
    if required is not None and actual != required:
        raise TypeError(
            f"source {description} has {actual}, but compilation expects {required}"
        )
    return actual


def _array_type(array: np.ndarray, description: str) -> tuple[InputTypeSpec, int]:
    dtype = np.dtype(array.dtype)
    if dtype.hasobject or dtype.fields is not None or dtype.subdtype is not None:
        raise TypeError(f"cpp_stream does not support dtype {dtype} for {description}")
    if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
        raise TypeError(f"cpp_stream requires native/little-endian data for {description}")
    if not array.flags.c_contiguous:
        raise ValueError(f"cpp_stream requires C-order data for {description}")
    shape = tuple(int(extent) for extent in array.shape)
    if not shape:
        raise ValueError(f"cpp_stream source {description} requires a row dimension")
    rows = shape[0]
    raw_row_shape = shape[1:]
    row_width = prod(raw_row_shape) if raw_row_shape else 1
    row_shape = () if row_width == 1 else raw_row_shape
    return InputTypeSpec(dtype.name, row_width, row_shape), rows


def _close_numpy_mapping(array: np.ndarray) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


class _NpyAdapter:
    name = "npy"

    def accepts(self, item: InputSource) -> bool:
        path = _local_path(item.value)
        return path is not None and path.suffix.lower() == ".npy"

    def inspect(
        self, item: InputSource, *, expected: InputTypeSpec | None
    ) -> SourceInfo:
        path = _local_path(item.value)
        assert path is not None
        info = inspect_npy(path)
        actual = _validate_actual(info.input_type, item, expected, str(path))
        return SourceInfo(self.name, actual, info.rows, str(path))

    def open(
        self, item: InputSource, *, expected: InputTypeSpec | None
    ) -> PreparedSource:
        path = _local_path(item.value)
        assert path is not None
        mapped = mmap_npy(path)
        try:
            actual = _validate_actual(
                mapped.info.input_type, item, expected, str(path)
            )
        except Exception:
            _close_numpy_mapping(mapped.array)
            raise
        info = SourceInfo(self.name, actual, mapped.info.rows, str(path))
        return PreparedSource(
            info=info,
            data_pointer=mapped.data_pointer,
            owner=mapped,
            _close=lambda: _close_numpy_mapping(mapped.array),
        )


class _RawAdapter:
    name = "raw"
    extensions = frozenset({".bin", ".raw"})

    def accepts(self, item: InputSource) -> bool:
        path = _local_path(item.value)
        return path is not None and path.suffix.lower() in self.extensions

    @staticmethod
    def _metadata(
        item: InputSource, expected: InputTypeSpec | None
    ) -> tuple[Path, InputTypeSpec, int]:
        path = _local_path(item.value)
        assert path is not None
        input_type = _expected_type(item, expected)
        if input_type is None:
            raise TypeError(
                f"raw source {path} has no header; supply input_types for this "
                "formula input or wrap it with source(..., input_type=...)"
            )
        row_bytes = input_type.row_width * np.dtype(input_type.dtype).itemsize
        size = path.stat().st_size
        if size % row_bytes:
            raise ValueError(
                f"raw source {path} has {size} bytes, not a multiple of "
                f"row size {row_bytes}"
            )
        return path, input_type, size // row_bytes

    def inspect(
        self, item: InputSource, *, expected: InputTypeSpec | None
    ) -> SourceInfo:
        path, input_type, rows = self._metadata(item, expected)
        return SourceInfo(self.name, input_type, rows, str(path))

    def open(
        self, item: InputSource, *, expected: InputTypeSpec | None
    ) -> PreparedSource:
        path, input_type, rows = self._metadata(item, expected)
        array = np.memmap(
            path,
            mode="r",
            dtype=np.dtype(input_type.dtype),
            shape=(rows, input_type.row_width),
            order="C",
        )
        info = SourceInfo(self.name, input_type, rows, str(path))
        return PreparedSource(
            info=info,
            data_pointer=int(array.ctypes.data),
            owner=array,
            _close=lambda: _close_numpy_mapping(array),
        )


class _ArrayAdapter:
    name = "array"

    def accepts(self, item: InputSource) -> bool:
        return isinstance(item.value, np.ndarray)

    def inspect(
        self, item: InputSource, *, expected: InputTypeSpec | None
    ) -> SourceInfo:
        array = item.value
        assert isinstance(array, np.ndarray)
        actual, rows = _array_type(array, "in-memory ndarray")
        actual = _validate_actual(actual, item, expected, "in-memory ndarray")
        return SourceInfo(self.name, actual, rows, "in-memory ndarray")

    def open(
        self, item: InputSource, *, expected: InputTypeSpec | None
    ) -> PreparedSource:
        array = item.value
        assert isinstance(array, np.ndarray)
        actual, rows = _array_type(array, "in-memory ndarray")
        actual = _validate_actual(actual, item, expected, "in-memory ndarray")
        info = SourceInfo(self.name, actual, rows, "in-memory ndarray")
        return PreparedSource(
            info=info,
            data_pointer=int(array.ctypes.data),
            owner=array,
            _close=lambda: None,
        )


def _adapter_for(item: InputSource) -> SourceAdapter:
    if item.adapter is not None:
        try:
            return _ADAPTERS[item.adapter]
        except KeyError as exc:
            raise ValueError(
                f"unknown source adapter {item.adapter!r}; registered adapters are "
                f"{sorted(_ADAPTERS)}"
            ) from exc
    for name in _ADAPTER_ORDER:
        adapter = _ADAPTERS[name]
        if adapter.accepts(item):
            return adapter
    value = item.value
    raise ValueError(
        f"cannot infer a cpp_stream source adapter for {value!r}; use a supported "
        "extension/URI, register an adapter, or pass source(..., adapter=...)"
    )


def inspect_source(
    value: SourceValue,
    *,
    expected: InputTypeSpec | None = None,
) -> SourceInfo:
    item = _as_source(value)
    return _adapter_for(item).inspect(item, expected=expected)


def open_source(
    value: SourceValue,
    *,
    expected: InputTypeSpec | None = None,
) -> PreparedSource:
    item = _as_source(value)
    return _adapter_for(item).open(item, expected=expected)


def inspect_source_mapping(
    data: Mapping[str, SourceValue],
    *,
    expected_types: Mapping[str, InputTypeSpec] | None = None,
) -> dict[str, SourceInfo]:
    expected_types = expected_types or {}
    return {
        name: inspect_source(value, expected=expected_types.get(name))
        for name, value in data.items()
    }


def open_source_mapping(
    data: Mapping[str, SourceValue],
    names: tuple[str, ...],
    expected_types: tuple[InputTypeSpec, ...],
) -> list[PreparedSource]:
    prepared: list[PreparedSource] = []
    try:
        for name, expected in zip(names, expected_types):
            prepared.append(open_source(data[name], expected=expected))
        return prepared
    except Exception:
        for item in reversed(prepared):
            item.close()
        raise


register_source_adapter(_ArrayAdapter())
register_source_adapter(_NpyAdapter())
register_source_adapter(_RawAdapter())


__all__ = [
    "InputSource",
    "InputTypeSpec",
    "PreparedSource",
    "SourceAdapter",
    "SourceInfo",
    "SourceValue",
    "inspect_source",
    "inspect_source_mapping",
    "open_source",
    "open_source_mapping",
    "register_source_adapter",
    "source",
]
