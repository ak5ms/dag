from __future__ import annotations

from pathlib import Path

import numpy as np

from trading_dsl_engine.cpp_stream import (
    InputSource,
    InputTypeSpec,
    PreparedSource,
    SourceInfo,
    compile_formula,
    register_source_adapter,
    source,
)


class _MemoryUriAdapter:
    name = "test-memory-uri"

    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self.arrays = arrays

    def accepts(self, item: InputSource) -> bool:
        return isinstance(item.value, str) and item.value.startswith("memtest://")

    def _metadata(
        self, item: InputSource, expected: InputTypeSpec | None
    ) -> tuple[np.ndarray, SourceInfo]:
        assert isinstance(item.value, str)
        array = self.arrays[item.value]
        row_shape = tuple(int(value) for value in array.shape[1:])
        row_width = int(np.prod(row_shape)) if row_shape else 1
        if row_width == 1:
            row_shape = ()
        actual = InputTypeSpec(array.dtype.name, row_width, row_shape)
        declared = item.input_type
        required = declared if declared is not None else expected
        if required is not None and actual != required:
            raise TypeError(f"memory source has {actual}, expected {required}")
        return array, SourceInfo(self.name, actual, array.shape[0], item.value)

    def inspect(
        self, item: InputSource, *, expected: InputTypeSpec | None
    ) -> SourceInfo:
        _, info = self._metadata(item, expected)
        return info

    def open(
        self, item: InputSource, *, expected: InputTypeSpec | None
    ) -> PreparedSource:
        array, info = self._metadata(item, expected)
        return PreparedSource(info, int(array.ctypes.data), array, lambda: None)


def test_custom_uri_and_npy_sources_share_one_compile_and_run_path(tmp_path: Path) -> None:
    rows, n = 23, 5
    left = np.arange(rows * n, dtype=np.float64).reshape(rows, n)
    right = 1000.0 + left
    right_path = tmp_path / "right.npy"
    np.save(right_path, right)

    uri = "memtest://left"
    register_source_adapter(
        _MemoryUriAdapter({uri: left}),
        replace=True,
        prepend=True,
    )
    runtime = compile_formula(
        "left + right",
        {"left": uri, "right": right_path},
        n_instruments=n,
    )
    output = tmp_path / "mixed_uri.bin"
    runtime.run(out_path=output)
    actual = np.asarray(
        np.memmap(output, mode="r", dtype=np.float64, shape=(rows, n))
    )
    np.testing.assert_array_equal(actual, left + right)

    generated = runtime.generated_cpp.read_text()
    assert "cpp_stream_run_arrays" in generated
    assert "cpp_stream_run_files" not in generated
    assert "memtest" not in generated
    assert ".npy" not in generated


def test_runtime_can_replace_bound_npy_with_compatible_raw_source(tmp_path: Path) -> None:
    rows, n = 13, 3
    original = np.arange(rows * n, dtype=np.float64).reshape(rows, n)
    replacement = 10.0 + original
    original_path = tmp_path / "x.npy"
    replacement_path = tmp_path / "x.bin"
    np.save(original_path, original)
    replacement.tofile(replacement_path)

    runtime = compile_formula("x * 2", {"x": original_path}, n_instruments=n)
    output = tmp_path / "replacement.bin"
    runtime.run(
        {
            "x": source(
                replacement_path,
                input_type=InputTypeSpec("float64", n),
            )
        },
        out_path=output,
    )
    actual = np.asarray(
        np.memmap(output, mode="r", dtype=np.float64, shape=(rows, n))
    )
    np.testing.assert_array_equal(actual, replacement * 2.0)
