from __future__ import annotations

from pathlib import Path

import numpy as np

from trading_dsl_engine.base.dsl import cumsum, ewm, groupby, self_, univ, var
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.cpp_stream import compile_npy_formula, inspect_npy, mmap_npy


def test_npy_reader_exposes_dtype_shape_and_row_scalar(tmp_path: Path):
    path = tmp_path / "timestamp.npy"
    values = np.arange(10, dtype=np.int64)
    np.save(path, values)

    info = inspect_npy(path)
    assert info.dtype == "int64"
    assert info.shape == (10,)
    assert info.rows == 10
    assert info.row_width == 1
    assert info.row_scalar is True
    assert info.cpp_type == "std::int64_t"

    mapped = mmap_npy(path)
    try:
        assert mapped.data_pointer == mapped.array.ctypes.data
        np.testing.assert_array_equal(mapped.array, values)
    finally:
        mapped.array._mmap.close()


def test_typed_row_scalar_npy_input_broadcasts_without_copy(tmp_path: Path):
    rows, n = 32, 9
    timestamp = np.arange(rows, dtype=np.int64) + 1_700_000_000_000_000
    close = np.arange(rows * n, dtype=np.float64).reshape(rows, n) / 10.0
    timestamp_path = tmp_path / "_ev_ts.npy"
    close_path = tmp_path / "close.npy"
    np.save(timestamp_path, timestamp)
    np.save(close_path, close)

    runtime = compile_npy_formula(
        "close + _ev_ts",
        {"close": close_path, "_ev_ts": timestamp_path},
        n_instruments=n,
    )
    assert runtime.input_types[0].dtype == "float64"
    assert runtime.input_types[0].row_width == n
    assert runtime.input_types[1].dtype == "int64"
    assert runtime.input_types[1].row_width == 1

    out_path = tmp_path / "out.bin"
    result = runtime.run_npy_files(
        {"close": close_path, "_ev_ts": timestamp_path},
        out_path=out_path,
    )
    assert result.rows == rows
    actual = np.memmap(out_path, mode="r", dtype=np.float64, shape=(rows, n))
    expected = close + timestamp[:, None]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_key_hints_drive_dense_row_scalar_minute_groupby_on_npy(tmp_path: Path):
    rows, n = 180, 9
    minute_us = 60_000_000
    day_us = 86_400_000_000
    base = (1_700_000_000_000_000 // day_us) * day_us
    timestamp = base + np.arange(rows, dtype=np.int64) * minute_us
    rng = np.random.default_rng(42)
    close = rng.normal(100.0, 1.0, size=(rows, n)).astype(np.float64)
    timestamp_path = tmp_path / "_ev_ts.npy"
    close_path = tmp_path / "close.npy"
    np.save(timestamp_path, timestamp)
    np.save(close_path, close)

    formula = groupby(
        (
            univ([0], [1, 2], list(range(3, 9))),
            Key(var("minute"), num_keys=60, row_scalar=True, dtype="int64"),
        ),
        var("close"),
        ewm(cumsum(self_), 3),
    )
    runtime = compile_npy_formula(
        formula,
        {"_ev_ts": timestamp_path, "close": close_path},
        n_instruments=n,
    )
    generated = runtime.generated_cpp.read_text()
    assert "DenseTupleGroupResolver" in generated
    assert "InputSrc<0, std::int64_t, 1>" in generated

    out_path = tmp_path / "grouped.bin"
    runtime.run_npy_files(
        {"_ev_ts": timestamp_path, "close": close_path},
        out_path=out_path,
    )
    actual = np.asarray(
        np.memmap(out_path, mode="r", dtype=np.float64, shape=(rows, n))
    )

    expected = np.empty_like(close)
    cumsum_state = np.zeros((60, n), dtype=np.float64)
    ewm_state = np.zeros((60, n), dtype=np.float64)
    initialized = np.zeros((60, n), dtype=bool)
    for row in range(rows):
        minute = int(((timestamp[row] % day_us) // minute_us) % 60)
        cumsum_state[minute] += close[row]
        for lane in range(n):
            value = cumsum_state[minute, lane]
            if initialized[minute, lane]:
                ewm_state[minute, lane] = 0.5 * value + 0.5 * ewm_state[minute, lane]
            else:
                ewm_state[minute, lane] = value
                initialized[minute, lane] = True
            expected[row, lane] = ewm_state[minute, lane]

    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-12)
