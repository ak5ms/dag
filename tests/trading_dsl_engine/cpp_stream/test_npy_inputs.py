from __future__ import annotations

from pathlib import Path

import numpy as np

from trading_dsl_engine.base.dsl import cumsum, ewm, groupby, self_, univ, var
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.cpp_stream import (
    InputTypeSpec,
    compile_formula,
    inspect_source,
    open_source,
    source,
)


def test_source_inference_exposes_npy_dtype_shape_and_row_scalar(tmp_path: Path):
    path = tmp_path / "timestamp.npy"
    values = np.arange(10, dtype=np.int64)
    np.save(path, values)

    info = inspect_source(path)
    assert info.adapter == "npy"
    assert info.rows == 10
    assert info.input_type.dtype == "int64"
    assert info.input_type.row_width == 1
    assert info.input_type.row_scalar is True
    assert info.input_type.cpp_type == "std::int64_t"

    mapped = open_source(path)
    try:
        assert mapped.data_pointer == mapped.owner.array.ctypes.data
        assert mapped.owner.info.data_offset == mapped.owner.array.offset
        assert mapped.data_pointer == int(mapped.owner.array.ctypes.data)
        np.testing.assert_array_equal(mapped.owner.array, values)
    finally:
        mapped.close()


def test_typed_row_scalar_npy_input_broadcasts_without_copy(tmp_path: Path):
    rows, n = 32, 9
    timestamp = np.arange(rows, dtype=np.int64) + 1_700_000_000_000_000
    close = np.arange(rows * n, dtype=np.float64).reshape(rows, n) / 10.0
    timestamp_path = tmp_path / "_ev_ts.npy"
    close_path = tmp_path / "close.npy"
    np.save(timestamp_path, timestamp)
    np.save(close_path, close)
    data = {"close": close_path, "_ev_ts": timestamp_path}

    runtime = compile_formula("close + _ev_ts", data, n_instruments=n)
    assert runtime.input_types[0].dtype == "float64"
    assert runtime.input_types[0].row_width == n
    assert runtime.input_types[1].dtype == "int64"
    assert runtime.input_types[1].row_width == 1

    out_path = tmp_path / "out.bin"
    result = runtime.run(out_path=out_path)
    assert result.rows == rows
    actual = np.memmap(out_path, mode="r", dtype=np.float64, shape=(rows, n))
    np.testing.assert_allclose(actual, close + timestamp[:, None], rtol=0.0, atol=0.0)


def test_npy_and_raw_sources_can_be_mixed_per_input(tmp_path: Path):
    rows, n = 17, 4
    left = np.arange(rows * n, dtype=np.float64).reshape(rows, n)
    right = 100.0 + left
    left_path = tmp_path / "left.npy"
    right_path = tmp_path / "right.bin"
    np.save(left_path, left)
    right.tofile(right_path)

    data = {
        "left": left_path,
        "right": source(
            right_path,
            input_type=InputTypeSpec("float64", n),
        ),
    }
    runtime = compile_formula("left + right", data, n_instruments=n)
    out_path = tmp_path / "mixed.bin"
    runtime.run(out_path=out_path)
    actual = np.asarray(
        np.memmap(out_path, mode="r", dtype=np.float64, shape=(rows, n))
    )
    np.testing.assert_array_equal(actual, left + right)


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
    runtime = compile_formula(
        formula,
        {"_ev_ts": timestamp_path, "close": close_path},
        n_instruments=n,
    )
    generated = runtime.generated_cpp.read_text()
    assert "DenseTupleGroupResolver" in generated
    assert "InputSrc<0, std::int64_t, 1>" in generated
    assert "stackdsl::CopyNode<1," in generated
    assert "stackdsl::NaryExpressionSrc<std::int64_t" in generated
    assert "stackdsl::BinaryNode<" not in generated
    assert "stackdsl::UnaryNode<" not in generated
    assert "stackdsl::SlotSrc<" in generated and ", true>" in generated
    assert "std::int64_t, stackdsl::DivOp" in generated
    assert "std::int64_t, stackdsl::ModOp" in generated

    out_path = tmp_path / "grouped.bin"
    runtime.run(out_path=out_path)
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


def test_dense_key_offset_maps_domain_to_zero_based_slots(tmp_path: Path):
    venue = np.array([10, 11, 10, 12, 11], dtype=np.int32)
    close = np.array([1.0, 10.0, 2.0, 100.0, 20.0], dtype=np.float64)
    venue_path = tmp_path / "venue.npy"
    close_path = tmp_path / "close.npy"
    np.save(venue_path, venue)
    np.save(close_path, close)

    formula = groupby(
        Key(var("venue"), num_keys=3, offset=10, row_scalar=True, dtype="int32"),
        var("close"),
        cumsum(self_),
    )
    runtime = compile_formula(
        formula,
        {"venue": venue_path, "close": close_path},
        n_instruments=1,
    )
    generated = runtime.generated_cpp.read_text()
    assert "DenseTupleGroupResolver" in generated
    assert "InputSrc<0, std::int32_t, 1>" in generated
    assert ", 3, 10, true>" in generated

    out_path = tmp_path / "offset.bin"
    runtime.run(out_path=out_path)
    actual = np.asarray(
        np.memmap(out_path, mode="r", dtype=np.float64, shape=(venue.size, 1))
    )[:, 0]
    np.testing.assert_array_equal(actual, np.array([1.0, 10.0, 3.0, 100.0, 30.0]))


def test_int64_hash_keys_remain_distinct_above_2_to_53(tmp_path: Path):
    key_values = np.array([2**53, 2**53 + 1, 2**53], dtype=np.int64)
    close = np.array([1.0, 10.0, 2.0], dtype=np.float64)
    key_path = tmp_path / "key_value.npy"
    close_path = tmp_path / "close.npy"
    np.save(key_path, key_values)
    np.save(close_path, close)

    formula = groupby(
        Key(var("key_value"), row_scalar=True, dtype="int64"),
        var("close"),
        cumsum(self_),
    )
    runtime = compile_formula(
        formula,
        {"key_value": key_path, "close": close_path},
        n_instruments=1,
    )
    generated = runtime.generated_cpp.read_text()
    assert "HashGroupResolver" in generated
    assert "InputSrc<0, std::int64_t, 1>" in generated

    out_path = tmp_path / "exact_hash.bin"
    runtime.run(out_path=out_path)
    actual = np.asarray(
        np.memmap(out_path, mode="r", dtype=np.float64, shape=(3, 1))
    )[:, 0]
    np.testing.assert_array_equal(actual, np.array([1.0, 10.0, 3.0]))
