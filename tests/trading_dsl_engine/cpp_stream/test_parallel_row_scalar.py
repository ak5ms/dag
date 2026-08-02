from __future__ import annotations

from pathlib import Path

import numpy as np

from flows.riskmodel import roll_rets
from trading_dsl_engine.base.dsl import ffill, var
from trading_dsl_engine.cpp_stream import compile_formula


def _read(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(np.memmap(path, mode="r", dtype=np.float64, shape=shape)).copy()


def test_stateful_row_scalar_is_recomputed_at_scalar_slot_for_each_worker(
    tmp_path: Path,
) -> None:
    rows, n = 4096, 9
    scalar = np.full(rows, np.nan, dtype=np.float64)
    scalar[0] = 1.0
    scalar[1500] = 2.0
    vector = np.arange(rows * n, dtype=np.float64).reshape(rows, n) / 1000.0
    runtime = compile_formula(
        var("vector") + ffill(var("scalar")),
        {"scalar": scalar, "vector": vector},
        n_instruments=n,
    )
    assert runtime.parallel_plan.mode == "lanes"
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::FFillNode<1," in generated

    serial_path = tmp_path / "scalar_serial.bin"
    parallel_path = tmp_path / "scalar_parallel.bin"
    runtime.run(out_path=serial_path, threads=1)
    runtime.run(out_path=parallel_path, threads=2, pin_threads=True)
    expected_scalar = np.where(np.arange(rows) < 1500, 1.0, 2.0)
    expected = vector + expected_scalar[:, None]
    np.testing.assert_array_equal(_read(serial_path, (rows, n)), expected)
    np.testing.assert_array_equal(_read(parallel_path, (rows, n)), expected)


def _roll_data(rows: int, n: int) -> dict[str, np.ndarray]:
    minute_us = 60_000_000.0
    day_us = 86_400_000_000.0
    base = 1_700_000_000_000_000.0
    t = np.arange(rows, dtype=np.float64)
    lane = np.arange(n, dtype=np.float64)[None, :]
    day = np.floor_divide(t.astype(np.int64), 1440)
    minute = np.remainder(t.astype(np.int64), 1440)
    session_start = base + day.astype(np.float64) * day_us
    event_ts = session_start + minute.astype(np.float64) * minute_us
    session_end = session_start + day_us
    weekday = (np.remainder(day + 2, 7) < 5).astype(np.float64)
    tradable_scalar = ((minute >= 60) & (minute < 1380)).astype(np.float64) * weekday
    phase = minute.astype(np.float64)[:, None] / 1440.0
    tradable = tradable_scalar[:, None] * np.ones((1, n))
    volume = np.maximum(
        100.0 + 25.0 * np.sin(2.0 * np.pi * phase) + lane,
        0.0,
    ) * tradable
    time_column = t[:, None]
    return {
        "_ev_ts": event_ts,
        "session_start0": session_start,
        "session_end0": session_end,
        "volume_out0": volume,
        "is_tradable_out0": tradable,
        "is_tradable_out1": tradable,
        "wdte_out0": np.where(np.remainder(day, 5) == 0, 1.0, 2.0),
        "mp_out0.close": 100.0 + 0.0010 * time_column + 0.01 * lane,
        "mp_out1.close": 101.0 + 0.0011 * time_column + 0.01 * lane,
    }


def test_roll_rets_with_production_row_scalar_sources_matches_two_workers(
    tmp_path: Path,
) -> None:
    rows, n = 2000, 9
    data = _roll_data(rows, n)
    runtime = compile_formula(
        roll_rets,
        data,
        n_instruments=n,
        default_group_capacity=64,
    )
    assert runtime.parallel_plan.mode == "lanes"
    serial_path = tmp_path / "roll_scalar_serial.bin"
    parallel_path = tmp_path / "roll_scalar_parallel.bin"
    runtime.run(out_path=serial_path, threads=1)
    runtime.run(out_path=parallel_path, threads=2, pin_threads=True)
    np.testing.assert_array_equal(
        _read(parallel_path, (rows, n)),
        _read(serial_path, (rows, n)),
    )
