from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flows.riskmodel import roll_rets
from trading_dsl_engine.base.dsl import (
    cumsum,
    ewm,
    einsum,
    groupby,
    self_,
    univ,
    var,
    xs_rank,
)
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.cpp_stream import compile_formula


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


def _read(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(np.memmap(path, mode="r", dtype=np.float64, shape=shape)).copy()


def test_parallel_planner_selects_rows_lanes_and_serial() -> None:
    rows, n = 16, 9
    rng = np.random.default_rng(7)
    vectors = {
        "x": rng.normal(size=(rows, n)),
        "y": rng.normal(size=(rows, n)),
    }

    row_runtime = compile_formula(
        einsum("i,i->", var("x"), var("y")), vectors, n_instruments=n
    )
    assert row_runtime.parallel_plan.mode == "rows"

    lane_runtime = compile_formula(
        ewm(var("x"), 21), {"x": vectors["x"]}, n_instruments=n
    )
    assert lane_runtime.parallel_plan.mode == "lanes"

    serial_runtime = compile_formula(
        xs_rank(ewm(var("x"), 21)),
        {"x": vectors["x"]},
        n_instruments=n,
    )
    assert serial_runtime.parallel_plan.mode == "serial"


def test_row_parallel_elementwise_and_scalar_reduction_match_serial(
    tmp_path: Path,
) -> None:
    rows, n = 200_000, 32
    rng = np.random.default_rng(11)
    x = rng.normal(size=(rows, n)).astype(np.float64)
    y = rng.normal(size=(rows, n)).astype(np.float64)
    formula = einsum(
        "i,i->",
        ((var("x") * 1.5 + var("y")) ** 2) / 3.0,
        var("x") - 0.25 * var("y"),
    )
    runtime = compile_formula(formula, {"x": x, "y": y}, n_instruments=n)
    assert runtime.parallel_plan.mode == "rows"

    serial_path = tmp_path / "row_serial.bin"
    parallel_path = tmp_path / "row_parallel.bin"
    serial = runtime.run(out_path=serial_path, threads=1)
    requested = min(4, _available_cpus())
    parallel = runtime.run(
        out_path=parallel_path,
        threads=requested,
        pin_threads=True,
    )
    expected = np.einsum(
        "ri,ri->r",
        ((x * 1.5 + y) ** 2) / 3.0,
        x - 0.25 * y,
    )
    np.testing.assert_allclose(
        _read(serial_path, (rows,)), expected, rtol=2e-13, atol=2e-13
    )
    np.testing.assert_allclose(
        _read(parallel_path, (rows,)), expected, rtol=2e-13, atol=2e-13
    )
    assert serial.threads == 1
    assert parallel.available_cpus >= 1
    assert parallel.threads == requested


def test_lane_parallel_ewm_matches_serial(tmp_path: Path) -> None:
    rows, n = 80_000, 32
    rng = np.random.default_rng(13)
    x = rng.normal(size=(rows, n)).astype(np.float64)
    x[111, 3] = np.nan
    x[507, 19] = np.nan
    runtime = compile_formula(
        ewm(var("x"), 21), {"x": x}, n_instruments=n
    )
    assert runtime.parallel_plan.mode == "lanes"

    serial_path = tmp_path / "ewm_serial.bin"
    parallel_path = tmp_path / "ewm_parallel.bin"
    runtime.run(out_path=serial_path, threads=1)
    requested = min(4, _available_cpus())
    result = runtime.run(
        out_path=parallel_path,
        threads=requested,
        pin_threads=True,
    )
    np.testing.assert_allclose(
        _read(parallel_path, (rows, n)),
        _read(serial_path, (rows, n)),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert result.threads == requested


def test_lane_parallel_dense_grouped_state_matches_serial(tmp_path: Path) -> None:
    rows, n = 20_000, 9
    rng = np.random.default_rng(17)
    close = rng.normal(size=(rows, n)).astype(np.float64)
    minute = np.remainder(np.arange(rows, dtype=np.int64), 60)
    formula = groupby(
        (
            univ([0], [1, 2], list(range(3, 9))),
            Key(var("minute_key"), num_keys=60, row_scalar=True, dtype="int64"),
        ),
        var("close"),
        ewm(cumsum(self_), 3),
    )
    runtime = compile_formula(
        formula,
        {"minute_key": minute, "close": close},
        n_instruments=n,
    )
    assert runtime.parallel_plan.mode == "lanes"

    serial_path = tmp_path / "group_serial.bin"
    parallel_path = tmp_path / "group_parallel.bin"
    runtime.run(out_path=serial_path, threads=1)
    requested = min(4, _available_cpus())
    result = runtime.run(
        out_path=parallel_path,
        threads=requested,
        pin_threads=True,
    )
    np.testing.assert_allclose(
        _read(parallel_path, (rows, n)),
        _read(serial_path, (rows, n)),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert result.threads == requested


def test_cross_sectional_temporal_graph_falls_back_to_one_thread(
    tmp_path: Path,
) -> None:
    rows, n = 4096, 9
    rng = np.random.default_rng(19)
    x = rng.normal(size=(rows, n)).astype(np.float64)
    runtime = compile_formula(
        xs_rank(ewm(var("x"), 21)), {"x": x}, n_instruments=n
    )
    assert runtime.parallel_plan.mode == "serial"
    result = runtime.run(
        out_path=tmp_path / "serial_fallback.bin",
        threads=min(4, _available_cpus()),
        pin_threads=True,
    )
    assert result.threads == 1
    assert result.parallel_mode == "serial"


def _roll_data(rows: int, n: int) -> dict[str, np.ndarray]:
    minute_us = 60_000_000.0
    day_us = 86_400_000_000.0
    base = 1_700_000_000_000_000.0
    t = np.arange(rows, dtype=np.float64)[:, None]
    lane = np.arange(n, dtype=np.float64)[None, :]
    session_start = np.full((rows, n), base, dtype=np.float64)
    session_end = session_start + day_us
    event_ts = (base + (60.0 + t) * minute_us) * np.ones((1, n))
    phase = (event_ts - session_start) / day_us
    tradable0 = np.ones((rows, n), dtype=np.float64)
    tradable1 = np.ones((rows, n), dtype=np.float64)
    tradable0[17:20, 2] = 0.0
    tradable1[31:33, 5] = 0.0
    volume = 100.0 + 25.0 * np.sin(2.0 * np.pi * phase) + lane
    close0 = 100.0 + 0.002 * t + 0.01 * lane
    close1 = 101.0 + 0.0022 * t + 0.01 * lane
    close0[tradable0 != 1.0] = np.nan
    close1[tradable1 != 1.0] = np.nan
    wdte = np.where((t // 32) % 2 == 0, 1.0, 2.0) * np.ones((1, n))
    return {
        "_ev_ts": event_ts,
        "session_start0": session_start,
        "session_end0": session_end,
        "volume_out0": volume,
        "is_tradable_out0": tradable0,
        "is_tradable_out1": tradable1,
        "wdte_out0": wdte,
        "mp_out0.close": close0,
        "mp_out1.close": close1,
    }


def test_roll_rets_lane_parallel_matches_serial(tmp_path: Path) -> None:
    rows, n = 4096, 9
    data = _roll_data(rows, n)
    runtime = compile_formula(
        roll_rets,
        data,
        n_instruments=n,
        default_group_capacity=256,
    )
    assert runtime.parallel_plan.mode == "lanes"

    serial_path = tmp_path / "roll_serial.bin"
    parallel_path = tmp_path / "roll_parallel.bin"
    runtime.run(out_path=serial_path, threads=1)
    requested = min(4, _available_cpus(), n)
    result = runtime.run(
        out_path=parallel_path,
        threads=requested,
        pin_threads=True,
    )
    np.testing.assert_allclose(
        _read(parallel_path, (rows, n)),
        _read(serial_path, (rows, n)),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    assert result.threads == requested


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two available CPUs")
def test_parallel_run_reports_multiple_busy_cores(tmp_path: Path) -> None:
    rows, n = 400_000, 64
    rng = np.random.default_rng(23)
    x = rng.normal(size=(rows, n)).astype(np.float64)
    y = rng.normal(size=(rows, n)).astype(np.float64)
    runtime = compile_formula(
        einsum("i,i->", var("x") * var("y") + var("x"), var("y")),
        {"x": x, "y": y},
        n_instruments=n,
    )
    result = runtime.run(
        out_path=tmp_path / "busy.bin",
        threads=min(4, _available_cpus()),
        pin_threads=True,
    )
    assert result.threads >= 2
    assert result.average_busy_cores > 1.1
