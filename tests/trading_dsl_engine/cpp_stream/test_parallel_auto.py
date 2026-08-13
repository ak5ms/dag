from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from flows.riskmodel import roll_rets
from trading_dsl_engine.base.dsl import ewm, var
from trading_dsl_engine.cpp_stream import compile_formula


def _available_cpus() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


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


def test_default_automatic_execution_avoids_threads_for_tiny_workloads(
    tmp_path: Path,
) -> None:
    rows, n = 128, 8
    x = np.arange(rows * n, dtype=np.float64).reshape(rows, n)
    runtime = compile_formula("x + 1.0", {"x": x}, n_instruments=n)
    assert runtime.parallel_plan.mode == "rows"
    assert runtime.parallel_plan.auto_multicore
    result = runtime.run(out_path=tmp_path / "default_auto.bin")
    assert result.threads == 1


def test_automatic_mode_uses_row_count_for_low_work_ewm(tmp_path: Path) -> None:
    rows, n = 4096, 32
    x = np.arange(rows * n, dtype=np.float64).reshape(rows, n)
    runtime = compile_formula(ewm(var("x"), 21), {"x": x}, n_instruments=n)
    assert runtime.parallel_plan.mode == "lanes"
    assert runtime.parallel_plan.auto_multicore
    result = runtime.run(out_path=tmp_path / "ewm_auto.bin")
    assert result.threads == 1


@pytest.mark.skipif(_available_cpus() < 2, reason="requires at least two available CPUs")
def test_default_automatic_mode_parallelizes_roll_rets(tmp_path: Path) -> None:
    rows, n = 2048, 9
    data = _roll_data(rows, n)
    runtime = compile_formula(
        roll_rets,
        data,
        n_instruments=n,
        default_group_capacity=256,
    )
    assert runtime.parallel_plan.mode == "lanes"
    assert runtime.parallel_plan.auto_multicore
    result = runtime.run(out_path=tmp_path / "roll_auto.bin", pin_threads=True)
    assert result.threads >= 2
