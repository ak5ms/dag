from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from trading_dsl_engine.base.dsl import cumsum, ewm, groupby, self_, univ, var
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.ir import compile_ir


def _formula():
    return groupby(
        (univ([0], [1, 2], list(range(3, 9))), var("minute")),
        var("close"),
        ewm(cumsum(self_), 3),
    )


def test_minute_terminal_is_derived_from_event_timestamp():
    program = compile_ir(_formula())
    assert program.input_names == ("_ev_ts", "close")


def test_minute_groupby_native_matches_reference(tmp_path: Path):
    if sys.platform == "win32" or shutil.which("g++") is None:
        pytest.skip("cpp_stream integration test requires POSIX and g++")

    rows, cols = 180, 9
    rng = np.random.default_rng(123)
    close = rng.normal(size=(rows, cols)).astype(np.float64)
    day_us = 86_400_000_000.0
    minute_us = 60_000_000.0
    base = np.floor(1_700_000_000_000_000.0 / day_us) * day_us
    row_ts = base + np.arange(rows, dtype=np.float64) * minute_us
    ev_ts = np.broadcast_to(row_ts[:, None], (rows, cols)).copy()

    close_path = tmp_path / "close.bin"
    ts_path = tmp_path / "_ev_ts.bin"
    out_path = tmp_path / "out.bin"
    close.tofile(close_path)
    ev_ts.tofile(ts_path)

    runtime = compile_formula(_formula(), n_instruments=cols)
    runtime.run_files({"_ev_ts": ts_path, "close": close_path}, out_path=out_path)
    actual = np.fromfile(out_path, dtype=np.float64).reshape(rows, cols)

    cumsum_state: dict[tuple[int, int], float] = {}
    ewm_state: dict[tuple[int, int], float] = {}
    expected = np.empty_like(close)
    alpha = 0.5
    for t in range(rows):
        minute = int((row_ts[t] % day_us) // minute_us) % 60
        for lane in range(cols):
            key = (lane, minute)
            cumulative = cumsum_state.get(key, 0.0) + close[t, lane]
            cumsum_state[key] = cumulative
            previous = ewm_state.get(key)
            current = cumulative if previous is None else previous + alpha * (cumulative - previous)
            ewm_state[key] = current
            expected[t, lane] = current

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
