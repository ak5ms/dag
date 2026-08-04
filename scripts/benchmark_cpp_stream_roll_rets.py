from __future__ import annotations

import os
from pathlib import Path
from statistics import mean, median
import tempfile

import numpy as np

from flows.riskmodel import roll_rets
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.ir import compile_ir


ROWS = int(os.environ.get("CPP_STREAM_ROLL_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_ROLL_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_ROLL_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_ROLL_WARMUPS", "1"))
PREFETCH_ROWS = int(os.environ.get("CPP_STREAM_ROLL_PREFETCH_ROWS", "16"))
OUTPUT_DIR = os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR")
MIN_MROWS = float(os.environ.get("CPP_STREAM_ROLL_MIN_MROWS", "0"))
MINUTE_US = 60_000_000.0
DAY_US = 86_400_000_000.0
SESSION_MINUTES = 1440


def _create(path: Path, shape: tuple[int, ...]) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=shape)


def _build_inputs(root: Path) -> dict[str, Path]:
    names = compile_ir(roll_rets).input_names
    paths = {name: root / f"input_{index}.npy" for index, name in enumerate(names)}
    scalar_names = {"_ev_ts", "session_start0", "session_end0", "wdte_out0"}
    arrays = {
        name: _create(paths[name], (ROWS,) if name in scalar_names else (ROWS, N))
        for name in names
    }
    base = 1_700_000_000_000_000.0
    lane = np.arange(N, dtype=np.float64)[None, :]
    chunk = 131_072
    for start in range(0, ROWS, chunk):
        stop = min(start + chunk, ROWS)
        t = np.arange(start, stop, dtype=np.float64)
        day = np.floor_divide(t.astype(np.int64), SESSION_MINUTES)
        minute = np.remainder(t.astype(np.int64), SESSION_MINUTES)
        session_start = base + day.astype(np.float64) * DAY_US
        event_ts = session_start + minute.astype(np.float64) * MINUTE_US
        session_end = session_start + DAY_US
        weekday = (np.remainder(day + 2, 7) < 5).astype(np.float64)
        tradable_scalar = ((minute >= 60) & (minute < 1380)).astype(np.float64) * weekday
        phase = minute.astype(np.float64)[:, None] / SESSION_MINUTES
        tradable = tradable_scalar[:, None] * np.ones((1, N))
        volume = np.maximum(
            100.0 + 25.0 * np.sin(2.0 * np.pi * phase) + lane,
            0.0,
        ) * tradable
        time_column = t[:, None]
        close0 = 100.0 + 0.0010 * time_column + 0.01 * lane
        close1 = 101.0 + 0.0011 * time_column + 0.01 * lane
        wdte = np.where(np.remainder(day, 5) == 0, 1.0, 2.0)

        arrays["_ev_ts"][start:stop] = event_ts
        arrays["session_start0"][start:stop] = session_start
        arrays["session_end0"][start:stop] = session_end
        arrays["volume_out0"][start:stop] = volume
        arrays["is_tradable_out0"][start:stop] = tradable
        arrays["is_tradable_out1"][start:stop] = tradable
        arrays["wdte_out0"][start:stop] = wdte
        arrays["mp_out0.close"][start:stop] = close0
        arrays["mp_out1.close"][start:stop] = close1

    for array in arrays.values():
        array.flush()
    arrays.clear()
    return paths


def main() -> None:
    if N != 9:
        raise ValueError("riskmodel.roll_rets currently expects the nine-instrument benchmark universe")
    with tempfile.TemporaryDirectory(prefix="cpp_stream_roll_rets_") as temporary:
        root = Path(temporary)
        data = _build_inputs(root)
        session_count = (ROWS + SESSION_MINUTES - 1) // SESSION_MINUTES
        capacity = 1
        while capacity < session_count + 16:
            capacity *= 2
        runtime = compile_formula(
            roll_rets,
            data,
            n_instruments=N,
            default_group_capacity=max(64, capacity),
            prefetch_rows=PREFETCH_ROWS,
        )
        output_root = Path(OUTPUT_DIR) if OUTPUT_DIR else root
        output_root.mkdir(parents=True, exist_ok=True)
        output = output_root / "cpp_stream_roll_rets.bin"

        for _ in range(WARMUPS):
            runtime.run(out_path=output, async_writeback_mb=0)
        rates = [
            runtime.run(out_path=output, async_writeback_mb=0).rows_per_second
            for _ in range(RUNS)
        ]
        values = np.memmap(output, mode="r", dtype=np.float64, shape=(ROWS, N))
        checksum = float(np.nansum(values[-min(8192, ROWS):]))
        finite_fraction = float(np.isfinite(values[-min(8192, ROWS):]).mean())
        del values

        median_mrows = median(rates) / 1e6
        print("formula=flows.riskmodel.roll_rets")
        print(f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS}")
        print(f"group_capacity={max(64, capacity)}")
        print(f"scratch_slots={runtime.plan.scratch_slots}")
        print(f"matrix_scratch_slots={runtime.plan.matrix_scratch_slots}")
        print(f"matrix_scratch_width={runtime.plan.matrix_scratch_width}")
        print(f"median={median_mrows:.6f} M rows/s")
        print(f"mean={mean(rates) / 1e6:.6f} M rows/s")
        print(f"best={max(rates) / 1e6:.6f} M rows/s")
        print("runs=" + ", ".join(f"{rate / 1e6:.6f}" for rate in rates) + " M rows/s")
        print(f"checksum={checksum:.12g}")
        print(f"tail_finite_fraction={finite_fraction:.12g}")
        print(f"generated_cpp={runtime.generated_cpp}")
        if MIN_MROWS > 0 and median_mrows < MIN_MROWS:
            raise SystemExit(
                f"roll_rets median {median_mrows:.6f} M rows/s is below "
                f"CPP_STREAM_ROLL_MIN_MROWS={MIN_MROWS:.6f}"
            )


if __name__ == "__main__":
    main()
