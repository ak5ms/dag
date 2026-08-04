from __future__ import annotations

import os
from pathlib import Path
from statistics import median
import tempfile

import numpy as np

from flows.riskmodel import roll_rets
from flows.roll_rets_keys import roll_rets_keys
from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("CPP_STREAM_ROLL_KEYS_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_ROLL_KEYS_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_ROLL_KEYS_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_ROLL_KEYS_WARMUPS", "1"))
THREAD_TEXT = os.environ.get("CPP_STREAM_ROLL_KEYS_THREADS", "1,4")
OUTPUT_DIR = Path(os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR", "/dev/shm"))
MIN_RATIO = float(os.environ.get("CPP_STREAM_ROLL_KEYS_MIN_RATIO", "0.97"))


def thread_counts() -> tuple[int, ...]:
    available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    requested = [int(value) for value in THREAD_TEXT.split(",") if value.strip()]
    return tuple(sorted({max(1, min(value, available)) for value in requested}))


def create_inputs(root: Path) -> dict[str, Path]:
    names = (
        "_ev_ts",
        "session_start0",
        "session_end0",
        "volume_out0",
        "is_tradable_out0",
        "is_tradable_out1",
        "wdte_out0",
        "mp_out0.close",
        "mp_out1.close",
    )
    paths = {name: root / f"input_{index}.npy" for index, name in enumerate(names)}
    scalar = {"_ev_ts", "wdte_out0"}
    arrays = {
        name: np.lib.format.open_memmap(
            paths[name],
            mode="w+",
            dtype=np.float64,
            shape=(ROWS,) if name in scalar else (ROWS, N),
        )
        for name in names
    }

    minute_us = 60_000_000.0
    day_us = 86_400_000_000.0
    session_minutes = 1440
    base = 1_700_000_000_000_000.0
    lanes = np.arange(N, dtype=np.float64)[None, :]
    for start in range(0, ROWS, 65_536):
        stop = min(start + 65_536, ROWS)
        t = np.arange(start, stop, dtype=np.float64)
        integer_t = t.astype(np.int64)
        day = np.floor_divide(integer_t, session_minutes)
        minute = np.remainder(integer_t, session_minutes)
        session_start = base + day.astype(np.float64) * day_us
        event_ts = session_start + minute.astype(np.float64) * minute_us
        session_end = session_start + day_us
        weekday = (np.remainder(day + 2, 7) < 5).astype(np.float64)
        tradable_scalar = (
            ((minute >= 60) & (minute < 1380)).astype(np.float64) * weekday
        )
        tradable = tradable_scalar[:, None] * np.ones((1, N))
        phase = minute.astype(np.float64)[:, None] / session_minutes
        volume = np.maximum(
            100.0 + 25.0 * np.sin(2.0 * np.pi * phase) + lanes,
            0.0,
        ) * tradable
        time_column = t[:, None]

        arrays["_ev_ts"][start:stop] = event_ts
        # Store session metadata redundantly by lane. The baseline compiler cannot
        # prove these rows are invariant; RollRetsWithKeys asserts that fact.
        arrays["session_start0"][start:stop] = session_start[:, None]
        arrays["session_end0"][start:stop] = session_end[:, None]
        arrays["volume_out0"][start:stop] = volume
        arrays["is_tradable_out0"][start:stop] = tradable
        arrays["is_tradable_out1"][start:stop] = tradable
        arrays["wdte_out0"][start:stop] = np.where(
            np.remainder(day, 5) == 0, 1.0, 2.0
        )
        arrays["mp_out0.close"][start:stop] = (
            100.0 + 0.0010 * time_column + 0.01 * lanes
        )
        arrays["mp_out1.close"][start:stop] = (
            101.0 + 0.0011 * time_column + 0.01 * lanes
        )

    for array in arrays.values():
        array.flush()
    arrays.clear()
    return paths


def compare_outputs(left: Path, right: Path) -> float:
    a = np.memmap(left, mode="r", dtype=np.float64, shape=(ROWS, N))
    b = np.memmap(right, mode="r", dtype=np.float64, shape=(ROWS, N))
    checksum = 0.0
    try:
        for start in range(0, ROWS, 262_144):
            stop = min(start + 262_144, ROWS)
            aa = np.asarray(a[start:stop])
            bb = np.asarray(b[start:stop])
            if not np.array_equal(np.isnan(aa), np.isnan(bb)):
                raise RuntimeError(f"NaN mismatch at rows [{start}:{stop}]")
            finite = np.isfinite(aa)
            if not np.array_equal(aa[finite], bb[finite]):
                raise RuntimeError(f"value mismatch at rows [{start}:{stop}]")
            checksum += float(np.sum(aa[finite], dtype=np.float64))
    finally:
        del a, b
    return checksum


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    counts = thread_counts()
    with tempfile.TemporaryDirectory(prefix="cpp_stream_roll_keys_") as temporary:
        data = create_inputs(Path(temporary))
        runtimes = {
            "baseline": compile_formula(
                roll_rets,
                data,
                default_group_capacity=4096,
            ),
            "keyed": compile_formula(
                roll_rets_keys,
                data,
                default_group_capacity=4096,
            ),
        }
        if any(runtime.n_instruments != N for runtime in runtimes.values()):
            raise RuntimeError(
                f"automatic shape inference failed: "
                f"{ {name: runtime.n_instruments for name, runtime in runtimes.items()} }"
            )

        for threads in counts:
            paths = {
                name: OUTPUT_DIR / f"roll_rets_{name}_{threads}.bin"
                for name in runtimes
            }
            for name, runtime in runtimes.items():
                for _ in range(WARMUPS):
                    runtime.run(
                        out_path=paths[name],
                        threads=threads,
                        pin_threads=True,
                        async_writeback_mb=0,
                    )

            timings = {name: [] for name in runtimes}
            for repetition in range(RUNS):
                order = (
                    ("baseline", "keyed")
                    if repetition % 2 == 0
                    else ("keyed", "baseline")
                )
                for name in order:
                    result = runtimes[name].run(
                        out_path=paths[name],
                        threads=threads,
                        pin_threads=True,
                        async_writeback_mb=0,
                    )
                    timings[name].append(result.seconds)

            checksum = compare_outputs(paths["baseline"], paths["keyed"])
            baseline = median(timings["baseline"])
            keyed = median(timings["keyed"])
            ratio = baseline / keyed

            print("---")
            print(f"rows={ROWS:,} instruments={N} threads={threads}")
            print(f"baseline_plan={runtimes['baseline'].parallel_plan}")
            print(f"keyed_plan={runtimes['keyed'].parallel_plan}")
            print(
                "baseline_runs="
                + ", ".join(f"{value:.6f}" for value in timings["baseline"])
            )
            print(
                "keyed_runs="
                + ", ".join(f"{value:.6f}" for value in timings["keyed"])
            )
            print(f"baseline_median_seconds={baseline:.6f}")
            print(f"keyed_median_seconds={keyed:.6f}")
            print(f"keyed_vs_baseline_speedup={ratio:.4f}x")
            print(f"checksum={checksum:.12g}")

            if ratio < MIN_RATIO:
                raise RuntimeError(
                    f"key hint throughput regressed: {ratio:.4f}x < {MIN_RATIO:.4f}x"
                )
            for path in paths.values():
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
