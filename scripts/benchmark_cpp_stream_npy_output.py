from __future__ import annotations

import os
from pathlib import Path
from statistics import median
import tempfile
from time import perf_counter

import numpy as np

from trading_dsl_engine.base.dsl import var
from trading_dsl_engine.cpp_stream import compile_formula


ROWS = int(os.environ.get("CPP_STREAM_NPY_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_NPY_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_NPY_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_NPY_WARMUPS", "1"))
OUTPUT_DIR = Path(os.environ.get("CPP_STREAM_BENCH_OUTPUT_DIR", "/dev/shm"))
MIN_NATIVE_RATIO = float(os.environ.get("CPP_STREAM_NPY_MIN_NATIVE_RATIO", "0.97"))
MIN_WALL_RATIO = float(os.environ.get("CPP_STREAM_NPY_MIN_WALL_RATIO", "0.95"))


def create_input(path: Path, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    array = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float64,
        shape=(ROWS, N),
    )
    for start in range(0, ROWS, 65_536):
        stop = min(start + 65_536, ROWS)
        array[start:stop] = rng.normal(size=(stop - start, N))
    array.flush()
    del array
    return path


def verify(raw_path: Path, npy_path: Path) -> None:
    raw = np.memmap(raw_path, mode="r", dtype=np.float64, shape=(ROWS, N))
    npy = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    try:
        if npy.shape != (ROWS, N) or npy.dtype != np.float64:
            raise RuntimeError(f"unexpected npy metadata: shape={npy.shape} dtype={npy.dtype}")
        for start in range(0, ROWS, 262_144):
            stop = min(start + 262_144, ROWS)
            if not np.array_equal(raw[start:stop], npy[start:stop], equal_nan=True):
                raise RuntimeError(f"raw/npy output mismatch at rows [{start}:{stop}]")
    finally:
        del raw, npy


def timed_run(runtime, path: Path):
    started = perf_counter()
    result = runtime.run(
        out_path=path,
        threads=1,
        async_writeback_mb=0,
    )
    wall = perf_counter() - started
    return result, wall


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="cpp_stream_npy_output_") as temporary:
        root = Path(temporary)
        data = {
            "x": create_input(root / "x.npy", 1),
            "y": create_input(root / "y.npy", 2),
        }
        x, y = var("x"), var("y")
        formula = ((x * 1.01 + y) ** 2) / (x * x + 0.25)
        runtime = compile_formula(formula, data)
        if runtime.n_instruments != N:
            raise RuntimeError(
                f"automatic n_instruments={runtime.n_instruments}, expected {N}"
            )

        raw_path = OUTPUT_DIR / "cpp_stream_output_benchmark.bin"
        npy_path = OUTPUT_DIR / "cpp_stream_output_benchmark.npy"

        for _ in range(WARMUPS):
            timed_run(runtime, raw_path)
            timed_run(runtime, npy_path)

        native = {"raw": [], "npy": []}
        wall = {"raw": [], "npy": []}
        paths = {"raw": raw_path, "npy": npy_path}
        for repetition in range(RUNS):
            order = ("raw", "npy") if repetition % 2 == 0 else ("npy", "raw")
            for mode in order:
                result, elapsed = timed_run(runtime, paths[mode])
                native[mode].append(result.seconds)
                wall[mode].append(elapsed)

        verify(raw_path, npy_path)
        loaded = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        npy_offset = int(loaded.offset)
        del loaded

        raw_native = median(native["raw"])
        npy_native = median(native["npy"])
        raw_wall = median(wall["raw"])
        npy_wall = median(wall["npy"])
        native_ratio = raw_native / npy_native
        wall_ratio = raw_wall / npy_wall

        print(
            f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS}"
        )
        print(f"automatic_n_instruments={runtime.n_instruments}")
        print(f"npy_payload_offset={npy_offset}")
        print("raw_native_runs=" + ", ".join(f"{x:.6f}" for x in native["raw"]))
        print("npy_native_runs=" + ", ".join(f"{x:.6f}" for x in native["npy"]))
        print("raw_wall_runs=" + ", ".join(f"{x:.6f}" for x in wall["raw"]))
        print("npy_wall_runs=" + ", ".join(f"{x:.6f}" for x in wall["npy"]))
        print(f"raw_native_median_seconds={raw_native:.6f}")
        print(f"npy_native_median_seconds={npy_native:.6f}")
        print(f"npy_native_vs_raw_throughput={native_ratio:.4f}x")
        print(f"raw_wall_median_seconds={raw_wall:.6f}")
        print(f"npy_wall_median_seconds={npy_wall:.6f}")
        print(f"npy_wall_vs_raw_throughput={wall_ratio:.4f}x")

        if native_ratio < MIN_NATIVE_RATIO:
            raise RuntimeError(
                f"direct npy native throughput regressed: {native_ratio:.4f}x < "
                f"{MIN_NATIVE_RATIO:.4f}x"
            )
        if wall_ratio < MIN_WALL_RATIO:
            raise RuntimeError(
                f"direct npy end-to-end throughput regressed: {wall_ratio:.4f}x < "
                f"{MIN_WALL_RATIO:.4f}x"
            )

        raw_path.unlink(missing_ok=True)
        npy_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
