from __future__ import annotations

import argparse
import os
from pathlib import Path
from statistics import median
import tempfile

import numpy as np

from trading_dsl_engine.cpp_stream import InputTypeSpec, compile_formula, source


ROWS = int(os.environ.get("CPP_STREAM_IO_ROWS", "1000000"))
N = int(os.environ.get("CPP_STREAM_IO_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_IO_RUNS", "5"))
WARMUPS = int(os.environ.get("CPP_STREAM_IO_WARMUPS", "1"))
FORMULA = "cat(x + 1, x + 2, x + 3)"


def _run(runtime, output: Path) -> tuple[float, float]:
    for _ in range(WARMUPS):
        runtime.run(out_path=output, async_writeback_mb=0)
    rates = [runtime.run(out_path=output, async_writeback_mb=0).rows_per_second for _ in range(RUNS)]
    values = np.memmap(output, mode="r", dtype=np.float64, shape=(ROWS, N * 3))
    checksum = float(np.sum(values[-min(1024, ROWS):]))
    del values
    return median(rates) / 1e6, checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--format", choices=("all", "npy", "raw"), default="all")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="cpp_stream_io_") as temporary:
        root = Path(temporary)
        rng = np.random.default_rng(42)
        npy_path = root / "x.npy"
        raw_path = root / "x.bin"
        values = np.lib.format.open_memmap(npy_path, mode="w+", dtype=np.float64, shape=(ROWS, N))
        chunk = 131_072
        for start in range(0, ROWS, chunk):
            stop = min(start + chunk, ROWS)
            values[start:stop] = rng.normal(size=(stop - start, N))
        values.flush()
        np.asarray(values).tofile(raw_path)
        del values

        results: dict[str, tuple[float, float]] = {}
        if args.format in {"all", "npy"}:
            runtime = compile_formula(FORMULA, {"x": npy_path}, n_instruments=N)
            generated = runtime.generated_cpp.read_text()
            assert generated.count("for (std::size_t t = 0; t < rows; ++t)") == 1
            assert generated.count("ctx.inputs[0] =") == 1
            results["npy"] = _run(runtime, root / "npy.out")
        if args.format in {"all", "raw"}:
            runtime = compile_formula(
                FORMULA,
                {"x": source(raw_path, input_type=InputTypeSpec("float64", N))},
                n_instruments=N,
            )
            results["raw"] = _run(runtime, root / "raw.out")

        for name, (rate, checksum) in results.items():
            print(f"format={name} median={rate:.6f} M rows/s checksum={checksum:.12g}")
        if len(results) == 2:
            np.testing.assert_allclose(results["npy"][1], results["raw"][1], rtol=0.0, atol=0.0)
            print(f"raw_to_npy_ratio={results['raw'][0] / results['npy'][0]:.6f}")


if __name__ == "__main__":
    main()
