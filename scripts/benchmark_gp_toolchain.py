"""A/B cpp_stream GP compilation without changing the native formula graph."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import shutil
import statistics
import time

import numpy as np

from trading_dsl_engine.base.dsl import (
    abs as dsl_abs,
    cat,
    ewm,
    ffill,
    purify,
    shift,
    var,
    where,
    xs_rank,
)
from trading_dsl_engine.cpp_stream import compile_formula


INPUT_GLOB = os.environ.get(
    "GP_TOOLCHAIN_INPUT_GLOB",
    "/tmp/gp-alpha-search-data/*.npy",
)
DERIVED_DIR = Path(
    os.environ.get("GP_TOOLCHAIN_DERIVED_DIR", "/tmp/gp-alpha-search-opt/derived")
)
OUTPUT_DIR = Path(
    os.environ.get("GP_TOOLCHAIN_OUTPUT_DIR", "/tmp/gp-toolchain-benchmark")
)
ROWS = int(os.environ.get("GP_TOOLCHAIN_ROWS", "5000000"))
N_INSTRUMENTS = int(os.environ.get("GP_TOOLCHAIN_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("GP_TOOLCHAIN_RUNS", "3"))
WARMUPS = int(os.environ.get("GP_TOOLCHAIN_WARMUPS", "1"))


FIELDS = (
    "bp0_out0",
    "ap0_out0",
    "vwap_out0",
    "volume_out0",
    "bp_out0.close",
    "ap_out0.close",
    "volume_a0_out0",
    "volume_b0_out0",
)


def _source_mapping():
    sources = {
        Path(path).stem: Path(path)
        for path in glob.glob(INPUT_GLOB)
        if Path(path).suffix == ".npy"
    }
    for name in ("clean_rets", "volatility"):
        path = DERIVED_DIR / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(path)
        sources[name] = path
    needed = {*FIELDS, "is_tradable_out0", "clean_rets", "volatility"}
    missing = sorted(needed - set(sources))
    if missing:
        raise KeyError(f"missing toolchain benchmark inputs: {missing}")
    if ROWS <= 0:
        return {name: sources[name] for name in needed}
    return {
        name: np.load(sources[name], mmap_mode="r")[:ROWS]
        for name in needed
    }


def _l1_norm(value):
    return purify(value / dsl_abs(value).sum(axis=-1))


def _formula(variant: int):
    pnls = []
    for index, field in enumerate(FIELDS):
        scaled = var(field) * (1.0 + 0.01 * variant)
        signal = xs_rank(ewm(scaled, (index + 1) * 5 + variant))
        weight = _l1_norm(signal) / var("volatility")
        held = shift(
            ffill(
                where(
                    var("is_tradable_out0"),
                    weight,
                    float("nan"),
                )
            ),
            1,
            1,
        )
        pnls.append(held * var("clean_rets"))
    candidate_pnl = cat(*pnls).sum(axis=1)
    return candidate_pnl.mean(axis=0) / candidate_pnl.std(axis=0)


def _configure(config):
    os.environ["CXX"] = config["compiler"]
    os.environ["TRADING_DSL_ENGINE_CPP_LTO"] = "1" if config["lto"] else "0"
    os.environ["TRADING_DSL_ENGINE_CPP_PCH"] = "1" if config["pch"] else "0"
    os.environ["TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS"] = config.get(
        "extra_flags", ""
    )
    os.environ["TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS"] = config["link_flags"]
    os.environ["TRADING_DSL_ENGINE_CPP_STREAM_CACHE"] = str(
        OUTPUT_DIR / "cache" / config["name"]
    )


def _benchmark(config, formulas, sources):
    cache = OUTPUT_DIR / "cache" / config["name"]
    shutil.rmtree(cache, ignore_errors=True)
    _configure(config)

    runtimes = []
    compile_seconds = []
    for formula in formulas:
        compile_started = time.perf_counter()
        runtimes.append(
            compile_formula(
                formula,
                sources,
                n_instruments=N_INSTRUMENTS,
                prefetch_rows=16,
            )
        )
        compile_seconds.append(time.perf_counter() - compile_started)
    runtime = runtimes[0]

    output = OUTPUT_DIR / f"{config['name']}.npy"
    for _ in range(WARMUPS):
        runtime.run(out_path=output, threads=1)
    native = []
    wall = []
    for _ in range(RUNS):
        started = time.perf_counter()
        result = runtime.run(out_path=output, threads=1)
        wall.append(time.perf_counter() - started)
        native.append(float(result.seconds))
    values = np.asarray(np.load(output, mmap_mode="r"))
    return {
        **config,
        "compile_seconds": compile_seconds,
        "compile_seconds_sum": sum(compile_seconds),
        "compile_seconds_after_first_median": statistics.median(
            compile_seconds[1:]
        ),
        "native_seconds": native,
        "median_native_seconds": statistics.median(native),
        "wall_seconds": wall,
        "median_wall_seconds": statistics.median(wall),
        "checksum": float(np.nansum(values)),
        "generated_cpp": str(runtime.generated_cpp),
        "generated_cpp_bytes": runtime.generated_cpp.stat().st_size,
        "library": str(runtime.library_path),
        "output_shape": list(runtime.plan.output_shape),
        "parallel_plan": {
            "mode": runtime.parallel_plan.mode,
            "reason": runtime.parallel_plan.reason,
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = _source_mapping()
    formulas = tuple(_formula(variant) for variant in range(4))
    clang = shutil.which("clang++")
    if clang is None:
        raise RuntimeError("benchmark requires clang++")
    lld = "-fuse-ld=lld" if shutil.which("ld.lld") else ""
    configs = (
        {
            "name": "clang_lto1_no_pch",
            "compiler": clang,
            "lto": True,
            "pch": False,
            "link_flags": lld,
            "extra_flags": "",
        },
        {
            "name": "clang_thinlto_pch",
            "compiler": clang,
            "lto": False,
            "pch": True,
            "link_flags": lld,
            "extra_flags": "-flto=thin",
        },
        {
            "name": "clang_lto0_pch",
            "compiler": clang,
            "lto": False,
            "pch": True,
            "link_flags": lld,
            "extra_flags": "",
        },
    )
    results = []
    for config in configs:
        print(f"BENCH_START {config['name']}", flush=True)
        result = _benchmark(config, formulas, sources)
        results.append(result)
        print(json.dumps(result, sort_keys=True), flush=True)

    reference = results[0]["checksum"]
    for result in results[1:]:
        if not np.isclose(result["checksum"], reference, rtol=1e-12, atol=1e-14):
            raise RuntimeError(
                f"toolchain checksum mismatch: {result['name']} "
                f"{result['checksum']} versus {reference}"
            )
    payload = {
        "rows": ROWS,
        "n_instruments": N_INSTRUMENTS,
        "runs": RUNS,
        "warmups": WARMUPS,
        "results": results,
    }
    output = OUTPUT_DIR / "toolchain_benchmark.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"result_json={output}")


if __name__ == "__main__":
    main()
