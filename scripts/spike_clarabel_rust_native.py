from __future__ import annotations

import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "src/trading_dsl_engine/cpp_stream/optimizer/templates/direct_clarabel_instance.hpp.j2"
OUT = Path("/dev/shm/mpo_rust_native_spike")
ROWS = int(os.environ.get("MPO_NATIVE_ROWS", "5000"))
RUNS = int(os.environ.get("MPO_NATIVE_RUNS", "5"))
N_ASSETS = 3
MINUTE_US = 60_000_000.0


def fake_data():
    rng = np.random.default_rng(7)
    base = 1_800_000_000_000_000.0
    row = np.arange(ROWS)
    session_rows, open_rows = 1000, 980
    ts = base + row * MINUTE_US
    sid = row // session_rows
    s0 = base + sid * session_rows * MINUTE_US
    e0 = s0 + open_rows * MINUTE_US
    ns = s0 + session_rows * MINUTE_US
    ne = ns + open_rows * MINUTE_US
    open_ = (ts >= s0) & (ts < e0)
    tradable = np.broadcast_to(open_[:, None], (ROWS, N_ASSETS)).astype(float).copy()
    returns = rng.normal(scale=2e-4, size=(ROWS, N_ASSETS))
    returns[tradable == 0] = 0
    for reopen in range(session_rows, ROWS, session_rows):
        returns[reopen] *= np.sqrt(session_rows - open_rows + 1.0)
    def lanes(x):
        return np.broadcast_to(x[:, None], (ROWS, N_ASSETS)).astype(float).copy()
    return {
        "returns": np.ascontiguousarray(returns),
        "is_tradable_out0": tradable,
        "vw_halfspread_out0": rng.uniform(3e-5, 8e-5, size=(ROWS, N_ASSETS)),
        "_ev_ts": lanes(ts),
        "session_start0": lanes(s0),
        "session_end0": lanes(e0),
        "next_session_start0": lanes(ns),
        "next_session_end0": lanes(ne),
    }


def patch_no_refine(original: str, enabled: bool) -> str:
    if not enabled:
        return original
    anchor = "    settings_.presolve_enable = false;\n"
    return original.replace(anchor, anchor + "    settings_.iterative_refinement_enable = false;\n", 1)


def child():
    from examples import cpp_stream_mpo_one_pass as example
    from trading_dsl_engine.base.dsl import var
    from trading_dsl_engine.cpp_stream import compile_formula
    name = os.environ["NATIVE_CHILD"]
    data = fake_data()
    shutil.rmtree(ROOT / ".generated/cpp_stream_mpo_one_pass/clarabel", ignore_errors=True)
    t0 = time.perf_counter()
    runtime = compile_formula(list(example._formula(var("returns"))), data, n_instruments=N_ASSETS)
    compile_s = time.perf_counter() - t0
    warm = runtime.run(out_path=OUT / f"{name}_warm.npy")
    times = []
    result = warm
    for i in range(RUNS):
        result = runtime.run(out_path=OUT / f"{name}_{i}.npy")
        times.append(result.seconds)
    values = result.load()
    np.save(OUT / f"{name}_weights.npy", np.asarray(values[3]))
    print(
        f"RESULT name={name} compile_s={compile_s:.6f} warmup_s={warm.seconds:.6f} "
        f"mean_s={statistics.mean(times):.6f} median_s={statistics.median(times):.6f} "
        f"rows_per_s={ROWS/statistics.mean(times):.3f}", flush=True)


def build_variant(name: str, rustflags: str | None):
    from trading_dsl_engine.cpp_stream.optimizer import build_current_clarabel
    cache = OUT / f"clarabel_{name}"
    old = os.environ.get("RUSTFLAGS")
    try:
        if rustflags is None:
            os.environ.pop("RUSTFLAGS", None)
        else:
            os.environ["RUSTFLAGS"] = rustflags
        t0 = time.perf_counter()
        paths = build_current_clarabel(cache_dir=cache, force=True)
        print(f"BUILD name={name} seconds={time.perf_counter()-t0:.6f} rustflags={rustflags!r}", flush=True)
        return paths
    finally:
        if old is None:
            os.environ.pop("RUSTFLAGS", None)
        else:
            os.environ["RUSTFLAGS"] = old


def parent():
    OUT.mkdir(parents=True, exist_ok=True)
    generic = build_variant("generic", None)
    native = build_variant("native", "-C target-cpu=native")
    original = TEMPLATE.read_text()
    configs = [
        ("generic_default", generic, False),
        ("native_default", native, False),
        ("generic_no_refine", generic, True),
        ("native_no_refine", native, True),
    ]
    try:
        for name, paths, no_refine in configs:
            TEMPLATE.write_text(patch_no_refine(original, no_refine))
            env = os.environ.copy()
            env["NATIVE_CHILD"] = name
            env["CLARABEL_INCLUDE_DIR"] = str(paths.include_dir)
            env["CLARABEL_STATIC_LIBRARY"] = str(paths.static_library)
            p = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(p.stdout, end="", flush=True)
            if p.returncode:
                raise RuntimeError((name, p.returncode))
        base = np.load(OUT / "generic_default_weights.npy")
        for name in ("native_default", "generic_no_refine", "native_no_refine"):
            x = np.load(OUT / f"{name}_weights.npy")
            print(f"WEIGHT_DIFF name={name} max_abs={np.max(np.abs(x-base)):.12g} rms={np.sqrt(np.mean((x-base)**2)):.12g}", flush=True)
    finally:
        TEMPLATE.write_text(original)


if __name__ == "__main__":
    child() if os.environ.get("NATIVE_CHILD") else parent()
