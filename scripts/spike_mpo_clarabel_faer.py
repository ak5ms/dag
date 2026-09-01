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
OUT = Path("/dev/shm/mpo_faer_spike")
ROWS = int(os.environ.get("MPO_FAER_ROWS", "5000"))
RUNS = int(os.environ.get("MPO_FAER_RUNS", "5"))
MINUTE_US = 60_000_000.0
N_ASSETS = 3
CPP_COMMIT = "0de6259a3edfd5cc041ec42b2148599ce63e73cb"
RS_TAG = "v0.11.1"


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


def build_faer():
    from trading_dsl_engine.cpp_stream.optimizer.clarabel_native import _patch_clarabel_allocation_free_timers
    root = OUT / "faer_clarabel"
    if root.exists():
        shutil.rmtree(root)
    cpp, rs = root / "Clarabel.cpp", root / "Clarabel.rs"
    subprocess.run(["git", "clone", "https://github.com/oxfordcontrol/Clarabel.cpp.git", str(cpp)], check=True)
    subprocess.run(["git", "checkout", CPP_COMMIT], cwd=cpp, check=True)
    subprocess.run(["git", "clone", "--depth", "1", "--branch", RS_TAG, "https://github.com/oxfordcontrol/Clarabel.rs.git", str(rs)], check=True)
    target = cpp / "Clarabel.rs"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(rs, target, ignore=shutil.ignore_patterns(".git"))
    _patch_clarabel_allocation_free_timers(target)
    subprocess.run([
        "cargo", "build", "--release",
        "--manifest-path", str(cpp / "rust_wrapper" / "Cargo.toml"),
        "--features", "clarabel/faer-sparse",
    ], check=True)
    native = root / "native"
    include = native / "include"
    libdir = native / "lib"
    include.mkdir(parents=True)
    libdir.mkdir(parents=True)
    shutil.copytree(cpp / "include", include, dirs_exist_ok=True)
    shutil.copy2(cpp / "rust_wrapper" / "target" / "release" / "libclarabel_c.a", libdir / "libclarabel_c.a")
    return include, libdir / "libclarabel_c.a"


def patch_template(original: str, *, faer: bool, no_refine: bool):
    text = original
    if faer:
        text = text.replace("#pragma once\n", "#pragma once\n#define FEATURE_FAER_SPARSE 1\n", 1)
        anchor = "    settings_.presolve_enable = false;\n"
        text = text.replace(anchor, anchor + "    settings_.direct_solve_method = FAER;\n", 1)
    if no_refine:
        anchor = "    settings_.presolve_enable = false;\n"
        text = text.replace(anchor, anchor + "    settings_.iterative_refinement_enable = false;\n", 1)
    return text


def child():
    from examples import cpp_stream_mpo_one_pass as example
    from trading_dsl_engine.base.dsl import var
    from trading_dsl_engine.cpp_stream import compile_formula
    name = os.environ["FAER_CHILD"]
    data = fake_data()
    shutil.rmtree(ROOT / ".generated/cpp_stream_mpo_one_pass/clarabel", ignore_errors=True)
    t0 = time.perf_counter()
    runtime = compile_formula(list(example._formula(var("returns"))), data, n_instruments=N_ASSETS)
    compile_s = time.perf_counter() - t0
    warm = runtime.run(out_path=OUT / f"{name}_warm.npy")
    ts = []
    result = warm
    for i in range(RUNS):
        result = runtime.run(out_path=OUT / f"{name}_{i}.npy")
        ts.append(result.seconds)
    values = result.load()
    np.save(OUT / f"{name}_weights.npy", np.asarray(values[3]))
    print(
        f"RESULT name={name} compile_s={compile_s:.6f} warmup_s={warm.seconds:.6f} "
        f"mean_s={statistics.mean(ts):.6f} median_s={statistics.median(ts):.6f} "
        f"rows_per_s={ROWS/statistics.mean(ts):.3f}", flush=True
    )


def parent():
    OUT.mkdir(parents=True, exist_ok=True)
    from examples import cpp_stream_mpo_one_pass as example
    t0 = time.perf_counter()
    baseline_paths = example._clarabel()
    print(f"BASELINE_BUILD {time.perf_counter()-t0:.6f}", flush=True)
    t0 = time.perf_counter()
    faer_include, faer_lib = build_faer()
    print(f"FAER_BUILD {time.perf_counter()-t0:.6f}", flush=True)
    original = TEMPLATE.read_text()
    configs = [
        ("baseline", False, False, baseline_paths.include_dir, baseline_paths.static_library),
        ("baseline_no_refine", False, True, baseline_paths.include_dir, baseline_paths.static_library),
        ("faer", True, False, faer_include, faer_lib),
        ("faer_no_refine", True, True, faer_include, faer_lib),
    ]
    try:
        for name, faer, no_refine, include, lib in configs:
            TEMPLATE.write_text(patch_template(original, faer=faer, no_refine=no_refine))
            env = os.environ.copy()
            env["FAER_CHILD"] = name
            env["CLARABEL_INCLUDE_DIR"] = str(include)
            env["CLARABEL_STATIC_LIBRARY"] = str(lib)
            p = subprocess.run([sys.executable, str(Path(__file__).resolve())], cwd=ROOT, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            print(p.stdout, end="", flush=True)
            if p.returncode:
                print(f"FAILED name={name} rc={p.returncode}", flush=True)
        base = np.load(OUT / "baseline_weights.npy")
        for name in ("baseline_no_refine", "faer", "faer_no_refine"):
            path = OUT / f"{name}_weights.npy"
            if path.exists():
                x = np.load(path)
                print(f"WEIGHT_DIFF name={name} max_abs={np.max(np.abs(x-base)):.12g} rms={np.sqrt(np.mean((x-base)**2)):.12g}", flush=True)
    finally:
        TEMPLATE.write_text(original)


if __name__ == "__main__":
    child() if os.environ.get("FAER_CHILD") else parent()
