import json
import os
import subprocess
import sys
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    import polars as pl
except Exception:
    pl = None

from trading_dsl_engine.jax_flat.engine import compile_formula as compile_flat
from trading_dsl_engine.jax_new.engine import compile_formula as compile_new
from trading_dsl_engine import build_engine, run_batch_from_mapping

RUN_PERF = os.getenv("RUN_PERF_TESTS", "0") == "1"
T_ROWS = 1440 * 365 * 3
N_INSTRUMENTS = 9
N_RUNS = 10


def _stats(samples: list[float]) -> dict[str, float]:
    arr = np.asarray(samples, dtype=np.float64)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "p90": float(np.percentile(arr, 90)),
    }

def _bench(fn: Callable, *args, warmup_runs: int = 1) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(warmup_runs):
        out = fn(*args)

    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - t0)
    return _stats(samples)


def _make_scan_runner(runtime):

    def run(state, open_arr, close_arr):
        def step(carry, x):
            o, c = x
            next_state, out = runtime.tick(carry, o, c)
            return next_state, out

        return jax.lax.scan(step, state, (open_arr, close_arr))

    return jax.jit(run)


def _run_case(case_name: str, formula: str, batch_fn: Callable, key0: int, key1: int):
    open_data = jax.random.normal(jax.random.PRNGKey(key0), (T_ROWS, N_INSTRUMENTS), dtype=jnp.float64)
    close_data = jax.random.normal(jax.random.PRNGKey(key1), (T_ROWS, N_INSTRUMENTS), dtype=jnp.float64)

    runtime_new = compile_new(formula)
    runtime_flat = compile_flat(formula)
    state_new = runtime_new.init_state(N_INSTRUMENTS)
    state_flat = runtime_flat.init_state(N_INSTRUMENTS)

    jit_new_scan = _make_scan_runner(runtime_new)
    jit_flat_scan = _make_scan_runner(runtime_flat)
    jit_batch = jax.jit(batch_fn)

    jax.block_until_ready(jit_new_scan(state_new, open_data, close_data))
    jax.block_until_ready(jit_flat_scan(state_flat, open_data, close_data))
    jax.block_until_ready(jit_batch(open_data, close_data))

    new_stats = _bench(jit_new_scan, state_new, open_data, close_data)
    flat_stats = _bench(jit_flat_scan, state_flat, open_data, close_data)
    batch_stats = _bench(jit_batch, open_data, close_data)
    jax_flat_batch_stats = _bench(lambda x, y: jax.block_until_ready(compile_flat(formula).run_batch(x, y)), (open_data, close_data), None)

    print(f"{case_name}::jax_new {new_stats}")
    print(f"{case_name}::jax_flat {flat_stats}")
    print(f"{case_name}::jax_flat_batch {jax_flat_batch_stats}")
    print(f"{case_name}::batch {batch_stats}")

    new_out = jit_new_scan(state_new, open_data, close_data)[1]
    flat_out = jit_flat_scan(state_flat, open_data, close_data)[1]
    batch_out = jit_batch(open_data, close_data)
    np.testing.assert_allclose(np.asarray(new_out), np.asarray(flat_out), rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(np.asarray(new_out), np.asarray(batch_out), rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_nary_chain_jax_flat_scan_batch_vs_tick_scan():
    formula = (
        "xstd("
        "add("
        "xstd(add(abs(close), exp(fraction(abs(open))))), "
        "xstd(sub(mul(close, open), div(close, add(abs(open), 1.0))))"
        ")"
        ")"
    )
    open_data = jax.random.normal(jax.random.PRNGKey(22), (T_ROWS, N_INSTRUMENTS), dtype=jnp.float64)
    close_data = jax.random.normal(jax.random.PRNGKey(23), (T_ROWS, N_INSTRUMENTS), dtype=jnp.float64)
    runtime = compile_flat(formula)
    state = runtime.init_state(N_INSTRUMENTS)
    jit_tick_scan = _make_scan_runner(runtime)

    def scan_batch(open_arr, close_arr):
        return runtime.run_batch((open_arr, close_arr))[1]

    jax.block_until_ready(jit_tick_scan(state, open_data, close_data))
    jax.block_until_ready(scan_batch(open_data, close_data))

    tick_scan_stats = _bench(jit_tick_scan, state, open_data, close_data)
    scan_batch_stats = _bench(scan_batch, open_data, close_data)

    print(f"nary_chain::tick_scan {tick_scan_stats}")
    print(f"nary_chain::scan_batch {scan_batch_stats}")

    tick_out = jit_tick_scan(state, open_data, close_data)[1]
    batch_out = scan_batch(open_data, close_data)
    np.testing.assert_allclose(np.asarray(tick_out), np.asarray(batch_out), rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_fusion_formula_jax_new_vs_jax_flat_and_batch_lowerbound():
    formula = (
        "mul(mul(mul(cumsum(xs_sort(add(close, open))), xs_sort(add(close, open))), "
        "xs_sort(add(close, open))), xs_sort(add(close, open)))"
    )

    def run_batch_lowerbound(open_arr, close_arr):
        ranked = jnp.sort(open_arr + close_arr, axis=1)
        valid = jnp.isfinite(ranked)
        safe = jnp.where(valid, ranked, 0.0)
        cumulative = jnp.cumsum(safe, axis=0)
        cumsum_out = jnp.where(jnp.cumsum(valid.astype(jnp.int32), axis=0) > 0, cumulative, jnp.nan)
        return cumsum_out * ranked * ranked * ranked

    _run_case("fusion", formula, run_batch_lowerbound, key0=2, key1=3)


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_groupby_formula_jax_new_vs_jax_flat_and_batch_lowerbound():
    formula = "groupby((open,), close, cumsum(self_))"

    def run_batch_lowerbound(open_arr, close_arr):
        # simple Python baseline for perf floor in perf mode only
        t, n = close_arr.shape
        out = np.empty((t, n), dtype=np.float64)
        state = {}
        for i in range(t):
            for j in range(n):
                k = float(open_arr[i, j])
                if np.isnan(k):
                    key = "__nan__"
                else:
                    key = k
                prev = state.get(key, 0.0)
                v = float(close_arr[i, j])
                if np.isfinite(v):
                    cur = prev + v
                    state[key] = cur
                    out[i, j] = cur
                else:
                    out[i, j] = prev if key in state else np.nan
        return jnp.asarray(out)

    _run_case("groupby", formula, run_batch_lowerbound, key0=4, key1=5)


@pytest.mark.skipif(not RUN_PERF or pl is None, reason="set RUN_PERF_TESTS=1 and install polars")
def test_perf_groupby_univ_stateful_vs_polars_batch():
    formula = "groupby((univ([0, 1], [2, 3, 4, 5, 6, 7, 8]), open), close, cumsum(self_))"
    t_rows = min(T_ROWS, 20000)
    open_data = jnp.column_stack([jnp.arange(0, t_rows)]*N_INSTRUMENTS) // int(t_rows / 500)
    close_data = jax.random.normal(jax.random.PRNGKey(43), (t_rows, N_INSTRUMENTS), dtype=jnp.float64)
    jax_flat_data = (open_data, close_data)

    def run_flat_batch(formula, data):
        return jax.block_until_ready(formula.run_batch(data))[-1]

    o = np.asarray(open_data)
    c = np.asarray(close_data)
    t, n = c.shape
    rows = {"t": np.repeat(np.arange(t), n), "col": np.tile(np.arange(n), t), "open": o.reshape(-1), "close": c.reshape(-1)}
    df = pl.DataFrame(rows).lazy()

    def run_polars(df):
        df = df.with_columns(
            pl.when(pl.col("col") < 2).then(pl.lit(0)).otherwise(pl.lit(1)).alias("ug")
        )
        out = (
            df#.sort(["ug", "col", "open", "t"])
            .with_columns(pl.col("close").cum_sum().over(["ug", "open", "col"]).alias("out"))
            #.sort(["t", "col"])
        )
        return jnp.asarray(out.collect()["out"].to_numpy().reshape(t, n))

    runtime_flat = compile_flat(formula)
    flat_out = run_flat_batch(formula=runtime_flat, data=jax_flat_data)
    polars_out = run_polars(open_data, close_data)

    polars_stats = _bench(run_polars, df)
    print(f"groupby_univ::polars {polars_stats}")
    flat_stats = _bench(run_flat_batch, runtime_flat, jax_flat_data, warmup_runs=1)
    print(f"groupby_univ::jax_flat {flat_stats}")

    np.testing.assert_allclose(np.asarray(flat_out), np.asarray(polars_out), rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_groupby_univ_stateful_jax_flat_vs_numba():
    formula = "close + groupby((univ([0, 1], [2, 3, 4, 5, 6, 7, 8]), open), close, cumsum(self_))"
    t_rows = int(1E6)
    groups = 500
    open_data = (jnp.column_stack([jnp.arange(0, t_rows)] * N_INSTRUMENTS) // int(t_rows / groups)).astype(jnp.float64)
    close_data = jax.random.normal(jax.random.PRNGKey(53), (t_rows, N_INSTRUMENTS), dtype=jnp.float64)

    numba_engine = build_engine(formula)
    numba_data = {"open": np.asarray(open_data), "close": np.asarray(close_data)}
    jax_flat_data = (close_data, open_data)

    def run_numba_batch(data):
        return run_batch_from_mapping(numba_engine, data=data, out_path=None)

    def run_flat_batch(formula, data):
        return jax.block_until_ready(formula.run_batch(data))[-1]

    runtime_flat = compile_flat(formula)

    flat_out = run_flat_batch(formula=runtime_flat, data=jax_flat_data)
    numba_out = run_numba_batch(numba_data)

    flat_stats = _bench(run_flat_batch, runtime_flat, jax_flat_data, warmup_runs=1)
    print(f"groupby_univ::jax_flat {flat_stats}")
    numba_stats = _bench(run_numba_batch, numba_data, warmup_runs=1)
    print(f"groupby_univ::numba {numba_stats}")

    np.testing.assert_allclose(np.asarray(flat_out), np.asarray(numba_out), rtol=1e-10, atol=1e-10, equal_nan=True)

@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_groupby_univ_stateful_jax_vs_numba():
    from trading_dsl_engine.jax.engine import build_jax_engine, run_batch_from_mapping as run_batch_from_mapping_jax
    formula = "groupby((univ([0, 1], [2, 3, 4, 5, 6, 7, 8]), open), close, cumsum(self_))"
    t_rows = int(10E3) #min(T_ROWS, 5_000_000)
    groups = 500
    open_data = (jnp.column_stack([jnp.arange(0, t_rows)]*N_INSTRUMENTS) // int(t_rows / groups)).astype(jnp.float64)
    close_data = jax.random.normal(jax.random.PRNGKey(53), (t_rows, N_INSTRUMENTS), dtype=jnp.float64)

    numba_engine = build_engine(formula)
    numba_data = {"open": np.asarray(open_data), "close": np.asarray(close_data)}
    jax_engine = build_jax_engine(formula)

    def run_numba_batch(data):
        return run_batch_from_mapping(numba_engine, data=data, out_path=None)

    def run_flat_batch(data):
        return jax.block_until_ready(run_batch_from_mapping_jax(jax_engine, data=data, out_path=None))

    # runtime_flat = compile_flat(formula)

    jax_out = run_flat_batch(numba_data)
    numba_out = run_numba_batch(numba_data)

    jax_stats = _bench(run_flat_batch, numba_data, warmup_runs=1)
    print(f"groupby_univ::jax {jax_stats}")
    numba_stats = _bench(run_numba_batch, numba_data, warmup_runs=1)
    print(f"groupby_univ::numba {numba_stats}")

    np.testing.assert_allclose(np.asarray(jax_out), np.asarray(numba_out), rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_jax_flat_memmap_input_output_streaming_vs_existing():
    script = r"""
import json
import os
import sys
import tempfile
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.jax_flat import engine as jax_flat_engine


def rss_bytes():
    with open("/proc/self/statm", "r", encoding="utf-8") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def sample_peak(stop, peak):
    while not stop.is_set():
        peak[0] = max(peak[0], rss_bytes())
        time.sleep(0.002)


def make_inputs(workdir, shape):
    close_path = os.path.join(workdir, "close.memmap")
    open_path = os.path.join(workdir, "open.memmap")
    close = np.memmap(close_path, mode="w+", dtype=np.float64, shape=shape)
    open_ = np.memmap(open_path, mode="w+", dtype=np.float64, shape=shape)
    close[:] = 2.0
    open_[:] = 3.0
    close.flush()
    open_.flush()
    del close, open_
    return close_path, open_path


def run(mode):
    shape = (16384, 512)
    chunk_size = 512
    with tempfile.TemporaryDirectory() as workdir:
        close_path, open_path = make_inputs(workdir, shape)
        jax_flat_engine._BATCH_CHUNK_SIZE = chunk_size
        runtime = jax_flat_engine.compile_formula("close + open")
        warm = jnp.ones((chunk_size, shape[1]), dtype=jnp.float64)
        if mode == "streaming":
            runtime.run_batch({"open": warm, "close": warm}, out_path=True)
        else:
            jax.block_until_ready(runtime.run_batch({"open": warm, "close": warm})[1])

        baseline = rss_bytes()
        peak = [baseline]
        stop = threading.Event()
        sampler = threading.Thread(target=sample_peak, args=(stop, peak), daemon=True)
        sampler.start()
        t0 = time.perf_counter()
        try:
            close = np.memmap(close_path, mode="r", dtype=np.float64, shape=shape)
            open_ = np.memmap(open_path, mode="r", dtype=np.float64, shape=shape)
            if mode == "streaming":
                out_path = os.path.join(workdir, "out.memmap")
                _, out = runtime.run_batch({"open": open_, "close": close}, out_path=out_path)
                checksum = float(out[0, 0] + out[-1, -1])
                out.flush()
            else:
                close_jax = jnp.asarray(close)
                open_jax = jnp.asarray(open_)
                _, out = runtime.run_batch({"open": open_jax, "close": close_jax})
                jax.block_until_ready(out)
                checksum = float(out[0, 0] + out[-1, -1])
        finally:
            elapsed = time.perf_counter() - t0
            stop.set()
            sampler.join(timeout=1.0)

        return {
            "mode": mode,
            "elapsed_s": elapsed,
            "peak_delta_bytes": peak[0] - baseline,
            "checksum": checksum,
        }


print(json.dumps(run(sys.argv[1])))
"""
    env = os.environ.copy()
    pythonpath = os.path.abspath("src")
    env["PYTHONPATH"] = pythonpath + os.pathsep + env.get("PYTHONPATH", "")

    def run_mode(mode: str) -> dict[str, float]:
        proc = subprocess.run(
            [sys.executable, "-c", script, mode],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        return json.loads(proc.stdout.strip().splitlines()[-1])

    existing = run_mode("existing")
    streaming = run_mode("streaming")

    print(f"close_plus_open::existing_memmap_to_jax {existing}")
    print(f"close_plus_open::streaming_memmap_in_out {streaming}")

    assert existing["checksum"] == pytest.approx(10.0)
    assert streaming["checksum"] == pytest.approx(10.0)
    assert streaming["peak_delta_bytes"] < existing["peak_delta_bytes"]
