import os
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
    formula = "groupby((univ([0, 1], [2, 3, 4, 5, 6, 7, 8]), open), close, cumsum(self_))"
    t_rows = int(10E3) #min(T_ROWS, 5_000_000)
    groups = 500
    open_data = (jnp.column_stack([jnp.arange(0, t_rows)]*N_INSTRUMENTS) // int(t_rows / groups)).astype(jnp.float64)
    close_data = jax.random.normal(jax.random.PRNGKey(53), (t_rows, N_INSTRUMENTS), dtype=jnp.float64)

    numba_engine = build_engine(formula)
    numba_data = {"open": np.asarray(open_data), "close": np.asarray(close_data)}
    jax_flat_data = (open_data, close_data)

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
