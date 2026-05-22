import os
import time
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trading_dsl_engine.jax_flat.engine import compile_formula as compile_flat
from trading_dsl_engine.jax_new.engine import compile_formula as compile_new

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


def _bench(fn: Callable, *args) -> dict[str, float]:
    samples: list[float] = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        out = fn(*args)
        jax.block_until_ready(out)
        samples.append(time.perf_counter() - t0)
    return _stats(samples)


def _make_scan_runner(runtime, tick_method_name: str):
    tick = getattr(runtime, tick_method_name)

    def run(state, open_arr, close_arr):
        def step(carry, x):
            o, c = x
            next_state, out = tick(carry, o, c)
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

    jit_new_scan = _make_scan_runner(runtime_new, "tick")
    jit_flat_scan = _make_scan_runner(runtime_flat, "tick_stream")
    jit_batch = jax.jit(batch_fn)

    jax.block_until_ready(jit_new_scan(state_new, open_data, close_data))
    jax.block_until_ready(jit_flat_scan(state_flat, open_data, close_data))
    jax.block_until_ready(jit_batch(open_data, close_data))

    new_stats = _bench(jit_new_scan, state_new, open_data, close_data)
    flat_stats = _bench(jit_flat_scan, state_flat, open_data, close_data)
    batch_stats = _bench(jit_batch, open_data, close_data)

    print(f"{case_name}::jax_new {new_stats}")
    print(f"{case_name}::jax_flat {flat_stats}")
    print(f"{case_name}::batch {batch_stats}")

    new_out = jit_new_scan(state_new, open_data, close_data)[1]
    flat_out = jit_flat_scan(state_flat, open_data, close_data)[1]
    batch_out = jit_batch(open_data, close_data)
    np.testing.assert_allclose(np.asarray(new_out), np.asarray(flat_out), rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(np.asarray(new_out), np.asarray(batch_out), rtol=1e-10, atol=1e-10, equal_nan=True)


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_jax_new_vs_jax_flat_and_batch_lowerbound():
    formula = "cumsum(xs_sort(add(close, open)))"

    def run_batch_lowerbound(open_arr, close_arr):
        ranked = jnp.sort(open_arr + close_arr, axis=1)
        valid = jnp.isfinite(ranked)
        safe = jnp.where(valid, ranked, 0.0)
        cumulative = jnp.cumsum(safe, axis=0)
        seen = jnp.cumsum(valid.astype(jnp.int32), axis=0) > 0
        return jnp.where(seen, cumulative, jnp.nan)

    _run_case("base", formula, run_batch_lowerbound, key0=0, key1=1)


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
