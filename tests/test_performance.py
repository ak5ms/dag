import cProfile
import os
import pstats
import time

import numpy as np
import pytest

from trading_dsl_engine import build_engine, run_batch_from_mapping


RUN_PERF = os.getenv("RUN_PERF_TESTS", "0") == "1"
T_1Y_MINUTES = 365 * 24 * 60
N_INSTRUMENTS = 16


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_in_memory_one_year_minutely():
    rng = np.random.default_rng(42)
    close = rng.lognormal(mean=0.0, sigma=0.03, size=(T_1Y_MINUTES, N_INSTRUMENTS)).astype(np.float64)
    open_ = rng.lognormal(mean=0.0, sigma=0.03, size=(T_1Y_MINUTES, N_INSTRUMENTS)).astype(np.float64)

    eng = build_engine("xs_rank(ewm(div(close, open), 21))")

    t0 = time.perf_counter()
    out = run_batch_from_mapping(eng, {"close": close, "open": open_})
    elapsed = time.perf_counter() - t0

    assert out.shape == (T_1Y_MINUTES, N_INSTRUMENTS)
    # Guardrail only; tune as needed for host.
    assert elapsed < 30.0


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_memmap_one_year_minutely(tmp_path):
    rng = np.random.default_rng(7)

    close_path = tmp_path / "close.dat"
    open_path = tmp_path / "open.dat"

    close = np.memmap(close_path, mode="w+", shape=(T_1Y_MINUTES, N_INSTRUMENTS), dtype=np.float64)
    open_ = np.memmap(open_path, mode="w+", shape=(T_1Y_MINUTES, N_INSTRUMENTS), dtype=np.float64)
    close[:] = rng.lognormal(mean=0.0, sigma=0.03, size=(T_1Y_MINUTES, N_INSTRUMENTS))
    open_[:] = rng.lognormal(mean=0.0, sigma=0.03, size=(T_1Y_MINUTES, N_INSTRUMENTS))
    close.flush()
    open_.flush()

    close_r = np.memmap(close_path, mode="r", shape=(T_1Y_MINUTES, N_INSTRUMENTS), dtype=np.float64)
    open_r = np.memmap(open_path, mode="r", shape=(T_1Y_MINUTES, N_INSTRUMENTS), dtype=np.float64)

    eng = build_engine("xs_rank(ewm(div(close, open), 21))")

    t0 = time.perf_counter()
    out = run_batch_from_mapping(eng, {"close": close_r, "open": open_r})
    elapsed = time.perf_counter() - t0

    assert out.shape == (T_1Y_MINUTES, N_INSTRUMENTS)
    assert elapsed < 45.0


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_profile_engine_vs_numpy_baseline_smoke():
    rng = np.random.default_rng(123)
    close = rng.uniform(1, 2, size=(4096, N_INSTRUMENTS)).astype(np.float64)
    open_ = rng.uniform(1, 2, size=(4096, N_INSTRUMENTS)).astype(np.float64)
    eng = build_engine("xs_rank(ewm(div(close, open), 21))")

    profiler = cProfile.Profile()
    profiler.enable()
    t0 = time.perf_counter()
    out = run_batch_from_mapping(eng, {"close": close, "open": open_}, chunk_size=1024)
    engine_elapsed = time.perf_counter() - t0
    profiler.disable()

    ratio = close / open_
    alpha = 2.0 / 22.0
    ewm = np.empty_like(ratio)
    ewm[0] = ratio[0]
    for t in range(1, ratio.shape[0]):
        ewm[t] = alpha * ratio[t] + (1.0 - alpha) * ewm[t - 1]

    t1 = time.perf_counter()
    baseline = np.empty_like(ewm)
    for t in range(ewm.shape[0]):
        order = np.argsort(ewm[t])
        baseline[t, order] = (np.arange(order.shape[0]) + 1) / order.shape[0]
    baseline_elapsed = time.perf_counter() - t1

    assert out.shape == baseline.shape
    assert np.isfinite(engine_elapsed)
    assert np.isfinite(baseline_elapsed)
    assert pstats.Stats(profiler).total_tt > 0.0
