from __future__ import annotations

import os
import threading
import time

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.jax_flat import engine as jax_flat_engine


def _rss_bytes() -> int:
    with open("/proc/self/statm", "r", encoding="utf-8") as fh:
        resident_pages = int(fh.read().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def _sample_peak_rss(stop: threading.Event, peak: list[int]) -> None:
    while not stop.is_set():
        peak[0] = max(peak[0], _rss_bytes())
        time.sleep(0.002)


def test_run_batch_accepts_mapping_by_input_name_not_formula_order():
    runtime = jax_flat_engine.compile_formula("close - open")
    close = jnp.array([[10.0, 20.0], [30.0, 40.0]], dtype=jnp.float64)
    open_ = jnp.array([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float64)

    _, out = runtime.run_batch({"open": open_, "close": close})

    np.testing.assert_allclose(np.asarray(out), np.asarray(close - open_))


def test_jnp_asarray_memmap_materializes_full_input(tmp_path):
    # This profiles the tempting simpler approach directly: handing an entire
    # memmap to JAX. On the CPU backend this creates a JAX buffer for the full
    # array, so the runtime must keep using bounded input chunks for memmaps.
    shape = (4096, 512)
    path = tmp_path / "direct.memmap"
    mapped = np.memmap(path, mode="w+", shape=shape, dtype=np.float64)
    mapped[:] = 1.0
    mapped.flush()
    del mapped

    mapped = np.memmap(path, mode="r", shape=shape, dtype=np.float64)
    jax.block_until_ready(jnp.asarray(np.zeros((1, 1), dtype=np.float64)))
    baseline = _rss_bytes()

    arr = jnp.asarray(mapped)
    jax.block_until_ready(arr)
    after = _rss_bytes()

    assert after - baseline >= mapped.nbytes // 2


def test_run_batch_streams_memmap_mapping_inputs_in_chunks(tmp_path, monkeypatch):
    n_steps = 4096
    n_instruments = 256
    chunk_size = 128
    close_path = tmp_path / "close.memmap"
    open_path = tmp_path / "open.memmap"

    close = np.memmap(close_path, mode="w+", shape=(n_steps, n_instruments), dtype=np.float64)
    open_ = np.memmap(open_path, mode="w+", shape=(n_steps, n_instruments), dtype=np.float64)
    rows = np.arange(n_steps, dtype=np.float64)[:, None]
    cols = np.arange(n_instruments, dtype=np.float64)[None, :]
    close[:] = rows + cols
    open_[:] = rows - cols
    close.flush()
    open_.flush()
    del close, open_

    close_r = np.memmap(close_path, mode="r", shape=(n_steps, n_instruments), dtype=np.float64)
    open_r = np.memmap(open_path, mode="r", shape=(n_steps, n_instruments), dtype=np.float64)

    runtime = jax_flat_engine.compile_formula("close + open")
    monkeypatch.setattr(jax_flat_engine, "_BATCH_CHUNK_SIZE", chunk_size)

    original_asarray = jax_flat_engine.jnp.asarray
    memmap_shapes_seen: list[tuple[int, ...]] = []

    def guarded_asarray(value, *args, **kwargs):
        if isinstance(value, np.memmap):
            memmap_shapes_seen.append(value.shape)
            assert value.shape[0] <= chunk_size
            assert value.shape != (n_steps, n_instruments)
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(jax_flat_engine.jnp, "asarray", guarded_asarray)

    runtime.run_batch({"open": open_r[:chunk_size], "close": close_r[:chunk_size]})
    memmap_shapes_seen.clear()
    baseline = _rss_bytes()
    peak = [baseline]
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_peak_rss, args=(stop, peak), daemon=True)
    sampler.start()
    try:
        _, out = runtime.run_batch({"open": open_r, "close": close_r})
        np.asarray(out).sum()  # Force host output realization before stopping the sampler.
    finally:
        stop.set()
        sampler.join(timeout=1.0)

    assert memmap_shapes_seen
    assert max(shape[0] for shape in memmap_shapes_seen) <= chunk_size
    input_bytes = close_r.nbytes + open_r.nbytes
    assert peak[0] - baseline < input_bytes + out.nbytes * 2
    np.testing.assert_allclose(out[0], np.zeros(n_instruments, dtype=np.float64))
    np.testing.assert_allclose(out[-1], np.full(n_instruments, 2.0 * (n_steps - 1)))
