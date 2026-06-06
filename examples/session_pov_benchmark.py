"""Synthetic high-throughput session POV/roll-return benchmark for jax_flat.

This mirrors the CME-style session-clock formula: explicit epoch session starts and
ends, tradability holes from 17:00-18:00 Central-session phase, weekend zeros,
per-instrument basis profile estimates, compact future session basis mass, and
grouped cumulative seen volume.

Usage:
    PYTHONPATH=src python examples/session_pov_benchmark.py --rows 1000000 --root roll
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import *
from trading_dsl_engine.jax_flat import stateless
from trading_dsl_engine.jax_flat.engine import compile_formula

MIN_US = 60_000_000.0
DAY_US = 86_400_000_000.0
N_INSTRUMENTS = 9
N_BASIS = 6
H = 1440


def _formula(root: str):
    def _in_current_session(ts, session_start, session_end):
        return (
            jnp.isfinite(ts)
            & jnp.isfinite(session_start)
            & jnp.isfinite(session_end)
            & (session_end > session_start)
            & (ts >= session_start)
            & (ts < session_end)
        )

    volume_for_fit = stateless(
        lambda volume, ts, session_start, session_end: jnp.where(
            _in_current_session(ts, session_start, session_end),
            jnp.maximum(jnp.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0), 0.0),
            jnp.nan,
        ),
        output_kind="vector",
        output_width=1,
        name="volume_for_fit_session",
    )
    volume_for_seen = stateless(
        lambda volume, ts, session_start, session_end, is_tradable: jnp.where(
            _in_current_session(ts, session_start, session_end) & jnp.isfinite(is_tradable) & (is_tradable == 1.0),
            jnp.maximum(jnp.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0), 0.0),
            0.0,
        ),
        output_kind="vector",
        output_width=1,
        name="volume_for_seen_session",
    )
    nonnegative = stateless(
        lambda x: jnp.maximum(jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0),
        output_kind="vector",
        output_width=1,
        name="nonnegative",
    )
    pct_seen = stateless(
        lambda seen, forecast, ts, session_start: jnp.where(
            jnp.isfinite(ts) & jnp.isfinite(session_start) & (ts >= session_start) & ((seen + forecast) > 0.0),
            seen / (seen + forecast),
            jnp.nan,
        ),
        output_kind="vector",
        output_width=1,
        name="pct_seen_session_volume",
    )

    def pct_change(x):
        return x / shift(x, 1, 1) - 1.0

    def mask(w, tradable_mask):
        return ffill(where(tradable_mask != 1, float("nan"), w), 1)

    ts = var("ev_ts")
    session_start = var("session_start")
    session_end = var("session_end")
    volume = var("volume")

    features = rbf_basis(ts, session_start, session_end, N_BASIS)
    fit_y = volume_for_fit(volume, ts, session_start, session_end)
    beta = get_beta(InstrumentBasisMean(features, fit_y, 1.0, 21 * H))
    forecast = nonnegative(einsum(beta, future_rbf_basis_sum(ts, session_start, session_end, N_BASIS, H), "nf,nf->n"))
    seen = groupby(
        (session_start,),
        volume_for_seen(volume, ts, session_start, session_end, var("is_tradable0")),
        cumsum(self_),
    )
    pov = pct_seen(seen, forecast, ts, session_start)

    if root == "pov":
        return pov

    w0 = where(var("wdte") == 1, pov, 1.0)
    w1 = where(var("wdte") == 1, 1.0 - pov, 0.0)
    return einsum(
        cat(shift(mask(w0, var("is_tradable0")), 1, 1), shift(mask(w1, var("is_tradable1")), 1, 1)),
        cat(pct_change(mask(var("vwap0"), var("is_tradable0"))), pct_change(mask(var("vwap1"), var("is_tradable1")))),
        "nf,nf->n",
    )


def _make_memmaps(workdir: str, rows: int):
    base = 1_700_000_000_000_000.0
    t = np.arange(rows, dtype=np.float64)[:, None]
    ev_ts = base + t * MIN_US
    day = np.floor(t / H)
    session_start = base + day * DAY_US
    session_end = session_start + DAY_US
    minute = np.mod(t, H)
    weekday = (np.mod(day.astype(np.int64) + 2, 7) < 5).astype(np.float64)
    tradable = ((minute >= 60) & (minute < 1380)).astype(np.float64) * weekday
    columns = np.arange(N_INSTRUMENTS, dtype=np.float64)[None, :]
    volume = np.maximum(100.0 + 20.0 * np.sin(2.0 * np.pi * minute / H) + columns, 0.0) * tradable
    vwap0 = 100.0 + 0.001 * t + 0.01 * columns
    vwap1 = 101.0 + 0.0011 * t + 0.01 * columns
    arrays = {
        "ev_ts": ev_ts * np.ones((1, N_INSTRUMENTS)),
        "session_start": session_start * np.ones((1, N_INSTRUMENTS)),
        "session_end": session_end * np.ones((1, N_INSTRUMENTS)),
        "volume": volume,
        "is_tradable0": tradable * np.ones((1, N_INSTRUMENTS)),
        "is_tradable1": tradable * np.ones((1, N_INSTRUMENTS)),
        "wdte": (np.mod(day, 5) == 0).astype(np.float64) * np.ones((1, N_INSTRUMENTS)),
        "vwap0": vwap0 * np.ones((1, N_INSTRUMENTS)),
        "vwap1": vwap1 * np.ones((1, N_INSTRUMENTS)),
    }
    out = {}
    for name, values in arrays.items():
        path = os.path.join(workdir, f"{name}.dat")
        mapped = np.memmap(path, mode="w+", dtype=np.float64, shape=values.shape)
        mapped[:] = values
        mapped.flush()
        del mapped
        out[name] = np.memmap(path, mode="r", dtype=np.float64, shape=values.shape)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1_000_000)
    parser.add_argument("--root", choices=("pov", "roll"), default="roll")
    parser.add_argument("--chunk-size", type=int, default=None)
    args = parser.parse_args()

    if args.chunk_size is not None:
        from trading_dsl_engine.jax_flat import engine as jax_flat_engine

        jax_flat_engine._BATCH_CHUNK_SIZE = args.chunk_size

    with tempfile.TemporaryDirectory() as workdir:
        data = _make_memmaps(workdir, args.rows)
        runtime = compile_formula(_formula(args.root))
        out_path = os.path.join(workdir, "out.dat")
        start = time.perf_counter()
        state, out = runtime.run_batch(data, out_path=out_path)
        jax.block_until_ready(state)
        elapsed = time.perf_counter() - start
        print(
            f"root={args.root} rows={args.rows} shape={out.shape} elapsed_s={elapsed:.3f} "
            f"head_nanmean={float(np.nanmean(out[:min(args.rows, 10000)])):.12g}"
        )


if __name__ == "__main__":
    main()
