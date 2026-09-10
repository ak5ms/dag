"""Generate deterministic full-terminal inputs for the GP timing benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import numpy as np

from flows.riskminer.semantics import INPUTDATA_ALPHA_KEYS


ROWS = int(os.environ.get("GP_SYNTHETIC_ROWS", "5000000"))
N_INSTRUMENTS = int(os.environ.get("GP_SYNTHETIC_INSTRUMENTS", "9"))
CHUNK_ROWS = int(os.environ.get("GP_SYNTHETIC_CHUNK_ROWS", "100000"))
SEED = int(os.environ.get("GP_SYNTHETIC_SEED", "42"))
OUTPUT_DIR = Path(
    os.environ.get(
        "GP_SYNTHETIC_OUTPUT_DIR",
        "/tmp/gp-alpha-search-data",
    )
)


def _open_base(name: str):
    return np.lib.format.open_memmap(
        OUTPUT_DIR / f"_base_{name}.npy",
        mode="w+",
        dtype=np.float64,
        shape=(ROWS, N_INSTRUMENTS),
        fortran_order=False,
    )


def _link(name: str, base: str) -> None:
    target = OUTPUT_DIR / f"{name}.npy"
    target.unlink(missing_ok=True)
    os.link(OUTPUT_DIR / f"_base_{base}.npy", target)


def main() -> None:
    if ROWS <= 0 or N_INSTRUMENTS <= 0 or CHUNK_ROWS <= 0:
        raise ValueError("rows, instruments, and chunk rows must be positive")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()

    price = _open_base("price")
    volume = _open_base("volume")
    halfspread = _open_base("halfspread")
    tradable = _open_base("tradable")
    event_ts = _open_base("event_ts")
    session_start = _open_base("session_start")
    session_end = _open_base("session_end")
    wdte = _open_base("wdte")

    last_price = np.linspace(90.0, 110.0, N_INSTRUMENTS)
    epoch = 1_700_000_000_000_000.0
    minute_us = 60_000_000.0
    session_minutes = 1_440
    open_minutes = 1_380

    for start in range(0, ROWS, CHUNK_ROWS):
        stop = min(ROWS, start + CHUNK_ROWS)
        width = stop - start
        row_number = np.arange(start, stop, dtype=np.int64)
        minute = row_number % session_minutes
        day = row_number // session_minutes

        returns = np.clip(
            rng.normal(
                0.0,
                3.5e-4,
                size=(width, N_INSTRUMENTS),
            ),
            -0.02,
            0.02,
        )
        log_prices = np.log(last_price) + np.cumsum(returns, axis=0)
        current_price = np.exp(log_prices)
        price[start:stop] = current_price
        last_price = current_price[-1]

        volume[start:stop] = rng.lognormal(
            mean=7.0,
            sigma=0.65,
            size=(width, N_INSTRUMENTS),
        )
        halfspread[start:stop] = np.clip(
            4.0e-4
            + rng.normal(
                0.0,
                7.5e-5,
                size=(width, N_INSTRUMENTS),
            ),
            5.0e-5,
            2.0e-3,
        )

        open_row = minute < open_minutes
        tradable[start:stop] = open_row[:, None]

        current_ts = epoch + row_number * minute_us
        current_start = epoch + day * session_minutes * minute_us
        current_end = current_start + open_minutes * minute_us
        event_ts[start:stop] = current_ts[:, None]
        session_start[start:stop] = current_start[:, None]
        session_end[start:stop] = current_end[:, None]

        # Descend through five trading days and reset. Positive differences at
        # reset exercise the rollover branch in RollRets.
        current_wdte = 5.0 - (day % 5)
        wdte[start:stop] = current_wdte[:, None]

        if stop == ROWS or stop % 1_000_000 == 0:
            print(
                f"generated_rows={stop:,}/{ROWS:,}",
                flush=True,
            )

    arrays = (
        price,
        volume,
        halfspread,
        tradable,
        event_ts,
        session_start,
        session_end,
        wdte,
    )
    for array in arrays:
        array.flush()
    del arrays
    del price, volume, halfspread, tradable
    del event_ts, session_start, session_end, wdte

    for name in INPUTDATA_ALPHA_KEYS:
        if name == "_ev_ts":
            base = "event_ts"
        elif name in {"session_start0", "next_session_start0"}:
            base = "session_start"
        elif name in {"session_end0", "next_session_end0"}:
            base = "session_end"
        elif name == "wdte_out0":
            base = "wdte"
        elif name == "vw_halfspread_out0":
            base = "halfspread"
        elif name == "is_tradable_out0":
            base = "tradable"
        elif (
            name.startswith("volume")
            or name.startswith("trade_cross_pct")
        ):
            base = "volume"
        else:
            base = "price"
        _link(name, base)

    # RollRets additionally references the deferred contract.
    _link("mp_out1.close", "price")
    _link("is_tradable_out1", "tradable")

    elapsed = time.perf_counter() - started
    payload = ROWS * N_INSTRUMENTS * np.dtype(np.float64).itemsize
    manifest = {
        "rows": ROWS,
        "n_instruments": N_INSTRUMENTS,
        "dtype": "float64",
        "seed": SEED,
        "chunk_rows": CHUNK_ROWS,
        "unique_base_arrays": 8,
        "bytes_per_base_array": payload,
        "unique_payload_bytes": payload * 8,
        "logical_source_count": len(INPUTDATA_ALPHA_KEYS) + 2,
        "generation_seconds": elapsed,
    }
    (OUTPUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
