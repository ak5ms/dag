# RiskMiner on InputData

Run from the repository root:

```bash
PYTHONPATH=src python scripts/run_riskminer_inputdata.py
```

The script loads `InputData(nrows=None, idx=None)`, keeps all extra InputData
fields available to `cpp_stream`, and exposes only the 69 user-approved alpha
fields in `INPUTDATA_ALPHA_KEYS` to MCTS. Evaluation-only data (`roll_rets`,
`vol`, `hs`, `is_tradable`) is never inserted into the alpha vocabulary.

## Semantic families

The terminal graph is defined by `inputdata_alpha_terminal_metadata()`:

- `ap0_out0` ... `ap9_out0`: sampled ask-level prices.
- `bp0_out0` ... `bp9_out0`: sampled bid-level prices.
- `ap_out0.*`, `bp_out0.*`: ask/bid OHLC quote prices.
- `mp_out0.*`: mid-price OHLC.
- `volume_a*`, `volume_b*`: sampled/reset `VolumeAtPx` level quantities.
- `volume_out0`: reset trade-quantity sum.
- `vwap_out0`: trade-price VWAP.
- `vwap_mp_out0`: trade-clock mid-price VWAP.
- `vw_halfspread_out0`: volume-weighted `(ask-bid)/(ask+bid)`, dimensionless.
- `trade_cross_pct_out0.first/last/min/max/sum`: quantity-like because the
  source is `TradeQty * ((trade_price-bid)/(ask-bid))`.
- `trade_cross_pct_out0.count`: count-like/dimensionless.
- `_ev_ts`, session starts/ends: microsecond timestamps.
- `wdte_out0`: weekdays-to-expiry horizon.
- `is_tradable_out0`: boolean 0/1.

Descriptor tags such as ask/bid/level/open/high do not make otherwise
incompatible values add/sub compatible. Timestamp subtraction produces a
microsecond duration, so the grammar can construct formulas such as:

```text
(_ev_ts - session_start0) / (session_end0 - session_start0)
```

while rejecting timestamp + timestamp.

## Main environment settings

```text
RISKMINER_ROWS=500000
RISKMINER_MAX_DEPTH=6
RISKMINER_ROUNDS_PER_DEPTH=1
RISKMINER_SIMULATIONS=64
RISKMINER_ROLLOUTS=4
RISKMINER_EVALUATION_BATCH=16
RISKMINER_ARCHIVE_SIZE=256
RISKMINER_POOL_SHORTLIST=3
RISKMINER_TARGET_POOL_SIZE=12
RISKMINER_RIDGE_RECOMPUTE_EVERY=1
RISKMINER_THREADS=0
RISKMINER_HEARTBEAT_SECONDS=5
RISKMINER_OUTPUT_DIR=/tmp/riskminer-inputdata
RISKMINER_REUSE_DERIVED=0
```

`RISKMINER_ROWS=0` uses all rows. `RISKMINER_RIDGE_RECOMPUTE_EVERY=1`
preserves the every-row Ridge behavior; larger values use the periodic Ridge
recomputation support from `agent/cpp-stream-operators`.

The search is progressive by exact expression depth. Candidate formulas are
scored individually inside MCTS; only the shortlist is appended to the current
root-level Ridge pool for exact additive pool-Sharpe testing.

## Logging

The script prints line-oriented JSON events for:

- InputData load and semantic families.
- `roll_rets`/`vol` compilation and execution, with heartbeats.
- Each MCTS round.
- Every native candidate batch compilation/execution.
- Top archive formulas.
- Ridge shortlist trials, with heartbeats.
- Accepted alpha and current root-level Ridge pool.

The final machine-readable report is written to
`$RISKMINER_OUTPUT_DIR/riskminer_inputdata_report.json`.
