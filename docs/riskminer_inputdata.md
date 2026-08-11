# Running RiskMiner on `InputData`

From the repository root:

```bash
PYTHONPATH=src python scripts/run_riskminer_inputdata.py
```

The script loads `InputData` directly, materializes `RollRets().roll_rets()` and
the current `flows.riskmodel` volatility definition through `cpp_stream`, then
runs exact-depth search from depth 1 through depth 8.

## Search inputs

Only the 69 fields in `INPUTDATA_ALPHA_KEYS` are exposed to the formula grammar.
Extra `InputData` fields remain available for constructing evaluation targets but
cannot be selected by MCTS. The semantic catalog covers:

- ask and bid prices at levels 0 through 9;
- ask, bid, and mid OHLC candles;
- ask- and bid-side level quantities;
- trade volume, trade-price VWAP, and trade-clock mid VWAP;
- volume-weighted half spread;
- trade-cross count and signed quantity-like aggregates;
- tradability;
- event/session timestamps; and
- weekdays to expiry.

Descriptor tags such as `ask`, `level_0`, and `high` do not create arithmetic
compatibility. Value domains such as `price`, `quantity`, `timestamp`, and
`dimensionless` do.

## Workstation-safe defaults

The default is intentionally diagnostic because every valid terminal episode
performs an exact validation Ridge-pool trial:

```text
rows                    100,000
max depth               8
max RPN tokens          30
iterations per depth    1
MCTS simulations        8
rollouts per expansion  1
candidate batch         8
pool capacity           100
```

The paper's 200-search-cycle setting can be selected with:

```bash
RISKMINER_SIMULATIONS=200 \
PYTHONPATH=src python scripts/run_riskminer_inputdata.py
```

## Useful environment variables

```text
RISKMINER_INPUT_GLOB
RISKMINER_ROWS                 # 0 means every row
RISKMINER_MAX_DEPTH
RISKMINER_MAX_TOKENS
RISKMINER_ITERATIONS_PER_DEPTH
RISKMINER_SIMULATIONS
RISKMINER_ROLLOUTS
RISKMINER_EVALUATION_BATCH
RISKMINER_POOL_CAPACITY
RISKMINER_POOL_MIN_IMPROVEMENT
RISKMINER_POOL_IMPORTANCE      # mean_abs or final_abs
RISKMINER_RIDGE_RECOMPUTE_EVERY
RISKMINER_TRAIN_FRACTION
RISKMINER_VALIDATION_FRACTION
RISKMINER_QUANTILE_CDF
RISKMINER_QUANTILE_LEARNING_RATE
RISKMINER_POLICY_LEARNING_RATE
RISKMINER_POLICY_EPOCHS
RISKMINER_POLICY_BATCH_SIZE
RISKMINER_EXPLORATION
RISKMINER_ROLLOUT_END_PROBABILITY
RISKMINER_THREADS
RISKMINER_OUTPUT_DIR
RISKMINER_REUSE_DERIVED
RISKMINER_RESUME_POLICY
```

`RISKMINER_RIDGE_RECOMPUTE_EVERY=1` preserves every-row coefficient solves.
Larger values use the periodic recomputation support from
`agent/cpp-stream-operators`.

## Progress output

The script emits flushed JSON records for:

```text
input loading and semantic families
derived formula compilation and execution
MCTS iteration start/end
orthogonal reward compilation and execution
pool compilation and execution
pool admission and eviction
rank-saturation warning
quantile and policy updates
policy checkpoint paths
final test evaluation
```

It also prints the complete current Ridge-pool tree after every mining iteration.
The final machine-readable report is written to:

```text
/tmp/riskminer-inputdata/riskminer_inputdata_report.json
```
