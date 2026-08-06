# RiskMiner-style alpha search over `cpp_stream`

This checkpoint adds a runnable first stage of the RiskMiner architecture on top of
`trading_dsl_engine.cpp_stream`.

## Included

- Independent weak market semantic types with multiple tags per node.
- Requested terminals: `ap0`, `bp0`, `av0`, `bv0`, `volume`, `vwap`, `open`,
  `high`, `low`, `close`, and `soft_side_wavg`.
- Algorithmic type-relation closure and generic arithmetic transfer rules.
- Typed Reverse Polish Notation state and legal-action masks.
- Canonical expression and partial-state keys.
- PUCT MCTS with progressive widening, virtual visits, multiple rollouts per
  expansion, dense formula rewards, deterministic seeds, and a fixed-size archive.
- Batched native candidate scoring through `trading_dsl_engine.cpp_stream`.
- Native Ridge pool construction and final pool Sharpe.
- Reduced synthetic market-data benchmark and focused CI.

## Individual score

For each candidate alpha `w`, the native batch formula computes exactly:

```python
pnl = shift(w, 1, 1).mul(roll_rets).sum(axis=1)
score = pnl.mean(axis=0) / pnl.std(axis=0, ddof=0)
```

A batch of candidates is represented as an instrument-by-candidate row tensor.
Instrument PnL is reduced on axis 1 and time is reduced on axis 0. The output is
final-only: the native runner writes one score per candidate, not a time-sized
candidate matrix.

## Simulation

One MCTS simulation consists of:

1. PUCT selection from the root.
2. Expansion of one action.
3. `rollouts_per_expansion` stochastic typed RPN completions.
4. One batched native evaluation of all unique complete formulas collected in the
   current evaluation wave.
5. Mean rollout reward backup through the selected path.

For example, 128 simulations and 8 rollouts per expansion produce up to 1,024
terminal rollouts before canonical deduplication. Every finite unique formula remains
eligible for the archive even though only the mean rollout reward is backed up.

## Native candidate evaluator

```python
from flows.riskminer import CppStreamCandidateEvaluator

scorer = CppStreamCandidateEvaluator(
    sources,
    n_instruments=9,
    work_dir="/tmp/riskminer-candidates",
    batch_size=32,
)
scores = scorer.evaluate(candidate_expressions)
```

There is no NumPy/JAX fallback. Failed batches are bisected until the offending
formula is isolated and marked invalid.

## Search

```python
from flows.riskminer import RiskMinerConfig, search_cpp_stream_alphas

result = search_cpp_stream_alphas(
    sources,
    n_instruments=9,
    work_dir="/tmp/riskminer",
    config=RiskMinerConfig(
        max_depth=8,
        simulations=128,
        rollouts_per_expansion=8,
        evaluation_batch_size=32,
        archive_size=100,
        seed=42,
    ),
)

for entry in result.search.archive[:10]:
    print(entry.score, entry.depth, entry.rpn, entry.expr)
```

`roll_rets`, `hs`, `vol`, and `is_tradable` are evaluation-only inputs and are not
formula terminals.

## Ridge pool

```python
from flows.riskminer import CppStreamPoolEvaluator

pool = CppStreamPoolEvaluator(
    sources,
    n_instruments=9,
    work_dir="/tmp/riskminer-pool",
)
pool_result = pool.evaluate(
    [entry.expr for entry in result.search.archive[:100]],
)
print(pool_result.score)
```

The graph uses:

- `clean_rets = where(roll_rets != 0, roll_rets, NaN)`;
- `weights = hs ** -2`;
- shifted `alpha * vol` Ridge features;
- `hl = 1440 * 5`, `lambda = 0`, `nonneg = False`;
- current beta projected onto current scaled alpha features;
- session masking and forward filling;
- one-row position shift;
- final `pool_pnl.mean() / pool_pnl.std()`.

The EWM risk half-life is converted to the equivalent pandas-style span because the
current native EWM interface is span-based.

## Benchmark

```bash
PYTHONPATH=src python scripts/benchmark_riskminer_cpp_stream.py
```

Useful overrides:

```text
RISKMINER_ROWS
RISKMINER_SIMULATIONS
RISKMINER_ROLLOUTS
RISKMINER_EVALUATION_BATCH
RISKMINER_ARCHIVE_SIZE
RISKMINER_POOL_SIZE
RISKMINER_OUTPUT_DIR
```

The benchmark prints backend/runtime identity, shape, search counts, compile and run
times, top formulas, and pool score. It also writes `riskminer_benchmark.json`.

## Deferred after this executable checkpoint

- Learned four-layer GRU policy and risk-quantile training.
- Runtime-expression EWM decay and shift lag in neutral IR/native C++.
- Structured search schemas for every object/tensor/group operator.
- Large-budget generation of a full 100-alpha archive on 5,000,000 by 9 data.
- Incremental or block-update optimization for repeated trial additions to a
  100-feature Ridge pool.

The policy layer is intentionally abstracted behind `ActionPolicy`, so a learned
masked GRU can replace `SchemaPriorPolicy` without changing the MCTS or native
candidate evaluator.
