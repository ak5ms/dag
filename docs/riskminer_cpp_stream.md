# RiskMiner-style alpha search on `cpp_stream`

This implementation starts from `agent/alpha-search-semantic-types-v2` and replaces
the primary formula-generation loop with a typed Reverse Polish Notation (RPN)
environment and PUCT Monte Carlo tree search.

## Execution split

- Python owns the symbolic RPN state, semantic action mask, MCTS tree, and
  risk-seeking token policy.
- `trading_dsl_engine.cpp_stream` owns every candidate-alpha calculation,
  candidate PnL calculation, temporal mean/std reduction, and Ridge-pool
  calculation.
- NumPy is used only to create hypothetical inputs, read final native output, and
  construct small test references.

There is no NumPy/JAX/Numba candidate-evaluation fallback. A candidate that cannot
compile or execute in `cpp_stream` receives `-inf`, and its rejection is recorded.

## Candidate objective

```python
pnl = (shift(alpha, 1, 1) * roll_rets).sum(axis=1)
score = pnl.mean(axis=0) / pnl.std(axis=0)
```

Several candidates are packed with `cat`, evaluated by `einsum`, reduced across
instruments, and then reduced temporally. The generated native program is
final-only and writes one score per candidate.

## Typed RPN

A state contains a token sequence and a stack of normal `dag` expressions plus
multiple semantic tags, shape, range, literal/static status, depth, and a canonical
key. The action mask checks semantic and structural legality before exposing an
operator token.

Examples:

- quote price plus quote price is legal through `price`;
- quoted size plus trade volume is legal through `quantity`;
- price plus quantity is illegal;
- price divided by price is dimensionless;
- cross-sectional ranks become dimensionless;
- EWM, rolling, and shift preserve their value type.

The root defaults to `dimensionless`.

## MCTS

One simulation performs PUCT selection, progressive widening, one expansion,
several policy-guided RPN rollouts, batched native evaluation, and mean finite
reward backup. Selection batches use virtual visits so several leaves can be
collected before compiling a candidate batch.

The first checkpoint uses a small state-conditional token policy. Following the
RiskMiner lower-tail rule, token logits observed in episodes below the tracked
reward quantile are reduced. A GRU can replace it through the same interface.

## Ridge pool

Top unique alphas are scaled by `vol`, shifted into a weighted streaming Ridge,
projected with `get_beta`, session-masked, risk-scaled, shifted, multiplied by clean
returns, row-summed, and scored with mean/std. The requested half-life is converted
to the equivalent EWM span because the current canonical EWM API is span-based.

## Running the demonstration

```bash
PYTHONPATH=src \
RISKMINER_ROWS=4096 \
RISKMINER_SIMULATIONS=48 \
RISKMINER_ROLLOUTS=4 \
RISKMINER_MAX_DEPTH=8 \
python scripts/benchmark_riskminer_cpp_stream.py
```

The script prints proposal counts, native compile/run timings, backend evidence,
top formulas and token sequences, and the Ridge pool score. It also writes a JSON
result file. CI deliberately uses a smaller dataset and search budget.

The draft validation PR is intentionally kept open while the reduced native run is
reviewed; its purpose is test execution, not automatic merging.

## Deliberate checkpoint boundaries

The directly searchable vocabulary covers arithmetic, unary transforms,
cross-sectional rank/normalization, EWM, shift, common rolling statistics,
comparisons, and conditionals. `cat`, `einsum`, and `Ridge` are structured
evaluation operators. Grouping, arbitrary einsum strings, object-valued model
nodes, the paper's GRU, and native runtime-expression lookbacks remain subsequent
checkpoints rather than being silently treated as complete.
