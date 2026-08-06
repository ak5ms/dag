# RiskMiner-style alpha search on `cpp_stream`

This implementation starts from `agent/alpha-search-semantic-types-v2` and replaces
the primary formula-generation loop with a typed Reverse Polish Notation (RPN)
environment and PUCT Monte Carlo tree search.

## Execution split

- Python owns the symbolic RPN state, semantic action mask, MCTS tree, replay
  episodes, and orchestration.
- JAX owns only the token-selection policy: a four-layer GRU with 64 hidden units
  followed by a 32/32 MLP, matching the RiskMiner architecture.
- `trading_dsl_engine.cpp_stream` owns every candidate-alpha calculation,
  candidate PnL calculation, temporal mean/std reduction, and Ridge-pool
  calculation.
- NumPy is used only to create hypothetical inputs, read final native output, and
  construct small correctness references.

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

The root defaults to `dimensionless`. Evaluation-only fields such as `roll_rets`,
`hs`, `vol`, and `is_tradable` are not formula terminals.

## MCTS

One simulation performs PUCT selection, progressive widening, one expansion,
several policy-guided RPN rollouts, batched native evaluation, and mean finite
reward backup. Selection batches use virtual visits so several leaves can be
collected before compiling a candidate batch.

The default benchmark uses `GRURiskSeekingTokenPolicy`:

- token embedding width 32;
- four GRU layers of width 64;
- two MLP hidden layers of width 32;
- masked legal-action softmax;
- tracked reward quantile;
- lower-tail log-probability suppression.

`RiskSeekingTokenPolicy`, a small deterministic state-conditional table policy,
remains available for focused tests and MCTS ablations.

## Ridge pool

Top unique alphas are scaled by `vol`, shifted into a weighted streaming Ridge,
projected with `get_beta`, session-masked, risk-scaled, shifted, multiplied by clean
returns, row-summed, and scored with mean/std. The requested half-life is converted
to the equivalent EWM span because the current canonical EWM API is span-based.

The initial checkpoint evaluates the exact Ridge pool after individual-alpha search.
It does not yet compile a 101-feature trial pool for every rollout. That separation
keeps the first end-to-end run tractable while preserving exact native pool scoring
for the selected family.

## Running the demonstration

```bash
PYTHONPATH=src \
RISKMINER_ROWS=4096 \
RISKMINER_SIMULATIONS=48 \
RISKMINER_ROLLOUTS=4 \
RISKMINER_MAX_DEPTH=8 \
python scripts/benchmark_riskminer_cpp_stream.py
```

The script prints proposal counts, GRU training state, native compile/run timings,
backend evidence, top formulas and token sequences, and the Ridge pool score. It
also writes a JSON result file. The environment-variable controls make it possible
to reduce rows and simulation count for a quick smoke run without changing the
search semantics.

## Deliberate checkpoint boundaries

The directly searchable vocabulary covers arithmetic, unary transforms,
cross-sectional rank/normalization, EWM, shift, common rolling statistics,
comparisons, and conditionals. `cat`, `einsum`, and `Ridge` are structured
evaluation operators.

The following remain explicit subsequent checkpoints rather than being silently
claimed as complete:

- native runtime-expression EWM decay, lag, and rolling windows;
- structured groupby/universe actions inside ordinary alpha formulas;
- arbitrary generated einsum signatures;
- object-valued model nodes as ordinary formula tokens;
- exact Ridge-pool terminal reward on every rollout;
- promotion of every `cpp_stream` entrypoint from classified inventory to a direct
  RPN action where its argument structure is economically meaningful.
