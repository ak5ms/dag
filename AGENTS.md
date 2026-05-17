# AGENTS.md

Guidance for AI/code agents working in this repo.

## Mission context

This project is a performance-sensitive trading-feature DSL engine that compiles formulas into nested Numba `jitclass` state machines.

Priorities, in order:
1. Preserve correctness and streaming state semantics.
2. Preserve or improve performance (avoid Python loops in hot path).
3. Keep architecture extensible (registry/composition-driven, no giant central branching).

## Key invariants

- Every operation should follow strict `on_data(...)` + `emit(...)` behavior (including stateless ops).
- Live updates must be incremental; do not recompute full history in update paths.
- Lagged operators such as `shift(x, nlag, max_size)` should keep bounded static history capacity from `max_size` while reading `x`/`nlag` through normal compiled sources.
- Avoid requiring `n_instruments` in constructors when shape can be inferred at first update.
- Keep compiler composition nested (no interpreter fallback in execution hot path).
- For the optional JAX backend, keep live tick and batch timestep hot paths under JAX JIT (`eqx.filter_jit`/`jax.jit` plus `lax.scan` for batch), and prefer functional PyTree state over Python mutation inside compiled execution.
- Support arity > 1 cleanly.
- Preserve column universe support for generic operators: `univ(...)` describes static column groups, `column_names` maps tickers to column positions, and grouped operators must run independently per universe on group sub-frames without interpreter fallback.
- Preserve both dynamic keyed grouping scopes: `groupby(key, op)` groups the full op subtree in each instrument/key single-column scope, while `groupby(key, lhs, op_using_self_)` computes `lhs` outside the keyed scope and groups only the local op that consumes `self_`. Do not special-case stateless ops into cross-sectional execution for dynamic keyed groupby; cross-column grouping belongs to `groupby(univ(...), op)`.
- Keep Python-composed formulas feature-complete with string formulas: every builtin op should have a Python helper, expression nodes should preserve infix operator composition, grouping sugar such as `lhs.groupby(key).apply(...)` should lower to the same AST forms as strings, and `compile_formula`/`build_engine` should accept composed `Expr` objects as well as strings.
- Ridge weights may be omitted in supported forms and must default to unit per-instrument weights without changing explicit-weight semantics.

## Where to change what

- Shared parser/validation changes: `src/trading_dsl_engine/base/parser.py`
- Shared DSL macro composition + registry isolation: `src/trading_dsl_engine/base/dsl.py`
- Shared operator plugin specs: `src/trading_dsl_engine/base/registry.py`
- Shared compile/lower pipeline: `src/trading_dsl_engine/base/compiler.py`
- Numba op kernels/factories: `src/trading_dsl_engine/numba/ops.py`
- Numba runtime execution and batch/live helpers: `src/trading_dsl_engine/numba/engine.py`
- Optional JAX + Equinox backend: `src/trading_dsl_engine/jax/`
- Numba behavior/performance regression tests: `tests/numba/`
- JAX correspondence regression tests: `tests/jax/`

## Performance guardrails

- Do not add Python-level per-timestep loops in runtime hot paths.
- Prefer compiled loops in jitclass methods.
- Minimize extra array copies/materialization in batch mode.
- Prefer clear NumPy/Numba slice and vectorized operations over unnecessary scalar loops, especially nested loops that only copy or assign contiguous rows, columns, or blocks (for example, use `dst[:] = src`, `dst[:, i:j] = block`, or `out[t, :] = values` where supported).
- Keep numerical code human-readable: avoid mechanically expanded, deeply nested NumPy/Numba code when a supported vectorized expression or slice assignment communicates the same semantics without changing streaming behavior.
- Keep batch output disk-backed by default (`run_batch_from_mapping(..., out_path=...)`) to avoid large RAM materialization; use `out_path=None` only when in-memory output is explicitly desired.
- If adding ops that emit non-ndarray state objects (`TypeInfo("object")`), keep the batch timestep loop in compiled/JIT code; project object state back to scalar/vector/matrix before root output.
- For any algorithmic change, consider complexity across ~1 year minutely x ~150 instruments (or larger).

## NaN and numerical behavior

When modifying ops, keep NaN handling explicit and tested:
- Binary propagation behavior.
- Divide-by-zero behavior.
- Stateful-op behavior when inputs include NaNs.
- Ranking/tie semantics and NaN masking.
- Ridge/object-op behavior when feature/target/parameter inputs include NaNs.
- Ridge pairwise sufficient-statistic behavior: `xx[j, k]` and `xy[j]` update only when their own finite row requirements are met, and their per-statistic clocks do not advance during outages.
- Ridge variadic-feature behavior (`Ridge(x1, ..., xk, y, weights, hl, lambda)`) and downstream shape expectations.
- Matrix-op shape behavior when emitted width differs from instrument count (e.g., basis expansions like `bspline`).
- Grouped-state behavior for dynamic keyed operators (e.g., `groupby(key, op)`) including key NaNs, per-instrument/per-key state transitions, singleton local stream behavior for stateless reducers, and key consistency expectations.
- Static universe grouping behavior (`groupby(univ(...), op)`) including ticker-to-column mapping, per-universe sub-frame state isolation, scatter/broadcast shape behavior, and NaN handling in cross-column reducers such as `mean`.

## Test expectations

Run these locally before finalizing:

```bash
pytest -q
RUN_PERF_TESTS=1 pytest tests/numba/test_performance.py -q
```

If perf tests are too heavy for the environment, clearly note that and at least run core tests.

- Always run non-performance tests (`pytest -q`) before finalizing unless explicitly told not to.
- Pytest output is configured to include per-test durations; use this to catch regressions in compile/runtime costs.
- Whenever behavior/architecture expectations change, update both `README.md` and `AGENTS.md` in the same PR.

## Coding style

- Always use absolute imports (e.g., `from trading_dsl_engine...`), not relative imports.
- Keep implementations concise and generic; avoid repetitive boilerplate.
- Prefer factories/templates/registries over hardcoded branching.
- Prefer `make_nary_op` for stateless scalar/vector/matrix operators, including axis reducers, before adding custom operator classes.
- Do not wrap imports in try/except blocks.
- Make extension points obvious for future ops (including potential matrix/tensor emitters and optimizer/model workflow nodes).

## Future roadmap hints

Planned direction includes graph-level typed IR, CSE/fusion, and non-eager model/portfolio optimizer nodes compiled through the same pipeline. Avoid changes that block this evolution.
