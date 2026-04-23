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
- Avoid requiring `n_instruments` in constructors when shape can be inferred at first update.
- Keep compiler composition nested (no interpreter fallback in execution hot path).
- Support arity > 1 cleanly.

## Where to change what

- Parser/validation changes: `src/trading_dsl_engine/parser.py`
- DSL macro composition + registry isolation: `src/trading_dsl_engine/dsl.py`
- Operator plugin specs: `src/trading_dsl_engine/registry.py`
- Builtin op kernels and factories: `src/trading_dsl_engine/ops.py`
- Compile/lower pipeline: `src/trading_dsl_engine/compiler.py`
- Runtime execution and batch/live helpers: `src/trading_dsl_engine/engine.py`
- Behavior regression tests: `tests/`

## Performance guardrails

- Do not add Python-level per-timestep loops in runtime hot paths.
- Prefer compiled loops in jitclass methods.
- Minimize extra array copies/materialization in batch mode.
- For any algorithmic change, consider complexity across ~1 year minutely x ~150 instruments (or larger).

## NaN and numerical behavior

When modifying ops, keep NaN handling explicit and tested:
- Binary propagation behavior.
- Divide-by-zero behavior.
- Stateful-op behavior when inputs include NaNs.
- Ranking/tie semantics and NaN masking.

## Test expectations

Run these locally before finalizing:

```bash
pytest -q
RUN_PERF_TESTS=1 pytest tests/test_performance.py -q
```

If perf tests are too heavy for the environment, clearly note that and at least run core tests.

## Coding style

- Keep implementations concise and generic; avoid repetitive boilerplate.
- Prefer factories/templates/registries over hardcoded branching.
- Do not wrap imports in try/except blocks.
- Make extension points obvious for future ops (including potential matrix/tensor emitters and optimizer/model workflow nodes).

## Future roadmap hints

Planned direction includes graph-level typed IR, CSE/fusion, and non-eager model/portfolio optimizer nodes compiled through the same pipeline. Avoid changes that block this evolution.
