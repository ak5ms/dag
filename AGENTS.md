# AGENTS.md

Guidance for AI/code agents working in this repo.

## Mission context

This project is a **test-driven, performance-sensitive trading-feature DSL engine** that compiles formulas into nested Numba `jitclass` state machines.

Backend direction (current priority):
- `src/trading_dsl_engine/jax/` is the legacy backend.
- `src/trading_dsl_engine/jax_new/` is the target backend.
- Active goal is **reconciliation**: implement missing ops/features in `jax_new` and close parity gaps against Numba while deprecating legacy JAX paths.

Primary objective order:
1. **TDD correctness loop first**: test -> code -> test -> code until the relevant suite is green.
2. Preserve streaming state semantics (`on_data(...)` + `emit(...)`) and incremental updates.
3. Preserve or improve runtime performance in hot paths.
4. Keep architecture extensible via registry/composition patterns.
5. Move backend effort to `jax_new`; avoid new feature investment in legacy `jax` except migration-critical fixes.

## Required development workflow

For all behavior changes:
1. Add/adjust focused tests first.
2. Run the smallest relevant failing slice.
3. Implement minimal code change.
4. Re-run relevant tests and iterate to green.
5. Run full non-performance suite before finalizing.
6. Run performance suite when changing perf-sensitive runtime behavior.

Do **not** start with broad refactors detached from tests.

## Repository structure and ownership map

- `src/trading_dsl_engine/base/`
  - Parser, DSL composition, registry specs, and shared compile/lower pipeline.
- `src/trading_dsl_engine/numba/`
  - Reference backend for streaming semantics and primary runtime behavior.
- `src/trading_dsl_engine/jax/`
  - Legacy backend; treat as deprecated migration surface.
- `src/trading_dsl_engine/jax_new/`
  - Target backend for parity and future development.
- `tests/numba/`
  - Ground-truth behavior and performance regression coverage.
- `tests/jax/`
  - Legacy JAX correspondence tests.
- `tests/jax_new/` (if present)
  - Preferred place for new JAX parity/correctness tests.

When expectations change, update **both** `AGENTS.md` and `README.md` in the same PR.

## Core invariants

- Every op must honor strict `on_data(...)` + `emit(...)` semantics (including stateless ops).
- Live updates must be incremental; no full-history recomputation in update paths.
- Keep compiler composition nested; no interpreter fallback in execution hot paths.
- Avoid requiring `n_instruments` in constructors when shape can be inferred at first tick.
- Support arity > 1 cleanly across parser, lowering, and backends.
- Keep Python-composed `Expr` formulas feature-par with string formulas.
- Ridge weights may be omitted in supported forms and default to unit per-instrument weights.

### Grouping invariants

- Canonical grouped form only:
  - `groupby(key_tuple, lhs, op_using_self_)`
  - Python sugar: `lhs.groupby(key_tuple).apply(op(self_, *others))`
- `key_tuple` supports arbitrary-length composite keys.
- `key_tuple` may contain at most one `univ(...)` element.
- Key NaNs are valid and must route to a dedicated NaN-key group.

## Parity and migration rules (`numba` -> `jax_new`)

- Treat Numba behavior as reference when adding missing `jax_new` ops/features.
- Add tests that assert reconciliation for parser/lowering/runtime semantics.
- Prefer adding new backend coverage under `tests/jax_new/`; keep legacy `tests/jax/` only for migration confidence.
- Do not introduce new feature-only APIs in legacy `jax`.

## Performance guardrails

- No Python-level per-timestep loops in runtime hot paths.
- Prefer compiled/JIT paths (`jitclass`, `jax.jit`/`eqx.filter_jit`, `lax.scan`).
- Minimize unnecessary array copies/materialization in batch mode.
- Keep batch output disk-backed by default (`run_batch_from_mapping(..., out_path=...)`) unless explicitly opting in to RAM.

## Numerical / NaN expectations

When modifying ops, add/update tests covering:
- Binary NaN propagation.
- Divide-by-zero behavior.
- Stateful-op behavior under NaN outages.
- Ranking/tie semantics and NaN masking.
- Matrix-output shape behavior.
- Ridge pairwise sufficient-stat clocks and finite-row gating.
- Canonical `groupby` keyed-state behavior (tuple keys, `univ(...)`, NaN keys).

## Environment and tests

```bash
python -m venv .venv
. .venv/bin/activate
PIP_CACHE_DIR=.pip-cache python -m pip install -e .
```

During iteration, run only targeted tests. Before finalizing:

```bash
pytest -q
RUN_PERF_TESTS=1 pytest -n 0 tests/numba/test_performance.py -q
```

If perf tests are too heavy for the environment, clearly report that and still run full non-performance suite.

## Coding style

- Use absolute imports (`from trading_dsl_engine...`).
- Whenever a new operator is added (or operator semantics change), update `README.md` in the same PR with user-facing behavior, NaN semantics, shape expectations, and usage notes.
- Keep implementations concise and generic; avoid repetitive boilerplate.
- Prefer factories/templates/registries over hardcoded branching.
- Do not wrap imports in try/except blocks.

## Roadmap compatibility

Planned direction includes typed IR, CSE/fusion, and non-eager model/portfolio optimizer nodes compiled through the same pipeline. Prioritize changes that accelerate `jax_new` parity without blocking this roadmap.

## Practical notes for future agents

- Start by locating existing coverage with `rg` under `tests/` before writing new tests; extend nearby test modules rather than scattering one-off files.
- For backend reconciliation tasks, prefer this sequence: confirm Numba behavior in `tests/numba/` -> add/port parity test in `tests/jax_new/` -> implement in `src/trading_dsl_engine/jax_new/`.
- When touching parser or lowering behavior, verify both string formulas and Python-composed `Expr` paths continue to lower identically.
- Preserve canonical grouping APIs only; if you see legacy groupby forms in code/tests, migrate to canonical tuple-key form instead of adding compatibility branches.
- Before finalizing, sanity-check docs for drift: if behavior expectations changed, update `README.md` + `AGENTS.md` in the same change.
- Keep performance in mind while coding: avoid introducing Python loops in per-tick runtime paths even when tests pass.

