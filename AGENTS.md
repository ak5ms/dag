# AGENTS.md

Guidance for AI/code agents working in this repo.

## Mission context

This project is a performance-sensitive trading-feature DSL engine. Active development now targets the `jax_flat` runtime and the shared DSL/parser/lowering layer; the older Numba and non-flat JAX implementations are deprecated compatibility code.

Unless a task explicitly says otherwise, make edits only for `jax_flat` and shared DSL functionality, and run only the targeted `tests/jax_flat/` or shared DSL tests needed for the behavior being changed. Do not update or run Numba/non-flat-JAX code paths by default.

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
- For grouped runtime implementations (especially JAX/JAX-flat), do not add per-operator custom hot-path branches (for example, special-casing `cumsum` inside `GroupbyOp`). Keep grouped execution generic/compositional and optimize shared mechanisms instead.
- For the JAX backend, keep live tick and batch timestep hot paths under JAX JIT (`eqx.filter_jit`/`jax.jit` plus `lax.scan` for batch), and prefer functional PyTree state over Python mutation inside compiled execution. JAX-flat-only lag-history operators such as `buffer(shift(...), min_lag, max_lag)` should expose ordered ring state through compositional JAX ops, not Numba kernels. User-supplied `jax_flat.stateless(...)` callables must stay stateless and execute through the same generic compiled `NaryOp` tick/`vmap` batch path. When adding a JAX-flat operator, implement `scan_batch` alongside `init_state` and `tick` rather than relying on the default per-tick scan.
- Support arity > 1 cleanly.
- Preserve column universe support for generic operators: `univ(...)` describes static column groups, `column_names` maps tickers to column positions, and grouped operators must run independently per universe on group sub-frames without interpreter fallback. `univ(...)` may also appear inside tuple keys such as `groupby((univ([0, 1]), ts), op)` to combine static column slicing with dynamic key routing.
- Grouped execution must use a single canonical form: `groupby(key_tuple, lhs, op_using_self_)` (or Python sugar `lhs.groupby(key_tuple).apply(op(self_, *others))`). Delete all legacy groupby forms and alternate flow paths. `key_tuple` must support arbitrary-length composite keys and may contain at most one `univ(...)` element.
- Keep Python-composed formulas feature-complete with string formulas: every builtin op should have a Python helper, expression nodes should preserve infix operator composition, grouping sugar such as `lhs.groupby(key).apply(...)` should lower to the same AST forms as strings, and `compile_formula`/`build_engine` should accept composed `Expr` objects as well as strings.
- Keep op-specific private helper functions on the relevant operator class as `@staticmethod`; these static methods may be reused from another class when that is the cleanest shared implementation.
- DSL/operator naming convention: functions that emit scalar/vector/matrix arrays use lower_snake_case; helpers that emit object/model state use UpperCamelCase (for example `Ridge` and `InstrumentBasisMean`).
- When adding new active `jax_flat` operators, implement both the pure JAX-flat operator and corresponding native C++ lowering/runtime support unless the task explicitly scopes C++ out; document and test any intentional C++ fallback.
- Ridge weights may be omitted in supported forms and must default to unit per-instrument weights without changing explicit-weight semantics.

## Where to change what

- Shared parser/validation changes: `src/trading_dsl_engine/base/parser.py`
- Shared DSL macro composition + registry isolation: `src/trading_dsl_engine/base/dsl.py`
- Shared operator plugin specs: `src/trading_dsl_engine/base/registry.py`
- Shared compile/lower pipeline: `src/trading_dsl_engine/base/compiler.py`
- Active JAX-flat op kernels/factories: `src/trading_dsl_engine/jax_flat/ops.py`
- Active JAX-flat runtime execution and batch/live helpers: `src/trading_dsl_engine/jax_flat/engine.py`
- Experimental native C++ JAX-flat tick core: Python lowering/wrapper lives in `src/trading_dsl_engine/jax_flat/engine_cpp.py`; native extension code is split between `src/trading_dsl_engine/jax_flat/engine.cpp` and `src/trading_dsl_engine/jax_flat/ops.cpp` (keep this optional, flattened, and allocation-conscious; `compile_formula(..., cpp=True)` enables automatic native acceleration for supported grouped hot paths, `cpp=False` forces pure JAX-flat, batch helpers must reuse the same non-batch row transition, unsupported automatic accelerator formulas should warn with the unsupported node before falling back to JAX-flat, and `TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL=1` must force the pure JAX path for behavior checks).
- Active JAX-flat behavior/performance regression tests: `tests/jax_flat/`; the groupby cartesian benchmark CLI lives at `tests/jax_flat/test_benchmark_groupby_matrix.py` and is a perf test when configured above 200 rows.
- Deprecated Numba implementation: `src/trading_dsl_engine/numba/` (do not edit unless explicitly requested)
- Deprecated non-flat JAX implementation: `src/trading_dsl_engine/jax/` (do not edit unless explicitly requested)
- Deprecated Numba/non-flat-JAX tests: `tests/numba/`, `tests/jax/` (do not run unless explicitly requested)

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
- Matrix-op shape behavior when emitted width differs from instrument count (e.g., basis expansions like `bspline` and JAX-flat lag-history cubes from `buffer`).
- Grouped-state behavior for canonical keyed operators (`groupby(key_tuple, lhs, op_using_self_)`) including arbitrary-length tuple-key composition, per-instrument/per-key state transitions, and key consistency expectations.
- Key NaNs in groupby are valid and must route into a dedicated NaN key group (do not raise).
- Tuple-key universe behavior (`groupby((..., univ(...), ...), lhs, op_using_self_)`) including ticker-to-column mapping, per-universe sub-frame state isolation, dynamic-key masking within universes, and scatter/broadcast shape behavior.

## Environment and test expectations

Use a repo-local virtualenv and pip cache when setting up a fresh cloud/agent environment so dependency downloads and wheels can be reused between iterations:

```bash
python -m venv .venv
. .venv/bin/activate
PIP_CACHE_DIR=.pip-cache python -m pip install -e .
```

JAX, Equinox, pytest, and pytest-xdist are mandatory project dependencies in `pyproject.toml`; do not treat the JAX backend as optional in setup or tests. The `.venv/` and `.pip-cache/` directories are gitignored and must not be committed.

During iteration, run only the relevant targeted tests for the files/behavior being changed. Do not repeatedly run the full suite while iterating.
If a long-running pytest command starts showing failures in streamed output, you may preemptively terminate that run early to iterate faster, then rerun targeted tests after fixes.

Unless explicitly instructed otherwise, final validation should stay limited to targeted `jax_flat` and shared DSL tests for the changed behavior. Do not run the deprecated Numba/non-flat-JAX suites by default. If performance behavior changes in `jax_flat`, run the relevant active performance test(s), for example:

```bash
pytest -q tests/jax_flat/test_changed_behavior.py
RUN_PERF_TESTS=1 pytest -n 0 tests/jax_flat/test_performance.py -q
```

`pytest` is configured in `pyproject.toml` to run with pytest-xdist using 12 workers (`-n 12`). Run perf tests with `-n 0` because their wall-clock guardrails are calibrated for serial benchmark execution. If perf tests are too heavy for the environment, clearly note that and run the targeted non-performance test(s) instead.

- Always run the targeted non-performance `jax_flat`/DSL tests at the end before finalizing unless explicitly told not to.
- Pytest output is configured to include per-test durations; use this to catch regressions in compile/runtime costs.
- Whenever behavior/architecture expectations change, update both `README.md` and `AGENTS.md` in the same PR.

## Coding style

- Always use absolute imports (e.g., `from trading_dsl_engine...`), not relative imports.
- Keep implementations concise and generic; avoid repetitive boilerplate.
- Native C++ should read like a carefully reviewed systems component: prefer named state structs, Eigen-backed math/state buffers, small factory helpers, and clear ownership boundaries over monolithic slots, nested container soup, or broad if/else initialization blocks.
- Prefer factories/templates/registries over hardcoded branching.
- Prefer `make_nary_op` for stateless scalar/vector/matrix operators, including axis reducers, before adding custom operator classes.
- Do not wrap imports in try/except blocks.
- Make extension points obvious for future ops (including potential matrix/tensor emitters and optimizer/model workflow nodes).

## Future roadmap hints

Planned direction includes graph-level typed IR, CSE/fusion, and non-eager model/portfolio optimizer nodes compiled through the same pipeline. Avoid changes that block this evolution.
