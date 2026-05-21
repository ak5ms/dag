# trading-dsl-engine

A test-driven, high-performance DSL engine for streaming trading features on aligned minutely NumPy data.

The engine compiles formulas (string DSL or Python-composed expressions) into nested Numba `jitclass` state machines for low-latency live updates and scalable batch runs.

## Current project goal

The operating goal is **TDD-first backend reconciliation**:

1. Write/adjust focused tests first.
2. Implement minimal code changes.
3. Re-run relevant tests and iterate until green.
4. Reconcile feature/behavior gaps from Numba into `jax_new`.
5. Deprecate legacy `jax` paths as parity lands in `jax_new`.

## Backend status

- **Numba (`src/trading_dsl_engine/numba/`)**: reference runtime semantics.
- **Legacy JAX (`src/trading_dsl_engine/jax/`)**: deprecated migration surface.
- **JAX New (`src/trading_dsl_engine/jax_new/`)**: target backend for missing-ops/features reconciliation and future development.

If a feature exists in Numba but not in `jax_new`, the expected direction is to add tests and implement parity in `jax_new`.

## Design priorities

- **Streaming-first semantics**: every op follows `on_data(...)` + `emit(...)`.
- **Incremental updates**: no full-history recomputation in live ticks.
- **Compiled hot paths**: avoid interpreter fallback in timestep loops.
- **Extensible architecture**: registry/composition-driven growth.
- **Parity-driven evolution**: converge `jax_new` behavior toward Numba.

## Repository structure

- `src/trading_dsl_engine/base/`: parser, DSL helpers, registry contracts, compile/lower pipeline.
- `src/trading_dsl_engine/numba/`: reference backend implementation.
- `src/trading_dsl_engine/jax/`: legacy backend being deprecated.
- `src/trading_dsl_engine/jax_new/`: target backend under active reconciliation.
- `tests/numba/`: correctness/performance baseline coverage.
- `tests/jax/`: legacy backend tests.
- `tests/jax_new/` (if present): preferred location for new parity tests.

## Typical usage

```python
from trading_dsl_engine import compile_formula, build_engine, run_batch_from_mapping

formula = "xs_rank(ewm(div(close, open), 21))"
artifact = compile_formula(formula)
engine = build_engine(formula)

# Live tick update: 1D vectors shaped (n_instruments,)
out_live = engine.update_from_mapping({"open": open_t, "close": close_t})

# Batch run: aligned 2D arrays shaped (time, n_instruments)
out_batch = run_batch_from_mapping(
    engine,
    {"open": open_2d, "close": close_2d},
    chunk_size=4096,
)
```

By default, batch output is disk-backed (memmap) to limit RAM pressure. Use `out_path=None` to opt into in-memory output.

## DSL composition and canonical grouping

Python-composed expressions and string formulas are both supported through the shared lowering pipeline.

- Compose via helpers such as `add`, `div`, `var`, and infix operators on `Expr` objects.
- Canonical grouped form:
  - `groupby((key1, key2, ...), lhs, op_using_self_)`
  - Python sugar: `lhs.groupby((key1, key2, ...)).apply(op_expr_using_self_)`
- Tuple keys can be arbitrary length and may include at most one `univ(...)` element.
- Key NaNs are valid and route to a dedicated NaN-key group.

## Data contract

- Batch inputs: aligned arrays of shape `(time, n_instruments)`.
- Live updates: vectors of shape `(n_instruments,)`.
- Some ops may emit matrix outputs where width differs from instrument count.
- Optional `column_names` enables ticker-to-column mapping for universe-aware grouping.

## NaN semantics (current)

- Binary ops propagate NaN values.
- `div` returns NaN on divide-by-zero.
- `shift(x, nlag, max_size)` stores a static ring capacity from numeric literal `max_size` while reading `x` and scalar `nlag` as sources; `shift(x, literal_nlag)` remains supported by using the literal lag as capacity.
- `ewm` skips updates for NaN inputs and can recover from NaN state.
- `xs_rank` ranks only valid values and emits NaN where input is NaN.
- `bspline(x, n_basis)` emits a per-instrument periodic basis matrix on `[0, 1]` with output width `n_basis` (inputs are clipped to `[0, 1]` and NaNs propagate).
- `col(matrix, index)` extracts one matrix column as a vector for explicit feature selection/probing.
- `mod(a, b)` provides elementwise modulo for scalar/vector/matrix combinations supported by binary broadcasting rules.
- `groupby` now has one canonical form: `groupby((key1, key2, ..., maybe_univ, ...), lhs, op_using_self_)`.
- The tuple key supports arbitrary length. It may contain at most one `univ(...)` element plus any number of dynamic key expressions.
- The local op expression must consume the outer stream value via `self_`, e.g. `groupby((day, bucket), get_preds(Ridge(...)), cumsum(self_))`.
- Python-composed formulas use the same canonical lowering via `lhs.groupby((...)).apply(op_expr_using_self_)`.
- Legacy groupby forms are intentionally removed (no `groupby(key, op)` and no alternate universe-only syntax path).
- Key NaNs are valid in groupby and are routed into a dedicated NaN-key group instead of raising.

## Ridge regression op (cross-sectional)

- `Ridge(x1, x2, ..., xk, y, weights, hl, lambda)` emits an object state and performs cross-sectional EWMA ridge updates each tick.
- `Ridge(x, y, hl, lambda)` omits weights and defaults them to 1.0 per instrument; Python-composed `Ridge(..., y=target, hl=21, lambda_=0.1)` supports the same default for multiple feature args without ambiguity.
- Ridge feature args can be vectors and/or matrices; matrix features are expanded by columns internally, so `Ridge(bspline(...), y, w, hl, lambda)` works without manual `col(...)` calls.
- Explicit `weights` can be a scalar, a vector `(n, 1)`/`(n,)` of sample weights, or a dense matrix `(n, n)` for `X'WX`/`X'Wy` weighting; scalar weights are broadcast per instrument.
- State uses pairwise-NaN-aware exponentially weighted sufficient statistics (`X'WX`, `X'Wy`) with per-statistic clocks, then solves `beta = (G + lambda * diag(G))^-1 h` fresh each tick.
- Pairwise clocks only advance when the corresponding `xx[j, k]` or `xy[j]` statistic receives at least one finite observation; missing rows or full outages leave only the affected statistics unchanged.
- A visual checklist for NaN behavior is available in `docs/ewm_regression_nan_handling.svg`.
- `get_preds(Ridge(...))` returns one-step-lagged predictions per instrument (`beta(t-1)·x(t)`).
- `get_beta(Ridge(...))` returns the current coefficient vector with shape `(k, 1)`.

## Development workflow

```bash
python -m venv .venv
. .venv/bin/activate
PIP_CACHE_DIR=.pip-cache python -m pip install -e .
```

Iterate with targeted tests, then run final validation:

```bash
pytest -q
RUN_PERF_TESTS=1 pytest -n 0 tests/numba/test_performance.py -q
```

## Contributor expectations

- Follow TDD by default: tests first, then implementation, then targeted reruns.
- For reconciliation work, start from failing parity tests and close gaps in `jax_new`.
- Preserve streaming semantics and performance guardrails.
- Keep architecture extensible (registries/factories/composition).
- Update both `README.md` and `AGENTS.md` when project expectations shift.
