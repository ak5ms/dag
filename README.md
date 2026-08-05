# trading-dsl-engine

A high-performance Python DSL engine for streaming trading features on aligned minutely NumPy data.

This repository compiles formulas (string DSL or Python-composed DSL calls) into nested Numba `jitclass` state machines that support both live incremental updates and batch execution. A JAX + Equinox backend is installed as a standard dependency and is available for formulas supported by `trading_dsl_engine.jax`, with live tick and batch scan hot paths wrapped in JAX JIT compilation.

## Core goals

- **Streaming-first stateful computation**: each op follows `on_data(...)` + `emit(...)`.
- **No interpreter hot loop**: runtime timestep loop executes in compiled Numba code.
- **Composable formulas**: parse string expressions and support Python-level DSL macro composition.
- **Extensible operators**: registry/plugin model for adding new ops without central branching.
- **Scalable IO**: supports in-memory NumPy arrays and disk-backed memmaps.

## Project layout

- `src/trading_dsl_engine/base/`
  - Shared parser (`parse_formula`), Python DSL constructors, registry metadata, and compile/lower pipeline.
- `src/trading_dsl_engine/numba/`
  - Numba built-in op implementations, jitclass state machines, and batch/live runtime helpers.
- `src/trading_dsl_engine/jax/`
  - Optional JAX + Equinox runtime that lowers supported DSL expressions to functional state transitions and executes live ticks/batch scans through JIT-compiled JAX functions.
- `tests/numba/`
  - Parser, composition, Numba runtime correctness, shape, state persistence, and performance tests.
- `tests/jax/`
  - JAX backend correspondence tests against the Numba runtime.

## Typical usage

```python
from trading_dsl_engine import compile_formula, build_engine, run_batch_from_mapping

artifact = compile_formula("xs_rank(ewm(div(close, open), 21))")
engine = build_engine("xs_rank(ewm(div(close, open), 21))")

# live tick update
out = engine.update_from_mapping({"open": open_t, "close": close_t})

# batch run (disk-backed output by default)
out2d = run_batch_from_mapping(engine, {"open": open_2d, "close": close_2d}, chunk_size=4096)  # memmap shape (time, n_instruments) for vector outputs
# opt into RAM materialization instead
out2d_ram = run_batch_from_mapping(engine, {"open": open_2d, "close": close_2d}, out_path=None)
```

Batch execution writes output to a NumPy memmap at `/tmp/trading_dsl_engine_out.memmap` by default to avoid materializing full results in RAM. Pass `out_path=None` to allocate in memory, or provide `out=` to write into a preallocated array.

Object-typed intermediate nodes are supported (e.g., stateful jitclass/structref emitters) as long as a downstream op projects them back to scalar/vector/matrix. Root object outputs are intentionally rejected in batch mode to keep the timestep loop on the compiled JIT path.

## DSL composition

You can define reusable macro-like composed functions with an explicit registry namespace:

```python
from trading_dsl_engine import DSLFunctionRegistry, register_dsl_function, add, div

my_registry = DSLFunctionRegistry()

@register_dsl_function("hlc3", registry=my_registry)
def hlc3(high, low, close):
    return div(add(add(high, low), close), 3.0)
```

Then compile with `compile_formula(..., dsl_registry=my_registry)`. Built-in composed DSL functions include `diff(x, nlag=1, max_size=1)`, which expands to `sub(x, shift(x, nlag, max_size))`.

Normal Python composition can use `var("close")`/`var("open")` identifiers and either prefix helpers or infix operator overloads: `xs_rank(ewm(var("close") / var("open"), 21))` is equivalent to `xs_rank(ewm(div(close, open), 21))`. Formula strings also support infix arithmetic/logical forms such as `close + open`, `close * 2`, `close % 5`, `close | open`, `close & open`, `close ^ open`, `close == open`, and `close != open`.

The returned artifact includes `stats` (`expanded_nodes`, `cache_hits`, `compile_seconds`) so compile-time CSE behavior and compile latency can be validated.

## Data contract

- Inputs are aligned 2D arrays with shape `(time, n_instruments)`.
- Optional `column_names` passed to `compile_formula(...)`/`build_engine(...)` maps universe ticker names to input column positions for static column grouping.
- Live `update` expects 1D vectors with shape `(n_instruments,)`.
- Some ops may emit matrix outputs (e.g., `outer`, `bspline`), with shape `(n_instruments, width)` where `width` can differ from `n_instruments`.

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

## JAX backend

JAX and Equinox are required project dependencies. The backend mirrors the core runtime helpers under `trading_dsl_engine.jax`:

```python
from trading_dsl_engine.jax import build_jax_engine, run_batch_from_mapping

engine = build_jax_engine("xs_rank(ewm(div(close, open), 21))")
out = run_batch_from_mapping(engine, {"open": open_2d, "close": close_2d}, out_path=None)
```

The JAX backend accepts the same string formulas and Python-composed `Expr` trees as the Numba backend, including infix math/comparison operators and grouped-expression sugar such as `lhs.groupby((...)).apply(...)`. It stores formula structure in per-operator Equinox modules and wraps both the single-tick state transition and the batch `lax.scan` path with `eqx.filter_jit`, keeping the per-timestep hot paths compiled. It covers the scalar/vector/matrix stateless operators, `ewm`, `cumsum`, `shift`, `rolling_quantile`, `xs_rank`, `outer`, `bspline`, `col`, `mean`, canonical tuple-key groupby `groupby((...), lhs, op_using_self_)` (including optional `univ(...)` inside the key tuple), and Ridge projections via `get_beta(Ridge(...))`/`get_preds(Ridge(...))`.

## Development quickstart

```bash
python -m venv .venv
. .venv/bin/activate
PIP_CACHE_DIR=.pip-cache python -m pip install -e .
pytest -q  # configured to use pytest-xdist with 12 workers
```

The `.venv/` and `.pip-cache/` paths are gitignored so cloud/agent environments can reuse a repo-local virtualenv and wheel/download cache between iterations without committing environment artifacts.

Performance tests (opt-in):

```bash
RUN_PERF_TESTS=1 pytest -n 0 tests/numba/test_performance.py -q
```

## Notes for future work

- Add graph-level IR + CSE for shared subtrees across multi-feature workflows.
- Expand shape system for richer multi-output model/optimizer nodes.
- Continue reducing memory movement in batch paths for large memmap workloads.

## Branch push validation

This repository accepts branch updates through the standard pull request workflow. Use a small documentation-only change like this section when validating branch push and PR automation without altering runtime behavior.
