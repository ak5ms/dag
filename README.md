# trading-dsl-engine

A high-performance Python DSL engine for streaming trading features on aligned minutely array data.

Current development targets the `trading_dsl_engine.jax_flat` runtime plus shared DSL/parser functionality. The older Numba and non-flat JAX implementations remain in the tree for compatibility, but they are deprecated: unless a task explicitly says otherwise, make code changes only in `jax_flat` and shared DSL modules, and run only the focused `tests/jax_flat/` and shared DSL tests needed for the behavior being changed.

## Core goals

- **Streaming-first stateful computation**: each op follows `on_data(...)` + `emit(...)`.
- **No interpreter hot loop**: `jax_flat` live tick and batch timestep paths execute through JAX JIT/`lax.scan`.
- **Composable formulas**: parse string expressions and support Python-level DSL macro composition.
- **Extensible operators**: registry/plugin model for adding new ops without central branching.
- **Scalable IO**: supports in-memory NumPy arrays and disk-backed memmaps.

## Project layout

- `src/trading_dsl_engine/base/`
  - Shared parser (`parse_formula`), Python DSL constructors, registry metadata, and compile/lower pipeline.
- `src/trading_dsl_engine/numba/`
  - Deprecated Numba built-in op implementations, jitclass state machines, and batch/live runtime helpers.
- `src/trading_dsl_engine/jax/`
  - Deprecated non-flat JAX + Equinox runtime.
- `src/trading_dsl_engine/jax_flat/`
  - Active JAX-flat runtime that lowers supported DSL expressions to a flat operator DAG and executes live ticks/batch scans through JIT-compiled JAX functions.
- `tests/numba/`
  - Deprecated Numba runtime tests; do not run or update unless explicitly requested.
- `tests/jax/`
  - Deprecated non-flat JAX backend tests.
- `tests/jax_flat/`
  - Active JAX-flat behavior, shape, state, and performance tests.

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
- Elementwise grouped RHS graphs such as nested arithmetic, `cumsum`, `ewm`, and `ffill` use a memberwise JAX-flat update path that keeps per-key/per-column state independent without invoking the heavier groupwise mask path reserved for cross-sectional RHS graphs.
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

The active `trading_dsl_engine.jax_flat` backend additionally supports `buffer(shift(x, lag, max_size), min_lag, max_lag)` (or keyword form `buffer(shift(x, lag=..., max_lag=...), min=..., max=...)`). This JAX-flat-only operator emits a lag cube whose last axis is lags `1..max_lag`, preserving shift ring ordering and masking lags below dynamic `min_lag`, above dynamic `lag`, or beyond available history. Vector inputs produce `(time, n_instruments, max_lag)` outputs; matrix/ndarray inputs such as `bspline(...)` preserve their feature axes and append the lag axis, e.g. `(time, n_instruments, n_basis, max_lag)`.

`jax_flat.stateless(fn, ...)` wraps user-supplied stateless JAX callables as compositional variadic operators that still run inside the compiled tick and batch paths. If `output_kind`/`output_width` are omitted, the operator inherits shape metadata from its first child, which is suitable for shape-preserving transforms such as reversing the lag axis:

```python
import jax.numpy as jnp
from trading_dsl_engine.base.dsl import buffer, shift, var
from trading_dsl_engine.jax_flat import compile_formula, stateless

rev = stateless(lambda x: jnp.flip(x, axis=-1), name="rev")
close = var("close")
upper_lag = var("upper_lag")
min_lag = var("min_lag")
runtime = compile_formula(
    rev(buffer(shift(close, lag=upper_lag, max_lag=4), min=min_lag, max=4))
)
```


## Native C++ tick prototype

`trading_dsl_engine.jax_flat.compile_formula(..., cpp=True)` enables the optional native accelerator by default for supported grouped hot paths, while `cpp=False` forces the pure JAX-flat path. `trading_dsl_engine.jax_flat.compile_formula_cpp(...)` lazily imports `trading_dsl_engine.jax_flat.engine_cpp` and exposes an experimental native tick-path runtime for flat formulas where C++ can currently preserve the same streaming semantics as JAX-flat. It compiles the existing shared parser/lowering output into a C++ flattened node table backed by `jax_flat/engine.cpp` + `jax_flat/ops.cpp`; `init_state(n_instruments)` preallocates per-node scratch buffers and state slots, while `tick_into(state, out, *rows)` reuses both the state and caller-owned output row to avoid hot-path Python/JAX allocations. The convenience `tick(...)` method allocates only its returned row.

The native batch helper intentionally stays a repeated tick loop over the same `eval_row` transition rather than a separate vectorized semantic path. It binds contiguous row pointers once, calls the same non-batch evaluator for each row, and uses `__restrict`/flat contiguous buffers so C++ compilers can optimize the row loop without changing streaming state behavior. The benchmark script compiles and warms both C++ and JAX-flat runtimes before timing, so printed results exclude extension import, formula compilation, and JAX first-use compilation. When the optional extension is installed, `JaxFlatRuntime.run_batch(...)` may use the same native flat evaluator for fully supported in-memory dynamic-key groupby programs (including `univ(...)` column partitions) with no caller-supplied state; set `TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL=1` to force the pure JAX path for behavior checks. Native groupby lowering uses a nested RHS node table rather than a per-operator grouped-cumsum branch, so additional scalar/vector-width-1 RHS operators can compose without adding a new grouped hot-path case. If a grouped formula is not yet native-supported, `run_batch` emits a one-time `RuntimeWarning` identifying the unsupported node and automatically falls back to the JAX-flat implementation.

Supported native operators now include inputs, literals, arithmetic/comparison/logical operators, `where`, `fillna`, `abs`, `ln`, `ceil`, `floor`, `round`, `exp`, `sign`, `arctan`, `isnan`, `purify`, `fraction`, `xstd`, `xs_rank`, `xs_sort`, `mean`, `outer`, a row-aligned `einsum` subset, `cat`, `bspline`, `col`, `cumsum`, literal-span `ewm`, static-limit `ffill`, `shift`, vector/`cat`-feature `Ridge` with scalar/vector weights, `get_beta`, `get_preds`, and dynamic-key `groupby(...)` with optional `univ(...)` column partitions for nested scalar/vector-width-1 RHS graphs over `self_` (for example `cumsum(self_)`, `cumsum(cumsum(self_))`, or `add(cumsum(self_), 1)`). Explicit `compile_formula_cpp(...)` still raises `NotImplementedError` for unsupported nodes, while automatic `compile_formula(..., cpp=True)` acceleration warns once and falls back to the default JAX-flat runtime.

Use the warmed comparison helper for quick local measurements:

```bash
python scripts/benchmark_cpp_flat.py --rows 100000 --cols 9 --runs 5
python tests/jax_flat/test_benchmark_groupby_matrix.py --rows 100000 --cols 9 --runs 1 --warmups 1 --assert
```

## Development quickstart

```bash
python -m venv .venv
. .venv/bin/activate
PIP_CACHE_DIR=.pip-cache python -m pip install -e .
pytest -q  # configured to use pytest-xdist with 12 workers
```

The `.venv/` and `.pip-cache/` paths are gitignored so cloud/agent environments can reuse a repo-local virtualenv and wheel/download cache between iterations without committing environment artifacts.

By default, contributors and agents should run only targeted tests for active `jax_flat` and shared DSL functionality. Do not run the full suite, deprecated Numba tests, or deprecated non-flat JAX tests unless the task explicitly asks for them. Active performance tests (opt-in) live under `tests/jax_flat/`:

```bash
RUN_PERF_TESTS=1 pytest -n 0 tests/jax_flat/test_performance.py -q
```

## Notes for future work

- Add graph-level IR + CSE for shared subtrees across multi-feature workflows.
- Expand shape system for richer multi-output model/optimizer nodes.
- Continue reducing memory movement in batch paths for large memmap workloads.
