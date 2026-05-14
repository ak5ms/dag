# trading-dsl-engine

A high-performance Python DSL engine for streaming trading features on aligned minutely NumPy data.

This repository compiles formulas (string DSL or Python-composed DSL calls) into nested Numba `jitclass` state machines that support both live incremental updates and batch execution.

## Core goals

- **Streaming-first stateful computation**: each op follows `on_data(...)` + `emit(...)`.
- **No interpreter hot loop**: runtime timestep loop executes in compiled Numba code.
- **Composable formulas**: parse string expressions and support Python-level DSL macro composition.
- **Extensible operators**: registry/plugin model for adding new ops without central branching.
- **Scalable IO**: supports in-memory NumPy arrays and disk-backed memmaps.

## Project layout

- `src/trading_dsl_engine/parser.py`
  - Formula parser (`parse_formula`) using Python AST with strict validation.
- `src/trading_dsl_engine/dsl.py`
  - Python DSL constructors (`add`, `div`, `shift`, `ewm`, `xs_rank`, etc.), composed helpers like `diff`, and `DSLFunctionRegistry`.
- `src/trading_dsl_engine/registry.py`
  - Operator metadata and registration primitives.
- `src/trading_dsl_engine/ops.py`
  - Built-in op implementations and generic op builders.
- `src/trading_dsl_engine/compiler.py`
  - Compile path from expression to `CompiledFormula` jitclass artifact, with CSE hash/cache stats.
- `src/trading_dsl_engine/engine.py`
  - Runtime `StreamingFeatureEngine` jitclass and batch/live helpers.
- `tests/`
  - Parser, composition, runtime correctness, shape, state persistence, and performance tests.

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


## Schema-bound compiled programs

`compile_program(...)` is the new primary runtime API for fixed-shape production workloads. It compiles a string formula or Python-composed `Expr` against an explicit `Schema` that declares input names, dtypes, time/instrument layout, instrument count, optional column names, static universes, and bounded key domains. The resulting `CompiledProgram` exposes a deterministic `output_schema` and a typed `runtime_plan` before the first tick is processed.

```python
from trading_dsl_engine import Float64, Instrument, Schema, Time, compile_program

program = compile_program(
    "xs_rank(ewm(div(close, open), 21))",
    schema=Schema(
        inputs={
            "open": Float64[Time, Instrument],
            "close": Float64[Time, Instrument],
        },
        n_instruments=150,
        layout="time_major",
    ),
)

bound = program.bind(open=open_arr, close=close_arr)
out = bound.run_batch(out_path="features.memmap")

state, workspace = program.initialize()
program.step(state, {"open": open_tick, "close": close_tick}, tick_out)
```

The fast path lowers formulas to typed IR, performs compile-time shape inference, plans persistent state/scratch/output buffers, records allocation counters, and runs batch mode directly over aligned positional input arrays rather than copying each tick into an intermediate live frame. Stateless elementwise chains are represented as fusible plan regions and execute into preallocated buffers. Stateful operators such as `ewm`, `cumsum`, `shift`, `xs_rank`, and `rolling_quantile` allocate their state and scratch buffers during `program.new_state()`/`program.new_workspace()` instead of probing shapes on the first tick.

The compatibility APIs remain available. `build_engine(..., schema=schema)` returns a `CompiledProgram`, and `run_batch_from_mapping(...)`/`update_from_mapping(...)` delegate to the program runtime when they receive one. Operators or dynamic behavior that cannot yet be planned from schema are explicitly routed to the existing jitclass compatibility runtime with a warning, preserving correctness while making fallback behavior visible.

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
- `groupby(key, op)` runs the full `op` subtree as partitioned state by key for any scalar/vector/matrix-emitting op and routes each tick to the keyed op instance.
- `groupby(key, lhs, op_using_self_)` evaluates `lhs` once in the outer stream, then runs only `op_using_self_` as keyed state over the emitted `lhs` values; the local op expression must reference the outer value through the `self_` placeholder, e.g. `groupby(day, get_preds(Ridge(...)), cumsum(self_))`.
- Python-composed formulas can spell the local-op form as `lhs.groupby(key).apply(op_fn, *args)`, `lhs.groupby(key).apply(op_expr_using_self_)`, or `lhs.groupby(key).some_op(...)`; for example, `reg.groupby(day).cumsum()` lowers to `groupby(day, reg, cumsum(self_))`.
- `groupby(univ(...), op)` runs the same scalar/vector/matrix op independently on static column universes and scatters each group result back to its member columns. Universe groups can be built in Python, e.g. `groupby(univ(["6E", "6C"], ["6A"]), mean(close))`, or in string formulas with `column_names=[...]`; string formulas also accept integer column indexes such as `univ([0, 1], [2])`.
- `mean(x)` emits the NaN-skipping mean of a scalar/vector/matrix input as a scalar, which is useful inside universe grouping to broadcast per-group means.

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

## Development quickstart

```bash
python -m pip install -e .
python -m pip install pytest numpy numba
pytest -q
```

Performance tests (opt-in):

```bash
RUN_PERF_TESTS=1 pytest tests/test_performance.py -q
```

## Notes for future work

- Add graph-level IR + CSE for shared subtrees across multi-feature workflows.
- Expand shape system for richer multi-output model/optimizer nodes.
- Continue reducing memory movement in batch paths for large memmap workloads.
