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
  - Python DSL constructors (`add`, `div`, `ewm`, `xs_rank`, etc.) and `DSLFunctionRegistry`.
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

# batch run
out2d = run_batch_from_mapping(engine, {"open": open_2d, "close": close_2d}, chunk_size=4096)  # shape (time, n_instruments) for vector outputs
```

## DSL composition

You can define reusable macro-like composed functions with an explicit registry namespace:

```python
from trading_dsl_engine import DSLFunctionRegistry, register_dsl_function, add, div

my_registry = DSLFunctionRegistry()

@register_dsl_function("hlc3", registry=my_registry)
def hlc3(high, low, close):
    return div(add(add(high, low), close), 3.0)
```

Then compile with `compile_formula(..., dsl_registry=my_registry)`.

The returned artifact includes `stats` (`expanded_nodes`, `cache_hits`) so compile-time common-subexpression hashing behavior can be validated.

## Data contract

- Inputs are aligned 2D arrays with shape `(time, n_instruments)`.
- Live `update` expects 1D vectors with shape `(n_instruments,)`.
- Some ops may emit matrix outputs (e.g., `outer`), and shape is represented in metadata.

## NaN semantics (current)

- Binary ops propagate NaN values.
- `div` returns NaN on divide-by-zero.
- `ewm` skips updates for NaN inputs and can recover from NaN state.
- `xs_rank` ranks only valid values and emits NaN where input is NaN.

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
