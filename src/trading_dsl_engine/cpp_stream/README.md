# `cpp_stream`

`cpp_stream` is a formula-specialized native backend for large row-major datasets. It consumes the shared `trading_dsl_engine` DSL through the backend-neutral `trading_dsl_engine.ir` frontend and does **not** depend on `jax_flat`.

```text
DSL Expr/string -> neutral IR -> cpp_stream physical lowering -> generated C++20
    -> mmap input rows -> flat native stages -> mmap output rows
```

The first integrated operator set is arithmetic (`add/sub/mul/div/mod/floor`), `cumsum`, `ewm`, `xs_rank`, and canonical `groupby(key_tuple, lhs, rhs_using_self_)`, including composite dynamic keys and one `univ(...)` static column partition. Group RHS graphs may compose the supported operators arbitrarily; nested `groupby` is intentionally rejected for now.

## Raw file API

```python
from trading_dsl_engine.cpp_stream import compile_formula

runtime = compile_formula(
    "xs_rank(ewm(close / open, 21))",
    n_instruments=9,
)

result = runtime.run_files(
    {"close": "/data/close.bin", "open": "/data/open.bin"},
    out_path="/data/alpha.bin",
    async_writeback_mb=64,
)
```

When `input_types` is omitted, raw inputs are row-major `float64` with `n_instruments` values per row. Typed headerless inputs can be declared with `InputTypeSpec`. The root output is currently a raw row-major `float64` vector file.

## Typed mmap `.npy` API

`compile_npy_formula` reads each NumPy header before native compilation. Dtype and row width become compile-time C++ template arguments; execution then maps the payload and passes its pointer directly to C++ without copying.

```python
from trading_dsl_engine.cpp_stream import compile_npy_formula

paths = {
    "_ev_ts": "/data/_ev_ts.npy",  # int64, shape (rows,)
    "close": "/data/close.npy",    # float64, shape (rows, 9)
}

runtime = compile_npy_formula(
    "close + _ev_ts",
    paths,
    n_instruments=9,
)

result = runtime.run_npy_files(paths, out_path="/data/result.bin")
```

Supported input dtypes are `float32`, `float64`, `int32`, `int64`, `uint32`, and `uint64`. Arrays must be native/little-endian, C-contiguous, and one- or two-dimensional:

- `(rows,)` and `(rows, 1)` are row-scalar and broadcast across lanes;
- `(rows, n_instruments)` is a normal vector input.

`inspect_npy` exposes `dtype`, `shape`, `data_offset`, `rows`, `row_width`, and `row_scalar`. `mmap_npy` returns the metadata plus a live `np.memmap` payload view.

## Per-key descriptors

Each dynamic group key can carry its own semantic and physical hints:

```python
from trading_dsl_engine import Key

groupby(
    (
        univ([0], [1, 2], list(range(3, 9))),
        Key(
            expr=var("minute"),
            num_keys=60,
            offset=0,
            row_scalar=True,
            dtype="int64",
        ),
    ),
    var("close"),
    ewm(cumsum(self_), 3),
)
```

The hints do not change key equality semantics:

- `num_keys=K` declares consecutive integer categories `[offset, offset + K)`;
- `offset` sets the first valid category;
- `row_scalar=True` asserts lane invariance;
- `dtype` records the semantic/input type expectation.

NaN remains an additional valid category. An out-of-range, nonintegral, or infinite value fails native execution rather than silently aliasing another state slot.

A tuple can contain independently described keys:

```python
groupby(
    (
        univ([0, 1], [2, 3, 4]),
        Key(var("venue"), num_keys=3, offset=10, dtype="int32"),
        Key(var("bucket"), num_keys=4, row_scalar=True, dtype="uint32"),
    ),
    var("close"),
    cumsum(self_),
)
```

If every dynamic key has `num_keys`, lowering uses one mixed-radix dense slot. The capacity is:

```text
product(num_keys_i + 1)
```

where each `+1` reserves that key's NaN digit. Otherwise, the complete tuple is stored exactly in the fixed-capacity hash resolver.

When every key in a tuple is row-scalar, group resolution evaluates the tuple once and broadcasts one slot. For `.npy` inputs, shape-derived row-scalar information is propagated through pure arithmetic. Thus `_ev_ts -> minute` stages compile with lane count one instead of computing the same calendar expression for every instrument.

The older global `key_cardinalities` argument remains for compatibility with direct input keys, but `Key(...)` is the preferred interface because metadata remains attached to the exact key expression and supports composite tuples.

## Calendar fields

Calendar aliases are derived from the canonical `_ev_ts` microseconds-since-epoch field:

```python
var("minute")
```

expands to the existing DSL `minute(_ev_ts)` expression. The current definition is minute within the hour (`0..59`). With an `int64 (rows,)` `_ev_ts.npy` input and the `Key` descriptor above, the input is loaded once per row, calendar stages execute at width one, and dense resolution avoids hashing.

Input loads preserve the `.npy` element type, while the current arithmetic operator layer converts numeric values to `float64`. Fully typed integer calendar arithmetic is a subsequent IR/codegen extension; the reader already supplies the dtype information needed for it.

## Code generation

Python lowering produces typed C++ template arguments and immutable template views. `python/templates/runner.cpp.j2` owns translation-unit structure, grouped inner-plan declarations, mmap setup, stage setup, the native row loop, and asynchronous writeback submission.

`Jinja2>=3.1` is a project dependency, and the `.j2` template is included in package data so code generation works from installed wheels as well as editable checkouts.

## Operator-agnostic group execution

There are no `GroupedCumsumNode`, `GroupedEwmNode`, `GroupedXsRankNode`, or other grouped operator classes. `groupby.hpp` owns only key resolution and invocation of an ordinary inner plan.

Every generated operator receives one final execution-scope template argument:

```cpp
stackdsl::DirectExecution<N>
stackdsl::GroupedExecution<N, Capacity>
```

Codegen emits the same node type inside and outside groupby. Stateful nodes obtain storage size and addresses through `Execution::state_size` and `Execution::state_index(ctx, lane)`. Cross-sectional nodes obtain group identity through `Execution::rank_group(ctx, lane)`. Stateless nodes accept the same scope and ignore it. Once an operator has its normal C++ implementation and one codegen mapping, it works inside groupby without a second class or grouped codegen branch.

## Allocation and state layout

The generated hot path uses compile-time `std::array` state/scratch and no dynamic allocation. Formula compilation and mapping setup may allocate normally; no allocation occurs per row. Grouped state uses `[group_slot][lane]`. The static `univ` partition is not multiplied into lane-local state because a lane's static partition cannot change; cross-sectional rank includes the static partition in its group identity.

`default_group_capacity=64` bounds unknown dynamic-key state unless the formula's `groupby(..., capacity=...)` overrides it.

Reusable output files are not truncated when their existing size is already correct. Every row is overwritten, so retaining the extent avoids repeated page-allocation noise without changing output semantics. `async_writeback_mb` requests nonblocking kernel writeback and does not call `fsync` or synchronous `msync` before returning.

## Performance regression checks

The default benchmark uses typed `.npy` inputs and the hinted timestamp-minute key:

```bash
python scripts/benchmark_cpp_stream.py
```

Other cases are selected through `CPP_STREAM_BENCH_CASE`. An environment-specific regression threshold can be supplied through `CPP_STREAM_BENCH_MIN_MROWS`. Detailed methodology and architecture comparisons are documented in [`PERFORMANCE.md`](PERFORMANCE.md).

## Backend-neutral IR

`trading_dsl_engine.ir` owns parser/DSL expansion, canonical groupby decomposition, per-key specifications, static universe resolution, capture extraction for grouped RHS graphs, and semantic operator parameters. It imports neither JAX nor C++ runtime code. Backend-specific scratch liveness, state layout, dense-vs-hash routing, C++ types, mmap behavior, and code generation stay under `cpp_stream`.

`jax_flat` remains unchanged. A later migration can make it consume the neutral IR without making `cpp_stream` depend on JAX.

## Current platform scope

The generated mmap runner currently targets POSIX/Linux and uses a C++20 compiler (`g++` by default). Windows support is not yet wired into the file-mapping/codegen layer.
