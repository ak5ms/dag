# `cpp_stream`

`cpp_stream` is a formula-specialized native backend for large row-major `float64` `.bin` datasets. It consumes the shared `trading_dsl_engine` DSL through the backend-neutral `trading_dsl_engine.ir` frontend and does **not** depend on `jax_flat`.

```text
DSL Expr/string -> neutral IR -> cpp_stream physical lowering -> generated C++20
    -> mmap input rows -> flat native stages -> mmap output rows
```

The first integrated operator set is arithmetic (`add/sub/mul/div/mod/floor`), `cumsum`, `ewm`, `xs_rank`, and canonical `groupby(key_tuple, lhs, rhs_using_self_)`, including composite dynamic keys and one `univ(...)` static column partition. Group RHS graphs may compose the supported operators arbitrarily; nested `groupby` is intentionally rejected for now.

## File API

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

Each input file is raw row-major `float64` with exactly `n_instruments` values per row. Input paths are bound by the neutral IR's `input_names`, not encoded into the formula. The root output is currently a vector and is written as the same row-major `float64` layout.

`async_writeback_mb` requests nonblocking kernel writeback of completed output ranges. The runtime does not call `fsync` or synchronous `msync` before returning.

## Code generation

Python lowering produces typed C++ template arguments and immutable template views. It does not construct complete C++ functions or the row loop through operator-specific string concatenation. `python/templates/runner.cpp.j2` owns translation-unit structure, grouped inner-plan declarations, mmap setup, stage setup, the native row loop, and asynchronous writeback submission.

`Jinja2>=3.1` is a project dependency, and the `.j2` template is included in package data so code generation works from installed wheels as well as editable checkouts.

## Operator-agnostic group execution

There are no `GroupedCumsumNode`, `GroupedEwmNode`, `GroupedXsRankNode`, or other grouped operator classes. `groupby.hpp` owns only key resolution and invocation of an ordinary inner plan.

Every generated operator receives one final execution-scope template argument:

```cpp
stackdsl::DirectExecution<N>
stackdsl::GroupedExecution<N, Capacity>
```

Codegen emits the same node type inside and outside groupby. For example, both forms use `EwmNode`; only the execution scope changes:

```cpp
EwmNode<N, In, Out, SpanBits, MinPeriods, IgnoreNa, Adjust,
        DirectExecution<N>>

EwmNode<N, In, Out, SpanBits, MinPeriods, IgnoreNa, Adjust,
        GroupedExecution<N, Capacity>>
```

Stateful nodes obtain storage size and addresses through `Execution::state_size` and `Execution::state_index(ctx, lane)`. Cross-sectional nodes obtain group identity through `Execution::rank_group(ctx, lane)`. Stateless nodes accept the same scope and ignore it. Therefore, once an operator has its normal C++ implementation and one codegen mapping, it works inside groupby without a second class or grouped codegen branch.

## Group keys and calendar fields

Existing DSL syntax is retained. Calendar aliases are derived from the canonical `_ev_ts` microseconds-since-epoch field:

```python
groupby(
    (univ([0], [1, 2], list(range(3, 9))), var("minute")),
    var("close"),
    ewm(cumsum(self_), 3),
)
```

Here `var("minute")` expands to the existing DSL `minute(_ev_ts)` expression. The current DSL definition is minute within the hour (`0..59`).

`univ(...)` is compiled to a static lane partition. Dynamic composite keys are stored exactly and mapped through a fixed-capacity open-addressed table. NaNs canonicalize to one key and `-0.0/+0.0` compare/hash identically.

For a direct bounded categorical input, a compile hint can bypass hashing:

```python
compile_formula(expr, n_instruments=9, key_cardinalities={"some_key": 1440})
```

A single direct input key with a declared cardinality indexes its state slot directly. One extra slot preserves valid NaN-key semantics. Domain propagation for derived expressions such as `minute(_ev_ts)` remains a planned neutral-IR optimization.

## Allocation and state layout

The generated hot path uses compile-time `std::array` state/scratch and no dynamic allocation. Input/output files are mmap mappings. Formula compilation and mapping setup may allocate normally; no allocation occurs per row. Grouped state uses `[group_slot][lane]`. The static `univ` partition is not multiplied into lane-local state because a lane's static partition cannot change; cross-sectional rank includes the static partition in its group identity.

`default_group_capacity=64` bounds unknown dynamic-key state unless the formula's `groupby(..., capacity=...)` overrides it.

Reusable output files are not truncated when their existing size is already correct. Every row is overwritten, so retaining the extent avoids repeated page-allocation noise without changing output semantics.

## Performance regression checks

Run the full 5M x 9 mmap benchmark with:

```bash
python scripts/benchmark_cpp_stream.py
```

An environment-specific regression threshold can be supplied through `CPP_STREAM_BENCH_MIN_MROWS`. Detailed methodology and architecture comparisons are documented in [`PERFORMANCE.md`](PERFORMANCE.md).

## Backend-neutral IR

`trading_dsl_engine.ir` owns parser/DSL expansion, canonical groupby decomposition, static universe resolution, capture extraction for grouped RHS graphs, and semantic op parameters. It deliberately imports neither JAX nor C++ runtime code. Backend-specific scratch liveness, state layout, dense-vs-hash routing, C++ types, mmap behavior, and code generation stay under `cpp_stream`.

`jax_flat` is intentionally unchanged by this integration. A later migration can make it consume the neutral IR without making `cpp_stream` depend on JAX.

## Current platform scope

The generated mmap runner currently targets POSIX/Linux and uses a C++20 compiler (`g++` by default). Windows support is not yet wired into the file-mapping/codegen layer.
