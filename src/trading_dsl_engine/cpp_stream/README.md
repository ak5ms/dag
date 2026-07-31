# `cpp_stream`

`cpp_stream` is a formula-specialized native backend for large row-major `float64` `.bin` datasets. It consumes the shared `trading_dsl_engine` DSL through the backend-neutral `trading_dsl_engine.ir` frontend and does **not** depend on `jax_flat`.

```text
DSL Expr/string -> neutral IR -> cpp_stream physical lowering -> generated C++20
    -> mmap input rows -> flat native stages -> mmap output rows
```

The first integrated operator set is arithmetic (`add/sub/mul/div`), `cumsum`, `ewm`, `xs_rank`, and canonical `groupby(key_tuple, lhs, rhs_using_self_)`, including composite dynamic keys and one `univ(...)` static column partition. Group RHS graphs may compose the supported arithmetic/stateful/rank operators arbitrarily; nested `groupby` is intentionally rejected for now.

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

## Group keys and dense domains

Existing DSL syntax is retained:

```python
groupby(
    (univ([0], [1, 2], [3, 4, 5], [6, 7, 8]), minute_of_day),
    close / open,
    ewm(cumsum(self_), 21),
)
```

`univ(...)` is compiled to a static lane partition. Dynamic composite keys are stored exactly and mapped through a fixed-capacity open-addressed table. NaNs canonicalize to one key and `-0.0/+0.0` compare/hash identically.

For dense bounded categories, pass a compile hint:

```python
compile_formula(expr, n_instruments=9, key_cardinalities={"minute_of_day": 1440})
```

A single direct input key with a declared cardinality bypasses hashing and indexes its state slot directly. One extra slot preserves valid NaN-key semantics.

## Allocation and state layout

The generated hot path uses compile-time `std::array` state/scratch and no dynamic allocation. Input/output files are mmap mappings. Formula compilation and mapping setup may allocate normally; no allocation occurs per row. Stateful grouped operators store `[group_slot][lane]` scalar state. The static `univ` partition is not multiplied into that state because a lane's static partition cannot change; cross-sectional grouped rank includes the static partition in its group identity.

`default_group_capacity=64` bounds unknown dynamic-key state unless the formula's `groupby(..., capacity=...)` overrides it. Use dense cardinality hints for domains such as minute-of-day instead of provisioning a hash table for 1,440 known integer values.

## Backend-neutral IR

`trading_dsl_engine.ir` owns parser/DSL expansion, canonical groupby decomposition, static universe resolution, capture extraction for grouped RHS graphs, and semantic op parameters. It deliberately imports neither JAX nor C++ runtime code. Backend-specific scratch liveness, state layout, dense-vs-hash routing, C++ types, mmap behavior, and code generation stay under `cpp_stream`.

`jax_flat` is intentionally unchanged by this integration. A later migration can make it consume the neutral IR without making `cpp_stream` depend on JAX.

## Current platform scope

The generated mmap runner currently targets POSIX/Linux and uses a C++20 compiler (`g++` by default). Windows support is not yet wired into the file-mapping/codegen layer.
