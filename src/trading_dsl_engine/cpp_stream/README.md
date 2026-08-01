# `trading_dsl_engine.cpp_stream`

`cpp_stream` is a formula-specialized C++20 streaming backend. It consumes the
backend-neutral `trading_dsl_engine.ir` graph, generates a typed translation unit
with Jinja, compiles a cached shared library, mmaps inputs, and executes the full
row loop in native code.

It does not depend on `jax_flat`.

```text
shared DSL / parser
        |
        v
backend-neutral ir.Program
        |
        v
cpp_stream physical lowering
  - native dtypes
  - row-scalar propagation
  - scratch liveness
  - execution scope
  - dense/hash key policy
        |
        v
Jinja C++ translation unit
        |
        v
mmap inputs -> native row loop -> mmap output
```

## Basic use

```python
from trading_dsl_engine.cpp_stream import compile_npy_formula

paths = {
    "close": "/data/close.npy",
    "open": "/data/open.npy",
}

runtime = compile_npy_formula(
    "xs_rank(ewm(close / open, 21))",
    paths,
    n_instruments=9,
)

result = runtime.run_npy_files(paths, out_path="/data/alpha.bin")
print(result.rows_per_second)
```

`.npy` files are mapped without copying. Their headers determine dtype, row count,
and whether each input is row-scalar or vector-valued before native compilation.

Supported input dtypes:

```text
float32, float64, int32, int64, uint32, uint64
```

Supported C-order shapes:

```text
(rows,)
(rows, 1)
(rows, n_instruments)
```

Object, structured, big-endian, Fortran-order, and higher-rank arrays are rejected.
Headerless raw files remain available through `compile_formula(..., input_types=...)`
and `run_files(...)`.

## Native typing

Mapped values are not eagerly converted to `float64`:

```cpp
InputSrc<Index, ValueType, RowWidth>
SlotSrc<Index, ValueType, RowScalar>
SlotDst<Index, ValueType>
```

Typed arithmetic retains native integer or floating-point types when the operation
permits it. Conversion occurs only at an operator whose result requires another
type, or at an explicitly float64 statistical/output boundary. This preserves exact
integer key identity, including values above `2**53`.

## Key descriptors

Grouping hints are attached to each dynamic key:

```python
from trading_dsl_engine import Key

Key(
    expr=var("minute"),
    num_keys=60,
    offset=0,
    row_scalar=True,
    dtype="int64",
)
```

`num_keys` is the number of consecutive bounded categories. `offset` is the first
valid category; dense routing uses `value - offset` as the zero-based digit.

```python
Key(var("month"), num_keys=12, offset=1)  # values 1..12
Key(var("venue"), num_keys=3, offset=10)  # values 10,11,12
```

`row_scalar=True` asserts that one value applies to every instrument in a row, so
the expression and resolver execute once per row. `dtype` is validation metadata,
not permission to cast an input.

When every key in a tuple has `num_keys`, cpp_stream uses mixed-radix dense routing.
Otherwise it hashes the exact complete tuple.

## Operator-agnostic groupby

Grouping changes the execution environment, not the operator class:

```cpp
DirectExecution<N>
GroupedExecution<N, Capacity, PartitionCount>
```

Every node receives one of these as its final template parameter. There are no
`GroupedEwmNode`, `GroupedCumsumNode`, `GroupedRidgeNode`, or `FastGrouped*` types.

The execution scope supplies generic state and cross-sectional addressing:

```cpp
Execution::state_index(ctx, lane)
Execution::rank_group(ctx, lane)
Execution::cross_group(ctx, lane)
Execution::state_size
Execution::cross_state_size
```

`groupby.hpp` owns only key resolution, grouped-context construction, and invocation
of the normal inner plan. It contains no indicator or regression implementation.

## `cat`

`cat(...)` has compile-time feature width and can be a matrix root:

```python
runtime = compile_npy_formula(
    "cat(x1, x2, x3)",
    paths,
    n_instruments=9,
)
```

Its output has logical shape:

```text
(rows, n_instruments, 3)
```

When `cat` feeds Ridge, it is a zero-copy compile-time `FeatureList`. Nested cat
expressions are flattened, and Ridge reads original mapped/scratch sources directly;
an intermediate `N x K` matrix is not materialized.

These formulas compile to identical native source:

```python
Ridge(cat(x1, x2, x3), y=y, hl=64, lambda_=0.1)
Ridge(x1, x2, x3, y=y, hl=64, lambda_=0.1)
```

## Ridge

Project a Ridge object with `get_preds` or `get_beta`:

```python
preds = "get_preds(Ridge(cat(x1, x2, x3), y=y, hl=64, lambda_=0.1))"
beta = "get_beta(Ridge(cat(x1, x2, x3), y=y, hl=0, lambda_=0.1))"
```

The generated node is always one generic template:

```cpp
RidgeNode<
    N,
    FeatureList<...>,
    Y,
    Weights,
    Out,
    AlphaBits,
    LambdaBits,
    Nonnegative,
    Stateful,
    Projection,
    Execution
>
```

There is one direct/grouped implementation with compile-time `K` and fixed
`std::array` storage. No allocation occurs during row execution.

Implemented semantics:

- weighted pairwise-missing `X'WX` and `X'Wy` moments;
- per-moment missing-data timing for stateful Ridge;
- prior-beta predictions for positive half-life;
- current-row solution when `hl <= 0` or nonfinite;
- regularization `XX + lambda * diag(diag(XX))`;
- Cholesky solve, pivoted solve, and pseudoinverse fallback;
- optional generic nonnegative coordinate-descent solve.

For fully finite panels, the same node uses synchronized group-level moment timing
and a fixed-size update without per-pair validity branches. Missing-data rows use the
complete pairwise path. This is a semantic data-validity branch, not a shape-specific
or groupby-specific implementation.

Output shapes:

```text
get_preds(...)                 (rows, n_instruments)
get_beta(...) direct           (rows, K)
get_beta(...) inside groupby   (rows, n_instruments, K)
```

Current Ridge restrictions:

- `hl` and `lambda_` must be compile-time numeric literals;
- weights may be scalar or instrument-vector, not an `N x N` matrix;
- a raw Ridge object cannot be written; select beta or predictions;
- non-root beta/matrix values are not yet materialized for arbitrary downstream
  matrix operations.

## Performance baseline

GitHub-hosted Ubuntu runner, 5,000,000 rows x 9 instruments, float64 mmap inputs,
`-O3 -march=native -mtune=native -flto`, one warmup, ten measured runs:

| Case | Median throughput |
| --- | ---: |
| `cat(x1,x2,x3)` root, 27 doubles written per row | 11.307 M rows/s |
| Stateful K=3 Ridge predictions using `cat` | 6.377 M rows/s |
| Stateful K=3 Ridge predictions using separate args | 6.367 M rows/s |
| Stateless K=3 beta | 9.169 M rows/s |
| One-group grouped stateful Ridge | 6.176 M rows/s |
| Three-group grouped stateful Ridge | 2.787 M rows/s |

`cat` and separate arguments produced the same generated source and checksum. The
one-group grouped form is about 3% below direct execution. The three-group case
performs three independent K=3 solves per row, so its lower throughput is real
additional statistical work rather than an operator-specific groupby layer.

Reproduce the full matrix with:

```bash
CPP_STREAM_RIDGE_CASE=all \
python scripts/benchmark_cpp_stream_ridge.py
```

The script defaults to 5M x 9, one warmup, and ten measured executions. Compilation,
input generation, and warmup are excluded from reported native runtime.

## Current native operators

```text
add, sub, mul, div, mod, floor
cumsum
ewm
xs_rank
cat
Ridge + get_beta/get_preds
groupby with univ and dynamic tuple keys
```

## Compilation cache

The cache key includes generated source, packaged headers, compiler identity,
compile/link flags, platform/machine, and Python ABI. The default cache is:

```text
~/.cache/trading_dsl_engine/cpp_stream
```

Override it with `TRADING_DSL_ENGINE_CPP_STREAM_CACHE`.

## Validation

Focused tests compile and execute generated native code for typed `.npy` inputs,
integer key identity, mixed-radix routing, cat layout, stateless/stateful Ridge,
pairwise missing data, finite-to-NaN transitions, nonnegative Ridge, and grouped
Ridge. The focused workflow does not represent the full repository/JAX suite.
