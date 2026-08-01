# `trading_dsl_engine.cpp_stream`

`cpp_stream` is a formula-specialized C++20 streaming backend. It consumes the
backend-neutral `trading_dsl_engine.ir` graph, generates a typed translation unit,
compiles a cached shared library, prepares each input through an independent source
adapter, and executes the row loop in native code. It does not depend on `jax_flat`
at runtime.

```text
shared DSL / parser
        -> backend-neutral ir.Program
        -> cpp_stream physical lowering
        -> generated C++20
        -> heterogeneous source adapters
        -> one typed-pointer native entrypoint
        -> mmap output
```

## One compile API and one run API

```python
from trading_dsl_engine.cpp_stream import compile_formula

runtime = compile_formula(
    "xs_rank(ewm(close / open, 21))",
    {
        "close": "/data/close.npy",
        "open": "/data/open.npy",
    },
    n_instruments=9,
)
runtime.run(out_path="/data/alpha.bin")
```

There is no `.npy`-specific compiler or runner. Every input independently selects a
source adapter from its object type, URI scheme, file extension, or explicit adapter
name. A single formula may therefore mix formats:

```python
from trading_dsl_engine.cpp_stream import InputTypeSpec, source

runtime = compile_formula(
    "left + right",
    {
        "left": "/data/left.npy",
        "right": source(
            "/data/right.bin",
            input_type=InputTypeSpec("float64", 9),
        ),
    },
    n_instruments=9,
)
runtime.run(out_path="/data/result.bin")
```

Built-in adapters currently cover zero-copy C-order `.npy`, headerless `.bin`/`.raw`
with explicit metadata, and C-contiguous in-memory NumPy arrays. Custom adapters can
match extensions such as `.parquet`, URI schemes such as `tcp://`, or application
source objects through `register_source_adapter(...)`. See `SOURCES.md`.

Sources supplied at compilation are bound to the runtime. `runtime.run(new_sources,
...)` may replace them with another compatible mapping, including different source
formats with the same dtype and per-row shape.

Any positive C-order per-row tensor shape is supported: `(rows,)`, `(rows, 1)`,
`(rows, N)`, `(rows, N, K)`, `(rows, B, N, K)`, and higher ranks. `(rows,)` and
`(rows, 1)` are row scalars. Supported native dtypes are `float32`, `float64`,
`int32`, `int64`, `uint32`, and `uint64`.

## NumPy-style `einsum`

The canonical call order matches NumPy:

```python
einsum("ij,jk->ik", left, right)
einsum("...ij,...jk->...ik", left, right)
einsum("ij,ij->", left, right)
einsum("ii->i", square)
einsum("ij,kj,kl->il", a, b, c, optimize="optimal")
```

Supported string-subscript behavior includes arbitrary case-sensitive ASCII labels,
implicit and explicit outputs, scalar operands and reductions, arbitrary rank,
diagonals, permutations, outer products, ellipsis broadcasting, and optimized n-ary
contraction paths. Named labels require equal dimensions; broadcasting is enabled
through `...`, matching NumPy.

The default is `optimize=False`. `True`, `"greedy"`, and `"optimal"` are supported.
`"optimal"` exhaustively searches paths through eight operands and falls back to
greedy for larger expressions.

Subscripts are parsed once in the neutral IR and lowered to static unary/binary
contraction stages. Generated C++ contains no runtime string parser, shape dispatch,
or path search. Contiguous inner reductions use bulk loads and FMA loops; generic
mapped loops cover diagonals, broadcasting, permutations, and arbitrary contraction
axes.

The native API does not yet implement NumPy's integer-sublist calling form,
precomputed path lists, `out=`, `dtype=`, `order=`, `casting=`, or writeable-view
semantics. Native einsum accumulation/output is currently `float64`.

## Execution model

Every operator has one native implementation and receives its execution scope as
the final template argument:

```cpp
DirectExecution<N>
GroupedExecution<N, Capacity, PartitionCount>
```

There are no `GroupedFooNode` or formula-specific fast-path classes. `groupby.hpp`
contains only key resolution, grouped-context construction, and inner-plan
invocation. No operator allocates from the heap during `on_data`.

`cat(...)`, RBF bases, coefficient matrices, and einsum use compile-time dimensions.
Lazy basis sources and nested Cat expressions flatten into `FeatureList<Sources...>`
so consumers read original inputs directly. Arbitrary intermediates use compact
fixed-size tensor scratch only when a contraction path requires them.

## `riskmodel.roll_rets`

```python
from flows.riskmodel import roll_rets
from trading_dsl_engine.cpp_stream import compile_formula

runtime = compile_formula(
    roll_rets,
    paths,
    n_instruments=9,
    default_group_capacity=4096,
)
runtime.run(out_path="roll_rets.bin")
```

The generated plan contains 50 scalar/vector scratch slots and one six-wide matrix
scratch slot. RBF and future-RBF basis values remain lazy. A native end-to-end test
compares the exact expression against JAX-flat with finite-output checks and
`rtol=2e-9`, `atol=2e-9`, equal-NaN semantics.

## 5M x 9 benchmarks

GitHub-hosted Ubuntu runner, GCC C++20 with
`-O3 -march=native -mtune=native -flto`, one warmup and ten measured executions:

| Workload | Median throughput |
| --- | ---: |
| `einsum("nf,nf->n", ...)`, six features | 12.076 M rows/s |
| equivalent ellipsis reduction | 12.097 M rows/s |
| scalar reduction `einsum("n,n->", ...)` | 85.027 M rows/s |
| three-operand contraction, `optimize=False` | 5.533 M rows/s |
| same contraction, greedy | 11.605 M rows/s |
| same contraction, optimal | 11.624 M rows/s |
| full `flows.riskmodel.roll_rets` | 0.865675 M rows/s |

For the n-ary case, planning reduced estimated work from 324 to 72 operations per
row and reduced the largest intermediate scratch width from 9 to 2. Checksums were
identical and every sampled output was finite. Full distributions and checksums are
in `PERFORMANCE.md`.

## Compilation cache

The cache key includes generated source, packaged headers, compiler identity,
compile/link flags, platform/machine, and Python ABI. The default cache is:

```text
~/.cache/trading_dsl_engine/cpp_stream
```

Override it with `TRADING_DSL_ENGINE_CPP_STREAM_CACHE`.
