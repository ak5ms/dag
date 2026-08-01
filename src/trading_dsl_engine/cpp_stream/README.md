# `trading_dsl_engine.cpp_stream`

`cpp_stream` is a formula-specialized C++20 streaming backend. It consumes the
backend-neutral `trading_dsl_engine.ir` graph, generates a typed translation unit,
compiles a cached shared library, mmaps inputs, and executes the full row loop in
native code. It does not depend on `jax_flat` at runtime.

```text
shared DSL / parser
        -> backend-neutral ir.Program
        -> cpp_stream physical lowering
        -> generated C++20
        -> mmap inputs -> native row loop -> mmap output
```

## Basic use

```python
from trading_dsl_engine.cpp_stream import compile_npy_formula

runtime = compile_npy_formula(
    "xs_rank(ewm(close / open, 21))",
    {"close": "/data/close.npy", "open": "/data/open.npy"},
    n_instruments=9,
)
runtime.run_npy_files(
    {"close": "/data/close.npy", "open": "/data/open.npy"},
    out_path="/data/alpha.bin",
)
```

`.npy` inputs are mapped without copying. Any positive C-order per-row tensor shape
is supported: `(rows,)`, `(rows,1)`, `(rows,N)`, `(rows,N,K)`, `(rows,B,N,K)`, and
higher ranks. `(rows,)` and `(rows,1)` remain row-scalar. Supported dtypes are
`float32`, `float64`, `int32`, `int64`, `uint32`, and `uint64`.

## NumPy-style `einsum`

The canonical call order matches NumPy:

```python
einsum("ij,jk->ik", left, right)
einsum("...ij,...jk->...ik", left, right)
einsum("ij,ij->", left, right)
einsum("ii->i", square)
einsum("ij,kj,kl->il", a, b, c, optimize="optimal")
```

The old project-local order remains accepted for compatibility:

```python
einsum(left, right, "ij,jk->ik")
```

Supported string-subscript behavior includes:

- arbitrary case-sensitive ASCII labels such as `ij`, `nf`, or `Qx`;
- implicit and explicit output;
- scalar operands and scalar reductions;
- arbitrary operand/output rank;
- repeated-label diagonal extraction;
- permutations and outer products;
- ellipsis expansion and NumPy-compatible ellipsis broadcasting;
- raw arbitrary-rank mmap operands without copying;
- lazy Cat/RBF feature matrices without eager materialization.

As in NumPy, named labels must have equal dimensions; broadcasting is enabled only
through `...`. The default is `optimize=False`. `True`, `"greedy"`, and `"optimal"`
are supported. `"optimal"` exhaustively searches paths through eight operands and
falls back to greedy for larger contractions.

The frontend parses and validates subscripts once, canonicalizes labels to integer
axis maps, and creates a static unary/binary contraction path. Generated C++ contains
no runtime string parser, dynamic shape dispatch, or path search. Contiguous inner
reductions use fixed-size bulk loads and FMA loops; generic loops cover broadcasting,
diagonals, permutations, and arbitrary contraction dimensions.

The native API intentionally does not yet implement NumPy's integer-sublist calling
form, explicit precomputed path lists, `out=`, `dtype=`, `order=`, `casting=`, or
writeable-view semantics. Native einsum results are currently accumulated and stored
as `float64`.

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
fixed-size tensor scratch only when a contraction path actually requires them.

## `riskmodel.roll_rets`

The actual expression object imported from `flows.riskmodel` compiles directly:

```python
from flows.riskmodel import roll_rets
from trading_dsl_engine.cpp_stream import compile_npy_formula

runtime = compile_npy_formula(
    roll_rets,
    paths,
    n_instruments=9,
    default_group_capacity=4096,
)
runtime.run_npy_files(paths, out_path="roll_rets.bin")
```

The generated plan contains 50 scalar/vector scratch slots and one six-wide matrix
scratch slot. RBF and future-RBF basis values remain lazy. A native end-to-end test
compares this exact expression against JAX-flat with finite-output checks and
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
identical and every sampled output was finite. The updated `roll_rets` median is
approximately 5.78 seconds for 5,000,000 rows and is above the prior 0.855752 M
rows/s baseline.

Reproduce with:

```bash
python scripts/benchmark_cpp_stream_einsum.py
python scripts/benchmark_cpp_stream_roll_rets.py
```

Both scripts default to 5M x 9, one warmup, and ten measured runs. Input generation,
source generation, native compilation, mmap setup, and warmup are excluded from the
reported execution time. Full run distributions and checksums are in
`PERFORMANCE.md`.

## Compilation cache

The cache key includes generated source, packaged headers, compiler identity,
compile/link flags, platform/machine, and Python ABI. The default cache is:

```text
~/.cache/trading_dsl_engine/cpp_stream
```

Override it with `TRADING_DSL_ENGINE_CPP_STREAM_CACHE`.
