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
result = runtime.run_npy_files(
    {"close": "/data/close.npy", "open": "/data/open.npy"},
    out_path="/data/alpha.bin",
)
```

`.npy` inputs are mapped without copying. Supported C-order shapes are `(rows,)`,
`(rows, 1)`, and `(rows, n_instruments)`. Supported dtypes are `float32`, `float64`,
`int32`, `int64`, `uint32`, and `uint64`.

## Execution model

Every operator has one native implementation and receives its execution scope as
the final template argument:

```cpp
DirectExecution<N>
GroupedExecution<N, Capacity, PartitionCount>
```

There are no `GroupedFooNode` or formula-specific fast-path classes. `groupby.hpp`
contains only key resolution, grouped-context construction, and inner-plan
invocation. Stateful and cross-sectional nodes obtain storage/group identity through
the generic execution interface.

Mapped values retain native types through typed scalar/vector scratch. Matrix-valued
intermediates use separate fixed-width matrix scratch. No operator allocates from the
heap during `on_data`.

## Matrix and feature values

`cat(...)`, RBF bases, coefficient matrices, and einsum use compile-time feature
widths. Lazy basis sources and nested Cat expressions are flattened into
`FeatureList<Sources...>` so consumers read original inputs directly.

For example, these produce the same generated Ridge source:

```python
Ridge(cat(x1, x2, x3), y=y, hl=64, lambda_=0.1)
Ridge(x1, x2, x3, y=y, hl=64, lambda_=0.1)
```

A standalone Cat root is written as logical shape `(rows, N, K)`. The
`InstrumentBasisMean` beta used by `roll_rets` is the only materialized six-wide
matrix in that plan; RBF and future-RBF basis values remain lazy.

## Supported native operators

The backend now supports every operator used by `flows.riskmodel.roll_rets`:

```text
arithmetic: add, sub, mul, div, mod, pow, floor
comparisons: eq, ne, lt, gt, le, ge
logic/select: and, or, xor, where, fillna
history: cumsum, ffill, shift, ewm
cross-sectional: xs_rank
matrix/features: cat, rbf_basis, future_rbf_basis_sum,
                 einsum("nf,nf->n")
models: Ridge, InstrumentBasisMean, get_beta, get_preds
grouping: univ plus bounded-dense or exact-hash dynamic tuple keys
named stateless calls used by POV/roll_rets
```

Named stateless expressions are backend-neutral. JAX backends retain their Python
callable; native lowering selects an explicitly registered C++ policy by stable name.

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
scratch slot. It includes generic RBF sources, session-grouped cumulative state,
`InstrumentBasisMean`, `einsum`, forward fill, shift, boolean selection, and the
remaining arithmetic graph.

A native end-to-end test compares this exact expression against JAX-flat on identical
input data, including missing/tradability transitions. The comparison is non-vacuous
and passes at `rtol=2e-9`, `atol=2e-9` with equal-NaN semantics.

## 5M x 9 `roll_rets` benchmark

GitHub-hosted Ubuntu runner, float64 mmap inputs, GCC C++20 with
`-O3 -march=native -mtune=native -flto`, one warmup and ten measured executions:

```text
median  0.855752 M rows/s
mean    0.855213 M rows/s
best    0.856243 M rows/s
```

This is approximately 5.84 seconds for 5,000,000 rows, or 7.70 million instrument
observations per second. All ten runs were within roughly 0.4% of the median. The
sampled output tail was 100% finite and had checksum `-0.790555667227`.

Reproduce with:

```bash
python scripts/benchmark_cpp_stream_roll_rets.py
```

The script defaults to 5M x 9, one warmup, and ten measured runs. Input generation,
source generation, native compilation, mmap setup, and warmup are excluded from the
reported execution time.

## Other measured baselines

On the same class of hosted runner:

| Workload | Median throughput |
| --- | ---: |
| Typed row-scalar minute groupby | 20-22 M rows/s |
| `cat(x1,x2,x3)` root | 11.307 M rows/s |
| Stateful K=3 Ridge predictions | 6.377 M rows/s |
| Stateless K=3 Ridge beta | 9.169 M rows/s |
| One-group grouped stateful Ridge | 6.176 M rows/s |
| Three-group grouped stateful Ridge | 2.787 M rows/s |

Absolute throughput is host-dependent. Same-host checksums and run distributions are
recorded in `PERFORMANCE.md`.

## Compilation cache

The cache key includes generated source, packaged headers, compiler identity,
compile/link flags, platform/machine, and Python ABI. The default cache is:

```text
~/.cache/trading_dsl_engine/cpp_stream
```

Override it with `TRADING_DSL_ENGINE_CPP_STREAM_CACHE`.
