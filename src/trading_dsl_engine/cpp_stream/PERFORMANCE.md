# cpp_stream performance baseline

This document records the July 31, 2026 comparisons used while integrating and
refactoring `trading_dsl_engine.cpp_stream`.

## Standalone comparison

The first standalone prototype used simpler semantics than the repository backend:

- EWM assumed an all-finite stream and synchronized initialization.
- `xs_rank` emitted ordinal `[0, 1]` ranks and broke ties by lane index.

The integrated backend preserves independent per-lane initialization, NaN carry,
`min_periods`, `ignore_na`, `adjust`, finite masking, and upper-tie normal scores.
The earlier ~21 M rows/s headline was also measured on a different AMD EPYC host,
so it is not an apples-to-apples threshold for this branch.

For `xs_rank(ewm(close / open, 21))` on the Intel Xeon E5-2673 v4 comparison host,
the integrated semantic path measured 14.36 M rows/s in a pinned,
file-I/O-excluded kernel test versus 10.43 M rows/s for the earlier standalone
implementation on the same host.

## Operator-agnostic group execution

There are no operator-specific grouped classes. In particular, the backend does
not define any of the following:

```text
GroupedCumsumNode
GroupedEwmNode
GroupedXsRankNode
FastGroupedFooNode
```

Every generated node receives one final execution-scope template argument:

```cpp
DirectExecution<N>
GroupedExecution<N, Capacity>
```

The same `CumsumNode`, `EwmNode`, `XsRankNode`, `BinaryNode`, and `UnaryNode`
types are emitted inside and outside groupby. The execution scope supplies generic
state indexing and cross-sectional group identity. `groupby.hpp` contains no
operator implementation.

### Zero-cost refactor check

The immediately prior implementation used separate per-operator grouped classes.
A controlled benchmark compared that implementation with the execution-scope
version using the same state layout, recurrence, generated calendar stages, and
output semantics.

Workload:

- 5,000,000 rows x 9 instruments
- `_ev_ts -> time-of-day -> minute` represented as generated vector stages
- static column groups `[0]`, `[1,2]`, `[3..8]`
- grouped `ewm(cumsum(self_), 3)`
- one pinned CPU
- GCC C++20, `-O3 -march=native -mtune=native -flto`
- Intel Xeon E5-2673 v4

| Implementation | Median throughput |
| --- | ---: |
| Prior separate grouped operator classes | 4.647 M rows/s |
| **Single node family + `GroupedExecution`** | **4.648 M rows/s** |

Checksums were identical. The difference is below measurement noise; passing the
execution scope as a compile-time type did not regress the hot loop.

A 2,000,000-row, twelve-run alternating comparison gave the same conclusion:
4.611 M rows/s versus 4.602 M rows/s.

## Timestamp-derived minute benchmark

Requested formula:

```python
groupby(
    (univ([0], [1, 2], list(range(3, 9))), var("minute")),
    var("close"),
    ewm(cumsum(self_), 3),
)
```

`var("minute")` is expanded by the neutral frontend to the existing DSL
`minute(_ev_ts)` derivation. The DSL definition is **minute within the hour**
(`0..59`), not minute-of-day (`0..1439`). `_ev_ts` is a float64 microseconds-since-
epoch input.

The operator-agnostic execution-scope implementation preserves the previously
measured optimized grouped result while removing grouped operator boilerplate.

## What still limits this formula

A controlled architecture experiment using the same stage-shaped calendar
computation produced:

| Key/value representation | Throughput |
| --- | ---: |
| Current: vector `_ev_ts` stages + hash lookup | 4.85 M rows/s |
| Vector `_ev_ts` stages + direct dense slot | 6.08 M rows/s |
| Row-scalar calendar derivation + hash lookup | 7.96 M rows/s |
| Row-scalar calendar derivation + direct dense slot | 15.97 M rows/s |

These are not operator fast paths. They identify missing semantic information in the
IR and data model:

1. `_ev_ts` is represented as a normal instrument vector, so the compiler computes
   the identical calendar expression nine times and reads nine timestamp values.
   The IR has no row-scalar/broadcast-invariant value kind.
2. Macro expansion turns `minute(_ev_ts)` into generic `mod/floor/div` nodes and
   drops the known integer domain `0..59`. Groupby lowering therefore uses a hash
   table instead of direct dense indexing.
3. Group resolution materializes one slot per lane. The downstream plan cannot
   represent or exploit the fact that all lanes often share the same dynamic key.
4. The semantic IR describes values and operators, but not value invariance,
   categorical domains, or grouped-state access runs. A C++ compiler cannot invent
   those semantic facts from arbitrary runtime arrays.

The next architecture-level optimizations should therefore be backend-neutral domain
propagation and a row-scalar value type, followed by a generic uniform-slot/run
representation for grouped state access. They should not be implemented as
operator-specific grouped classes.

## Reproducible repository benchmark

The requested minute formula is the default case:

```bash
python scripts/benchmark_cpp_stream.py
```

The rank benchmark remains available with:

```bash
CPP_STREAM_BENCH_CASE=rank python scripts/benchmark_cpp_stream.py
```

The script defaults to one warmup and ten measured 5M x 9 runs. Do not commit a
universal regression floor derived from one CPU or filesystem. Record host,
compiler, CPU affinity, page-cache state, prefetch distance, and writeback settings
with every reported result.
