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
the final integrated semantic path measured 14.36 M rows/s in a pinned,
file-I/O-excluded kernel test versus 10.43 M rows/s for the earlier standalone
implementation on the same host.

## No operator-specific groupby fast classes

`FastGroupedEwmNode` and `FastGroupedXsRankNode` were removed. Grouped and
ungrouped execution now instantiate the same implementation through compile-time
policies:

- EWM uses `DirectStateIndex` or `GroupedStateIndex`.
- Rank uses `GlobalRankGroup` or `ContextRankGroup`.

The common recursive EWM policy and the all-finite small-width rank policy live once
inside the shared templates. `if constexpr` removes paths that do not apply to the
chosen policy. `groupby.hpp` owns key resolution and grouped execution plumbing,
not EWM or rank implementations.

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

Controlled workload:

- 5,000,000 rows x 9 instruments
- one pinned CPU
- warmed mmap input pages
- shared, pre-sized output mapping
- GCC C++20, `-O3 -march=native -mtune=native -flto`
- Intel Xeon E5-2673 v4

| Implementation | Median throughput |
| --- | ---: |
| Legacy generic grouped EWM with weight/count traffic | 3.68 M rows/s |
| Shared policy-based grouped EWM, no `FastGrouped*` class | **4.09 M rows/s** |

Ten measured policy-based runs were:

```text
4.136, 4.135, 4.125, 4.121, 4.113,
4.064, 4.022, 3.916, 3.911, 3.849 M rows/s
```

The checksum matched the legacy implementation.

## What still limits this formula

A controlled architecture experiment produced:

| Key representation | Throughput |
| --- | ---: |
| Current: vector `_ev_ts` derivation + hash lookup | 3.96 M rows/s |
| Vector derivation + direct dense slot | 4.81 M rows/s |
| Row-scalar derivation + hash lookup | 8.45 M rows/s |
| Row-scalar derivation + direct dense slot | 15.11 M rows/s |

These are not operator fast paths. They identify missing semantic information in the
IR and data model:

1. `_ev_ts` is represented as a normal instrument vector, so the compiler computes
   the identical calendar expression nine times and reads nine timestamp values.
   The IR has no row-scalar/broadcast-invariant value kind.
2. Macro expansion turns `minute(_ev_ts)` into generic `mod/floor/div` nodes and
   drops the known integer domain `0..59`. The groupby lowering therefore uses a
   hash table instead of direct dense indexing.
3. Group resolution materializes one slot per lane. The downstream plan cannot
   express or exploit the fact that all lanes often share the same dynamic key.

The next architecture-level optimizations should therefore be backend-neutral domain
propagation and a row-scalar value type, followed by a generic uniform-slot/run
representation for grouped state access. They should not be implemented as
`FastGroupedFooNode` classes.

## Reproducible repository benchmark

The requested minute formula is now the default case:

```bash
python scripts/benchmark_cpp_stream.py
```

The previous rank benchmark remains available with:

```bash
CPP_STREAM_BENCH_CASE=rank python scripts/benchmark_cpp_stream.py
```

The script defaults to one warmup and ten measured 5M x 9 runs. Do not commit a
universal regression floor derived from one CPU or filesystem. Record host,
compiler, CPU affinity, page-cache state, prefetch distance, and writeback settings
with every reported result.
