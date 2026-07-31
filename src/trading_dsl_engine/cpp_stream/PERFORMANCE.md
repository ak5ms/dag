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

## Same-host ablation against the old 21 M rows/s result

The old table used a precomputed key file and, for the 21.43 M rows/s case, direct
dense indexing. The requested integrated formula derives the key from `_ev_ts` as
generic vector arithmetic and then sends the result through the generic hash
resolver. Those are materially different workloads.

The following ablation used one pinned Intel Xeon E5-2673 v4 core, GCC 14 C++20,
`-O3 -march=native -mtune=native -flto`, 2,000,000 x 9 finite rows, warmed input
pages, and an in-memory output. The grouped `cumsum -> EWM(span=3)` state layout and
output semantics were held constant unless the row says otherwise.

| Variant | Median throughput |
| --- | ---: |
| Precomputed dense minute, fused inner state loop | 21.92 M rows/s |
| Precomputed dense minute, normal separated cumsum/EWM stages | **21.71 M rows/s** |
| Precomputed minute, current full hash resolver | 11.72 M rows/s |
| Generated four-stage vector calendar + current hash resolver, fused inner | 5.14 M rows/s |
| **Generated four-stage vector calendar + current hash resolver, normal separated inner** | **4.74 M rows/s** |
| Row-scalar float calendar + dense slot, separated inner | 12.73 M rows/s |
| Row-scalar integer calendar + dense slot, separated inner | **20.32 M rows/s** |

The 4.74 M rows/s row reproduces the apparent integrated result. The 21.71 M rows/s
control shows that the generic execution-scope groupby/state architecture still
reaches the old dense-key throughput when key production and routing are held equal.

Expressed as elapsed time per million rows:

| Step | Time per million rows | Incremental cost |
| --- | ---: | ---: |
| Precomputed dense key | 46.1 ms | baseline |
| Precomputed key + current hash resolver | 85.3 ms | +39.2 ms |
| Vector calendar stages + current hash resolver | 211.0 ms | +125.7 ms |

About 78% of the final row time above the dense-key baseline comes from producing and
resolving the key, not from cumsum, EWM, `GroupedExecution`, or output handling.

### Resolver ablation

For a uniform minute key shared by all nine lanes:

| Resolver behavior | Median throughput |
| --- | ---: |
| Resolve/hash once and broadcast the slot | 15.87 M rows/s |
| Generic row-reuse loop, no per-lane temporal cache | 13.52 M rows/s |
| Current row-reuse plus per-lane temporal cache | 11.60 M rows/s |

The per-lane cache is counterproductive for this key because minute changes on every
row. All nine lane caches miss, are rewritten, and then the same-row reuse logic
still performs the useful sharing. That cache can help persistent per-lane keys, but
it should not be the only generic resolver policy.

### Dense resolver and semantic checks

Additional isolated controls:

| Change | Median throughput |
| --- | ---: |
| Current formula, dense row-shared slot, simplified finite-only semantics | 22.53 M rows/s |
| Current formula, dense row-shared slot, repository NaN-safe semantics | 21.34 M rows/s |
| Dense validation repeated independently for all nine lanes | 17.46 M rows/s |
| Dense validation once plus row-slot broadcast | 21.63 M rows/s |

Repository-safe cumsum/EWM semantics cost about 5% on all-finite data. Revalidating an
identical dense key nine times costs about 19%. Neither accounts for the overall
4.7-versus-21 M rows/s gap.

### Calendar representation ablation

The existing DSL macro expands `minute(_ev_ts)` to floating-point `mod`, division,
`floor`, and another `mod`. The IR currently treats `_ev_ts` as a nine-lane vector,
so those operations are repeated for every lane and materialized as separate vector
stages.

| Calendar/key representation | Median throughput |
| --- | ---: |
| Vector floating calendar expression, current hash resolver | 4.74 M rows/s |
| Row-scalar floating calendar expression, dense slot | 12.73 M rows/s |
| Row-scalar integer timestamp arithmetic, dense slot | 20.32 M rows/s |

The integer row-scalar version is close to the precomputed dense control. This is the
main architectural path to recovering the old performance without adding any
operator-specific groupby implementation.

### What is not causing the regression

- `GroupedExecution` versus separate grouped operator classes: effectively zero.
- Splitting cumsum and EWM into normal graph stages: small, roughly 1-8% depending
  on the run and surrounding key work.
- Output placement after warmup: anonymous output, `/dev/shm`, and a reused normal
  `MAP_SHARED` file measured within noise of one another in the controlled test.
- The requested formula itself: with a precomputed dense key it reaches roughly
  21-22 M rows/s on the comparison host.

## Architectural fixes indicated by the ablation

1. Add a backend-neutral row-scalar or lane-invariant value kind. `_ev_ts` should
   normally be loaded and transformed once per row, then broadcast only when a
   downstream vector operator requires it.
2. Preserve timestamp/integer type information. Lower `minute(_ev_ts)` to integer
   calendar arithmetic rather than four generic floating-point vector operators.
3. Propagate categorical domains through the neutral IR. `minute(_ev_ts)` has known
   domain `0..59`, so groupby can select direct dense state indexing automatically.
4. Represent uniform group slots. A row-scalar key should resolve once and expose a
   broadcast/uniform slot rather than materializing and validating nine independent
   slots.
5. Make resolver caching a compile-time policy selected from key metadata. Temporal
   per-lane caching is useful for persistent lane-specific keys; it is harmful for a
   row-scalar key that changes every row.
6. Add generic producer-consumer fusion for cheap stateless key graphs where useful.
   This is a graph/lowering optimization, not a grouped EWM/cumsum fast path.

## Reproducible repository benchmark

The requested derived-minute formula remains the default case:

```bash
python scripts/benchmark_cpp_stream.py
```

The two principal key ablations are now available directly:

```bash
CPP_STREAM_BENCH_CASE=minute_groupby_precomputed_hash \
python scripts/benchmark_cpp_stream.py

CPP_STREAM_BENCH_CASE=minute_groupby_precomputed_dense \
python scripts/benchmark_cpp_stream.py
```

To match the old benchmark's output placement:

```bash
CPP_STREAM_BENCH_OUTPUT_DIR=/dev/shm \
CPP_STREAM_BENCH_CASE=minute_groupby_precomputed_dense \
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
