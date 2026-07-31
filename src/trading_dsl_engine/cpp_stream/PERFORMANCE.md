# cpp_stream performance baseline

This document records the July 31, 2026 comparison between the integrated
`trading_dsl_engine.cpp_stream` backend and the earlier standalone prototype.

## Why the earlier headline was not directly comparable

The first standalone prototype used materially simpler semantics:

- EWM assumed an all-finite stream and one synchronized initialization state.
- `xs_rank` emitted ordinal `[0, 1]` ranks and broke ties by lane index.

The integrated backend preserves repository semantics instead:

- EWM is initialized independently per lane and supports NaN carry,
  `min_periods`, `ignore_na`, and `adjust`.
- `xs_rank` masks non-finite values and emits upper-tie normal scores.

The earlier ~21 M rows/s headline was also recorded on a different AMD EPYC host.
It must not be treated as an apples-to-apples regression threshold for this branch.

## Controlled kernel comparison

Workload:

- 5,000,000 rows x 9 instruments
- `xs_rank(ewm(close / open, 21))`
- one pinned CPU
- warmed mmap input pages
- pre-touched anonymous output memory
- file creation, output extent allocation, and storage writeback excluded
- GCC C++20, `-O3 -march=native -mtune=native -flto`
- host used for this comparison: Intel Xeon E5-2673 v4

| Implementation | Median throughput |
| --- | ---: |
| Earlier standalone prototype, simpler semantics | 10.43 M rows/s |
| Integrated backend before fast paths, repository semantics | 10.33 M rows/s |
| Integrated backend after fast paths, repository semantics | **14.36 M rows/s** |

The integrated backend was effectively tied with the standalone implementation
before optimization. The final implementation is about 39% faster than the
pre-optimization integrated path and about 38% faster than the standalone
comparison on this host while retaining the repository semantics.

## Fixes retained in the production path

- `EwmNode` specializes the common `min_periods=0`, `ignore_na=True`,
  `adjust=False` policy. Once every lane is initialized and the row is finite, it
  executes only the recursive FMA update. A lane-aware fallback preserves NaN
  and initialization behavior.
- `XsRankNode` and grouped rank use an all-finite exact rank-count path for
  `N <= 16`. This avoids a repeated finite-mask test inside every `N x N`
  comparison. Wider universes retain the sorting implementation.
- Grouped EWM/rank use the same specialized policies.
- Reused output files are not truncated when their existing size is already
  correct. Every output row is overwritten, so retruncation only reintroduces
  extent/page-allocation noise.

## Full mmap benchmark

On the same constrained virtual host, the full file-backed benchmark was dominated
by dirty-page and backing-store behavior. Depending on page-cache state, the
pre-optimization path measured roughly 1.58-1.74 M rows/s and the optimized path
roughly 1.71-1.98 M rows/s. These absolute values are not suitable for comparing
machines, but the optimized version remained faster.

Use the pinned/controlled kernel benchmark when judging code-generation or operator
regressions. Use the full mmap benchmark when judging I/O and writeback changes.

## Reproducible repository benchmark

```bash
python scripts/benchmark_cpp_stream.py
```

The script defaults to one warmup plus ten measured 5M x 9 runs and reuses the same
output file. An environment-specific regression floor can be enforced with:

```bash
CPP_STREAM_BENCH_MIN_MROWS=12 python scripts/benchmark_cpp_stream.py
```

Do not commit one universal floor derived from a specific CPU or filesystem. Record
host, compiler, CPU affinity, page-cache state, prefetch distance, and writeback
settings with every reported result.
