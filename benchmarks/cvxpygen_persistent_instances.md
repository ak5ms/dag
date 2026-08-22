# Persistent CVXPYgen/Clarabel hot-path benchmark

This benchmark exercises the production generated C++ adapter at six MPO sizes,
through 150 assets × 8 horizons. Each process performs one warm-up, ten measured
iterations, and three repetitions. Table entries are the median of the three
per-process medians; the companion JSON contains every measured iteration and
checksum.

Environment: 9-vCPU affinity on an AMD EPYC 9V74 host, Linux 6.18.35, GCC 13.3,
CVXPYgen 1.0.0, and the allocation-free Clarabel 0.11.1 build. Native code uses
`-O3 -march=native -mtune=native`.

## Serial hot-path regimes

- **All changed:** expected returns, spreads, current weights, risk radii, and
  the dense risk factor change; canonical `A`, `q`, and `b` update.
- **Objective only:** only expected returns change; unchanged parameter buffers
  are scanned but not copied and their canonical blocks remain clean.
- **Unchanged:** every parameter is bitwise identical; the prior solution and
  already-retrieved projections are reused without entering Clarabel.

| assets × horizons | all changed | objective only | unchanged | unchanged speedup | projected bytes |
|---:|---:|---:|---:|---:|---:|
| 9 × 8 | 1.294 ms | 1.175 ms | 0.000091 ms | 14,298× | 72 B |
| 24 × 8 | 7.584 ms | 7.165 ms | 0.000481 ms | 15,767× | 192 B |
| 50 × 8 | 39.640 ms | 38.543 ms | 0.001342 ms | 29,538× | 400 B |
| 75 × 8 | 112.787 ms | 116.465 ms | 0.003726 ms | 30,270× | 600 B |
| 100 × 8 | 249.461 ms | 247.965 ms | 0.006330 ms | 39,412× | 800 B |
| 150 × 8 | 792.384 ms | 803.335 ms | 0.013501 ms | 58,693× | 1,200 B |

Changing even one coefficient remains solver-bound, which is why objective-only
timings are close to all-changing timings. The high-value optimization is the
exact unchanged case: it reduces work to fixed-buffer comparisons and direct
projection from the cached solution.

## Allocation and memory checks

GNU linker wrappers counted `malloc`, `calloc`, `realloc`, and `aligned_alloc`
after warm-up. Every serial scenario at every size recorded **zero calls**. The
allocator audit includes parameter comparisons/copies, generated canonical maps,
Clarabel update/solve when required, lazy result retrieval, and projection.

Median resident growth during the ten all-changing iterations was 192 KiB at
every size; objective-only and unchanged regimes recorded no additional resident
growth. The output is only first-horizon weights, so projected bytes scale as
`assets × sizeof(double)`.

## Independent-problem parallel throughput

These are all-changing problems. Each worker owns a separate generated program
and persistent solver; only immutable generated maps are shared. `ms/problem`
is wall time divided by total problems across workers.

| assets × horizons | workers | ms/problem | problems/s |
|---:|---:|---:|---:|
| 9 × 8 | 2 | 0.718 | 1,393.2 |
| 9 × 8 | 4 | 0.313 | 3,195.4 |
| 9 × 8 | 8 | 0.214 | 4,675.1 |
| 24 × 8 | 2 | 4.196 | 238.3 |
| 24 × 8 | 4 | 2.171 | 460.5 |
| 24 × 8 | 8 | 1.274 | 784.9 |
| 50 × 8 | 2 | 20.216 | 49.5 |
| 50 × 8 | 4 | 10.493 | 95.3 |
| 50 × 8 | 8 | 7.154 | 139.8 |
| 75 × 8 | 2 | 78.623 | 12.7 |
| 75 × 8 | 4 | 46.958 | 21.3 |
| 75 × 8 | 8 | 22.663 | 44.1 |
| 100 × 8 | 2 | 128.521 | 7.8 |
| 100 × 8 | 4 | 62.916 | 15.9 |
| 100 × 8 | 8 | 33.877 | 29.5 |
| 150 × 8 | 2 | 408.015 | 2.5 |
| 150 × 8 | 4 | 210.105 | 4.8 |
| 150 × 8 | 8 | 115.182 | 8.7 |

## Compile-time scaling

CVXPYgen 1.0 previously densified an affine parameter map merely to apply its
sign, and the 150 × 8 case attempted a roughly 34.9 GiB temporary. The adapter
now applies that scalar/row-wise sign directly to the sparse map. The 150 × 8
program successfully generated, compiled, and executed.

A fresh 150 × 8 generation after both sparse fixes took 36.844 s with a 4.46
GiB process peak RSS. The remaining peak comes from CVXPYgen's retained
canonical/code-writer structures, rather than a dense 34.9 GiB map.

Representative uncached measurements from the clean sweep:

| assets × horizons | CVXPYgen | C++ compile/link | generated header | binary |
|---:|---:|---:|---:|---:|
| 9 × 8 | 0.099 s | 1.277 s | 0.26 MiB | 2.51 MiB |
| 24 × 8 | 0.249 s | 1.313 s | 1.00 MiB | 2.74 MiB |
| 50 × 8 | 1.669 s | 1.986 s | 3.43 MiB | 3.55 MiB |
| 75 × 8 | 4.785 s | 2.695 s | 7.13 MiB | 4.80 MiB |
| 100 × 8 | 11.442 s | 3.710 s | 12.16 MiB | 6.52 MiB |
| 150 × 8 | 36.844 s | 7.562 s | 26.28 MiB | 11.36 MiB |

## Reproduction and raw data

Run `scripts/benchmark_cvxpygen_persistent_instances.py` with the Clarabel
include and archive paths set. The script checkpoints after every size, verifies
stable finite checksums, fails on any warm-path allocation, and retains all ten
raw samples for every process repetition.

Raw samples and checksums: [`cvxpygen_persistent_instances.json`](cvxpygen_persistent_instances.json).
