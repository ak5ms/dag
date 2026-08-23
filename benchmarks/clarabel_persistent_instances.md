# Persistent direct Clarabel benchmark

This benchmark exercises the production generated C++ adapter at six MPO sizes,
through 150 assets × 8 horizons. Each process performs three warm-ups, ten
measured iterations, and three repetitions. Table entries are the median of the
three per-process medians; the companion JSON retains every sample and checksum.

Environment: 9-vCPU affinity on an Intel Xeon Platinum 8573C host, Linux
6.18.35, GCC 13.3, CVXPY 1.9.2, and the allocation-free Clarabel 0.11.1 build.
CVXPYgen is neither imported nor used and is no longer a project dependency.
Native code uses
`-O3 -march=native -mtune=native`.

## Selected generation approach

Four generic generation routes and one formula-specific prototype were measured
on the 150 × 8 MPO. Full-map routes ask CVXPY for the complete affine
parameter-to-canonical-data tensor. The selected route instead canonicalizes
bounded 512-scalar parameter shards, merges their sparse nonzeros, and performs
Clarabel cone formatting as a signed coordinate permutation.

| route | fresh generation | peak RSS | production choice |
|---|---:|---:|---|
| patched CVXPYgen 1.0 | 36.443 s | 4,675,040 KiB | rejected |
| CVXPY full map, CPP backend | 20.03 s | 4,736,504 KiB | rejected |
| CVXPY full map, SCIPY backend | 30.31 s | 4,415,284 KiB | rejected |
| CVXPY full map, COO backend | 12.68 s | 4,384,532 KiB | rejected |
| hand-specialized MPO sparsity | 0.057 s | 145,920 KiB | rejected: formula-specific |
| **bounded-shard direct Clarabel** | **1.001 s** | **405,068 KiB** | **selected** |

The rejected routes were measured before installation of the project's JAX
runtime. The selected row conservatively includes it: CVXPY eagerly imports JAX
while discovering optional solvers. Five final guarded runs had a 1.001-second
median, a 364,820-KiB loaded baseline, a 405,068-KiB (395.6 MiB) absolute process
peak, and only 40,132 KiB (39.2 MiB) of generation growth. They emitted a
5,449,354-byte (5.20 MiB) header and made no rejected dense allocation attempt.
The conservative final comparison still reduces generation time by 97.3%,
absolute peak RSS by 91.3%, and header size by 80.4% relative to patched
CVXPYgen.

## The former 34.895 GiB temporary

CVXPYgen 1.0 called an eager sparse `toarray()` while applying signs to an
affine map with shape `186,900 × 25,059`. At float64 width, its 4,683,527,100
entries request 37,468,216,800 bytes, or 34.895 GiB. This was a real dense
allocation request, not a lazy sparse view; if admitted, `toarray()` writes the
output and materializes its pages. A prior sparse-only patch removed that call,
but CVXPYgen and full-map CVXPY still retained multi-GiB intermediate tensors.

The selected audit wraps `numpy.zeros` and every SciPy sparse `toarray()` entry
point and rejects any individual request at or above 512 MiB. It completed with
zero rejected attempts and a 270.3 MiB process peak, confirming that the former
temporary is neither requested nor materialized by the production route.

## Serial hot-path regimes

- **All changed:** expected returns, spreads, current weights, risk radii, and
  the dense risk factor change; canonical `A`, `q`, and `b` update.
- **Objective only:** only expected returns change; unchanged parameter buffers
  are scanned but not copied and their canonical blocks remain clean.
- **Unchanged:** every parameter is bitwise identical; the prior solution and
  already-retrieved projections are reused without entering Clarabel.

| assets × horizons | all changed | objective only | unchanged | allocations | output |
|---:|---:|---:|---:|---:|---:|
| 9 × 8 | 1.193 ms | 1.148 ms | 0.000115 ms | 0 | 72 B |
| 24 × 8 | 6.818 ms | 6.823 ms | 0.000487 ms | 0 | 192 B |
| 50 × 8 | 41.242 ms | 39.777 ms | 0.002003 ms | 0 | 400 B |
| 75 × 8 | 106.585 ms | 115.068 ms | 0.003253 ms | 0 | 600 B |
| 100 × 8 | 236.803 ms | 228.996 ms | 0.006515 ms | 0 | 800 B |
| 150 × 8 | 759.478 ms | 796.221 ms | 0.011711 ms | 0 | 1,200 B |

Changing even one objective coefficient remains solver-bound, which is why
objective-only timings are close to all-changing timings. The exact unchanged
case reduces work to fixed-buffer comparisons and direct projection from the
cached solution.

## Allocation and memory layout checks

GNU linker wrappers count `malloc`, `calloc`, `realloc`, and `aligned_alloc`
after warm-up. Every serial scenario at every size recorded **zero calls**. The
audit includes parameter comparisons and copies, compact-map canonicalization,
Clarabel update/solve when required, lazy result retrieval, and projection.

Mutable parameter, canonical, and result buffers are instance-owned contiguous
arrays with cache-line-aligned canonical workspaces. Immutable map values,
32-bit parameter columns/row pointers, cone descriptors, and CSC structure are
static structure-of-arrays shared by all instances. The Clarabel ABI's native
`uintptr_t` width is retained only for CSC indices. Each worker owns one solver;
the solver allocates during construction, then the warmed solve path allocates
nothing.

Median resident growth during ten all-changing iterations was 192 KiB at every
size; objective-only and unchanged regimes recorded no additional resident
growth. The dedicated feedback audit also binds one parameter through
`previous_solution`, retrieves the carried primal after each solve, and runs 100
changed warmed solves with zero wrapped allocator calls.

## Independent-problem parallel throughput

These are all-changing problems. `ms/problem` is wall time divided by total
problems across workers. Immutable generated maps are shared; mutable state is
worker-local.

| assets × horizons | workers | ms/problem | problems/s | serial speedup |
|---:|---:|---:|---:|---:|
| 9 × 8 | 2 | 0.896 | 1,116.2 | 1.33× |
| 9 × 8 | 4 | 0.922 | 1,084.3 | 1.29× |
| 9 × 8 | 8 | 0.533 | 1,875.1 | 2.24× |
| 24 × 8 | 2 | 5.423 | 184.4 | 1.26× |
| 24 × 8 | 4 | 6.135 | 163.0 | 1.11× |
| 24 × 8 | 8 | 3.864 | 258.8 | 1.76× |
| 50 × 8 | 2 | 29.913 | 33.4 | 1.38× |
| 50 × 8 | 4 | 19.162 | 52.2 | 2.15× |
| 50 × 8 | 8 | 12.264 | 81.5 | 3.36× |
| 75 × 8 | 2 | 129.565 | 7.7 | 0.82× |
| 75 × 8 | 4 | 90.017 | 11.1 | 1.18× |
| 75 × 8 | 8 | 35.836 | 27.9 | 2.97× |
| 100 × 8 | 2 | 190.094 | 5.3 | 1.25× |
| 100 × 8 | 4 | 100.457 | 10.0 | 2.36× |
| 100 × 8 | 8 | 67.378 | 14.8 | 3.51× |
| 150 × 8 | 2 | 620.582 | 1.6 | 1.22× |
| 150 × 8 | 4 | 329.348 | 3.0 | 2.31× |
| 150 × 8 | 8 | 206.870 | 4.8 | 3.67× |

## Compile-time scaling

| assets × horizons | code generation | C++ compile/link | header | binary |
|---:|---:|---:|---:|---:|
| 9 × 8 | 0.036 s | 1.235 s | 0.04 MiB | 2.48 MiB |
| 24 × 8 | 0.048 s | 1.320 s | 0.16 MiB | 2.59 MiB |
| 50 × 8 | 0.146 s | 1.342 s | 0.60 MiB | 2.98 MiB |
| 75 × 8 | 0.249 s | 1.342 s | 1.31 MiB | 3.60 MiB |
| 100 × 8 | 0.431 s | 1.543 s | 2.30 MiB | 4.44 MiB |
| 150 × 8 | 1.035 s | 2.331 s | 5.20 MiB | 6.81 MiB |

## Reproduction and raw data

Run `scripts/benchmark_clarabel_persistent_instances.py` with the Clarabel
include and archive paths set. The legacy filename is retained for compatibility;
the benchmark invokes only direct Clarabel generation. It checkpoints after
each size, verifies stable finite checksums, fails on any warm-path allocation,
and retains all ten samples for every repetition.

Run `scripts/audit_clarabel_sparse_generation.py` for the fresh guarded 150 × 8
generation. `CLARABEL_AUDIT_ASSETS`, `CLARABEL_AUDIT_HORIZONS`,
`CLARABEL_AUDIT_DENSE_LIMIT_BYTES`, `CLARABEL_AUDIT_PARAMETER_SHARD_SIZE`, and
`CLARABEL_AUDIT_OUTPUT_DIR` override its defaults.

Raw samples and checksums:
[`clarabel_persistent_instances.json`](clarabel_persistent_instances.json).
Fresh guarded generation samples:
[`clarabel_generation_audit.json`](clarabel_generation_audit.json).
Final fresh generation/compile sweep:
[`clarabel_codegen_sweep.json`](clarabel_codegen_sweep.json).
