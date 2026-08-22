# Persistent CVXPYgen/Clarabel instance benchmark

Every problem changes the covariance factor and therefore canonical `A`,
as well as `q` and `b`. Timings cover bulk parameter copies, generated
CVXPYgen canonicalization, Clarabel updates/solve, and primal projection.
The timed path is entirely native C++/C.

## Serial

| assets × horizons | median | mean | RSS growth |
|---:|---:|---:|---:|
| 9 × 8 | 1.251 ms | 1.280 ms | 0.125 MB |
| 24 × 8 | 7.540 ms | 7.464 ms | 0.125 MB |
| 50 × 8 | 39.432 ms | 39.100 ms | 0.125 MB |

## Independent-problem parallel throughput

`ms/problem` is wall time divided by total problems across workers.
Each worker owns a separate generated CVXPYgen object and persistent
Clarabel solver; immutable generated maps are shared.

| assets × horizons | workers | ms/problem | problems/s | speedup |
|---:|---:|---:|---:|---:|
| 9 × 8 | 2 | 0.622 ms | 1608.5 | 2.01× |
| 9 × 8 | 4 | 0.353 ms | 2834.7 | 3.55× |
| 24 × 8 | 2 | 3.780 ms | 264.5 | 1.99× |
| 24 × 8 | 4 | 1.940 ms | 515.5 | 3.89× |
| 50 × 8 | 2 | 21.665 ms | 46.2 | 1.82× |
| 50 × 8 | 4 | 10.486 ms | 95.4 | 3.76× |

## Compile-time cost

| assets × horizons | CVXPYgen | C++ compile/link | header | binary |
|---:|---:|---:|---:|---:|
| 9 × 8 | 0.097 s | 1.277 s | 0.23 MB | 2.51 MB |
| 24 × 8 | 0.414 s | 1.345 s | 0.94 MB | 2.75 MB |
| 50 × 8 | 3.310 s | 2.302 s | 3.30 MB | 3.57 MB |

## Build and ownership properties

- CVXPYgen owns DPP validation, parameter maps, canonical cone data,
  and primal/dual result mapping.
- Every generated C++ object owns mutable parameter/canonical/result
  buffers and exactly one persistent Clarabel solver.
- Read-only sparse maps and cone descriptors are `inline static` and
  shared across instances.
- The destructor calls `clarabel_DefaultSolver_free`.
- No Python or CVXPY work occurs in the measured solve path.
