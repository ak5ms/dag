# cpp_stream Eigen/NNQP refactor benchmark

Same GitHub-hosted runner; 1,000,000 rows x 9 instruments; one warmup and seven measured executions.

| Case | Original backend | Final backend | Ratio |
| --- | ---: | ---: | ---: |
| `stateful_cat` | 7.856398 M rows/s | 7.801014 M rows/s | 0.993x |
| `stateless_beta` | 10.557726 M rows/s | 10.485059 M rows/s | 0.993x |
| `grouped_one_stateful` | 7.718430 M rows/s | 7.667669 M rows/s | 0.993x |
| `grouped_stateful` | 3.155419 M rows/s | 3.155425 M rows/s | 1.000x |
| `stateful_nonnegative` | 6.311343 M rows/s | 6.278696 M rows/s | 0.995x |
| `stateless_nonnegative_beta` | 3.474131 M rows/s | 5.736041 M rows/s | 1.651x |
| `roll_rets` | 1.161185 M rows/s | 1.163170 M rows/s | 1.002x |

## Design decision

A full fixed-size Eigen replacement was tested and rejected: it materially reduced throughput for the common K=3 unconstrained and warm-started stateful paths. The final implementation therefore preserves the allocation-free fixed-array Cholesky, pivoted Gaussian, Jacobi pseudoinverse, and stateful coordinate-descent kernels. Fixed-size Eigen is used by the stateless active-set NNQP path, where it produced the measured 65.1% gain. Eigen is compiled with `EIGEN_DONT_PARALLELIZE` so cpp_stream remains the sole owner of outer worker parallelism. No `Eigen::Dynamic` or Eigen Tensor object is used in `on_data`.

## Repeated-field Cat / source-format audit

```text
format=npy median=11.367258 M rows/s checksum=55216.0230234
format=raw median=11.722402 M rows/s checksum=55216.0230234
raw_to_npy_ratio=1.031243
```

The audit formula is `cat(x + 1, x + 2, x + 3)`. It asserts one generated outer row loop, one row-pointer binding for the source, and identical `.npy`/raw checksums.

A separate `strace -f -c` run covered `openat`, `mmap`, `munmap`, `read`, `pread64`, `brk`, and `mremap`. It is supporting evidence for mapping/syscall behavior only: the totals include Python startup, imports, and native compilation and do not prove the number of userspace loads or heap allocations. The stronger checks are the generated-loop assertions, fixed-size storage, and absence of dynamic Eigen types in the row path.
