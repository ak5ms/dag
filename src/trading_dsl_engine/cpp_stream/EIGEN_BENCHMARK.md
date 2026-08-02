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

## Repeated-field Cat / source-format audit

```text
format=npy median=11.367258 M rows/s checksum=55216.0230234
format=raw median=11.722402 M rows/s checksum=55216.0230234
raw_to_npy_ratio=1.031243
```

The audit formula is `cat(x + 1, x + 2, x + 3)`. It asserts one generated outer row loop, one row-pointer binding for the source, and identical `.npy`/raw checksums.
