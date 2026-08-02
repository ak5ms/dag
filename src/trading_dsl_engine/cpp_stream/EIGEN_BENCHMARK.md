# cpp_stream Eigen/NNQP refactor benchmark

Same GitHub-hosted runner; 1,000,000 rows x 9 instruments; one warmup and five measured executions.

| Case | Before | After | Ratio |
| --- | ---: | ---: | ---: |
| `stateful_cat` | 5.873543 M rows/s | 6.130991 M rows/s | 1.044x |
| `stateless_beta` | 9.101382 M rows/s | 8.708761 M rows/s | 0.957x |
| `grouped_one_stateful` | 6.111490 M rows/s | 5.900436 M rows/s | 0.965x |
| `grouped_stateful` | 2.765531 M rows/s | 2.715668 M rows/s | 0.982x |
| `stateful_nonnegative` | 4.746722 M rows/s | 4.741463 M rows/s | 0.999x |
| `stateless_nonnegative_beta` | 2.677686 M rows/s | 4.642962 M rows/s | 1.734x |
| `roll_rets` | 0.888963 M rows/s | 0.890017 M rows/s | 1.001x |

## Repeated-field Cat / source-format audit

```text
format=npy median=6.140916 M rows/s checksum=55216.0230234
format=raw median=6.275379 M rows/s checksum=55216.0230234
raw_to_npy_ratio=1.021896
```

The audit formula is `cat(x + 1, x + 2, x + 3)`. It asserts one generated outer row loop, one row-pointer binding for the source, and identical `.npy`/raw checksums. `strace` syscall totals are retained in the workflow log.
