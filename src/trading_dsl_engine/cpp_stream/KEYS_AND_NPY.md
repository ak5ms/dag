# Per-key hints and typed `.npy` inputs

## API

Dynamic group keys can carry independent metadata through the backend-neutral
`Key` expression wrapper:

```python
from trading_dsl_engine import Key

groupby(
    (
        univ([0], [1, 2], list(range(3, 9))),
        Key(
            expr=var("minute"),
            num_keys=60,
            offset=0,
            row_scalar=True,
            dtype="int64",
        ),
    ),
    var("close"),
    ewm(cumsum(self_), 3),
)
```

The neutral IR stores one `GroupKeySpec` per dynamic key. `num_keys` describes
consecutive integer categories beginning at `offset`; NaN is one additional valid
category. `row_scalar` asserts lane invariance. `dtype` records the expected semantic
or direct-input type.

A tuple may contain multiple independently described keys. When every key has
`num_keys`, cpp_stream uses generic mixed-radix dense routing with capacity:

```text
product(num_keys_i + 1)
```

The extra digit for each key preserves exact tuple combinations containing NaN. If
any key is unbounded, the complete tuple uses the generic fixed-capacity hash
resolver. If every key is row-scalar, either resolver evaluates one tuple and
broadcasts one group slot.

Direct integer keys are read and hashed in their native input type. In particular,
`int64` and `uint64` values are not converted to `double` before key equality, so
values above `2^53` remain distinct. Dense integer range checks also use exact
128-bit intermediate arithmetic rather than floating-point rounding.

## Typed `.npy` mapping

`compile_npy_formula` inspects all input headers before native compilation:

```python
runtime = compile_npy_formula(
    formula,
    {
        "_ev_ts": "/data/_ev_ts.npy",
        "close": "/data/close.npy",
    },
    n_instruments=9,
)

runtime.run_npy_files(
    {
        "_ev_ts": "/data/_ev_ts.npy",
        "close": "/data/close.npy",
    },
    out_path="/data/result.bin",
)
```

The implementation uses public `numpy.load(..., mmap_mode=..., allow_pickle=False)`
semantics and passes live payload pointers to the generated shared object. It does
not copy the input arrays.

Supported dtypes:

```text
float32  float64  int32  int64  uint32  uint64
```

Supported C-order shapes:

```text
(rows,)                 row scalar
(rows, 1)               row scalar
(rows, n_instruments)   vector
```

Dtype and row width are embedded into `InputSrc<Index, ValueType, RowWidth>`. A
width-one source broadcasts at typed load sites. Pure arithmetic whose inputs are
all row-scalar is lowered with lane count one; its scratch source broadcasts only if
a later vector operator consumes it.

The current arithmetic layer converts loaded numeric values to `float64`. The input
metadata and neutral key dtype are available for a later generic integer expression
lowering; no calendar-specific operator path is required.

## Correctness validation

The focused Linux workflow compiles and executes generated C++ for:

- exact `.npy` dtype/shape inspection and payload mapping;
- `int64 (rows,)` timestamp broadcasting against `float64 (rows, 9)` data;
- dense row-scalar `minute(_ev_ts)` groupby with grouped cumsum and EWM;
- independently hinted composite key tuples and mixed-radix capacity;
- width-one generated calendar stages;
- distinct hashed `int64` keys `2^53` and `2^53 + 1`.

The final focused run completed with ten passing tests.

## Non-gating same-run benchmark

GitHub Actions runner configuration:

```text
Ubuntu 24.04
1,000,000 rows x 9 instruments
1 warmup + 5 measured runs
output in /dev/shm
GCC native/LTO cpp_stream defaults
```

Two separate hosted-runner executions produced:

| Run | Vector/hash baseline | Typed row-scalar/dense | Speedup |
| --- | ---: | ---: | ---: |
| A | 4.117 M rows/s | 10.924 M rows/s | 2.65x |
| B | 4.407 M rows/s | 9.752 M rows/s | 2.21x |

The hosted CPU varied between runs, so absolute throughput is not a regression
threshold. In both same-run comparisons, the typed hinted path was substantially
faster.

The optimized path used:

```text
int64 _ev_ts.npy, shape (rows,)
float64 close.npy, shape (rows, 9)
Key(minute, num_keys=60, row_scalar=True, dtype="int64")
width-one floating calendar stages
dense resolver
```

It does not yet reach the earlier row-scalar integer-calendar ablation because the
generic calendar arithmetic still executes in `float64`.
