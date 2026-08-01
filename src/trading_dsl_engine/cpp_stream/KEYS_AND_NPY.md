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

The complete parameter semantics are also documented directly on
`trading_dsl_engine.base.keys.Key` and on the neutral IR's `GroupKeySpec`.

### `expr`

The dynamic expression whose value identifies a group. Wrapping it in `Key` does
not otherwise alter its mathematical meaning.

### `num_keys`

The number of consecutive non-NaN integer categories. When supplied, valid values
are:

```text
offset, offset + 1, ..., offset + num_keys - 1
```

This finite domain permits direct dense state indexing instead of hashing. A
floating-point key retains one additional NaN category.

### `offset`

The first valid value in the bounded domain. Dense routing maps a key value `v` to
zero-based digit `v - offset`.

Examples:

```python
Key(var("minute"), num_keys=60, offset=0)  # 0..59
Key(var("month"),  num_keys=12, offset=1)  # 1..12
Key(var("venue"),  num_keys=3,  offset=10) # 10, 11, 12
```

`offset` has no effect when `num_keys` is omitted.

### `row_scalar`

Whether one key value applies to every instrument lane in the row.

- `True`: evaluate the expression and resolve the group slot once per row, then
  broadcast that slot.
- `False`: each lane may contain a different key.
- `None`: infer lane invariance from mmap shapes and expression dependencies.

`True` is an assertion. Marking a lane-varying expression row-scalar changes
results and is therefore the caller's responsibility.

### `dtype`

The expected native scalar type of the completed key expression:

```text
float32  float64  int32  int64  uint32  uint64
```

For a direct input, the compiler verifies `dtype` against the `.npy` header or the
explicit `InputTypeSpec`. For a derived expression, it verifies the inferred native
result type. `dtype` is not permission to cast a mapped input.

For an explicitly integral key graph, every mapped input leaf must already have the
asserted integral dtype. Integral constants are compiled at that type only after
exact integrality and range validation.

## Tuple routing

A tuple may contain independently described keys. When every dynamic key has
`num_keys`, cpp_stream uses generic mixed-radix dense routing with capacity:

```text
product(num_keys_i + 1)
```

The additional digit preserves each floating key's NaN category. If any key is
unbounded, the complete tuple uses the generic fixed-capacity hash resolver. If all
keys are row-scalar, either resolver evaluates one tuple and broadcasts one group
slot.

Direct integer keys are read and hashed in their native input type. In particular,
`int64` and `uint64` values are not converted to `double` before key equality, so
values above `2^53` remain distinct. Dense integer range checks use exact 128-bit
intermediate arithmetic.

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

Supported C-order shapes:

```text
(rows,)                 row scalar
(rows, 1)               row scalar
(rows, n_instruments)   vector
```

Dtype and row width are embedded into:

```cpp
InputSrc<Index, ValueType, RowWidth>
```

A width-one source broadcasts only when consumed by a vector operation.

## Native expression typing

Mapped inputs are read with `read_native<Source>()`; `RowContext` no longer performs
an eager `static_cast<double>` for every read. Scratch sources and destinations also
carry their scalar type:

```cpp
SlotSrc<Index, ValueType, RowScalar>
SlotDst<Index, ValueType>
```

The runtime has separate typed scratch storage for all supported dtypes. Therefore,
an `int64` key expression can remain `int64` through every generated stateless stage:

```text
int64 _ev_ts
  -> int64 modulo
  -> int64 floor-division
  -> int64 floor identity
  -> int64 modulo
  -> dense group slot
```

Stateless operator policies are templated on their result type. A same-typed integer
operation does not pass through floating point. Mixed-type operations promote only
because that operation's declared result requires promotion, for example
`float64 + int64 -> float64`.

Stateful/statistical operators such as EWM and rank continue to define float64
semantics. The current root output file also remains float64, so an integer root is
converted only at final serialization.

## Correctness validation

The focused Linux workflow compiles and executes generated C++ for:

- exact `.npy` dtype/shape inspection and payload mapping;
- `int64 (rows,)` timestamp broadcasting against `float64 (rows, 9)` data;
- dense row-scalar `minute(_ev_ts)` groupby with integral key stages;
- independently hinted composite key tuples and mixed-radix capacity;
- width-one typed scratch stages;
- distinct hashed `int64` keys `2^53` and `2^53 + 1`.

The August 1, 2026 focused run completed with ten passing tests.

## Non-gating same-run benchmark

GitHub Actions runner configuration:

```text
Ubuntu 24.04
1,000,000 rows x 9 instruments
1 warmup + 5 measured runs
output in /dev/shm
GCC native/LTO cpp_stream defaults
```

Latest same-run comparison:

| Path | Median throughput |
| --- | ---: |
| vector float64 timestamp + vector floating calendar + hash resolver | 4.349 M rows/s |
| `int64 (rows,)` timestamp + integral row-scalar calendar + dense resolver | **22.244 M rows/s** |

The native typed path was approximately **5.11x faster** on that runner. Its five
measured runs were:

```text
22.280, 22.185, 22.273, 22.228, 22.244 M rows/s
```

Hosted CPUs vary, so absolute throughput is not a universal regression threshold.
The important architectural result is that the generic typed path now reaches the
earlier approximately 20-22 M rows/s integer-calendar ceiling without a
calendar-specific or grouped-operator-specific implementation.
