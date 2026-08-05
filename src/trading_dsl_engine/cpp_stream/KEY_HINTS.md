# cpp_stream key hints

`Key(...)` attaches grouping metadata without changing the mathematical value of
an expression. Hints are optional. The compiler remains correct without them but
may need per-lane hashing and a larger generic state table.

```python
from trading_dsl_engine.base.dsl import groupby, cumsum, self_, var
from trading_dsl_engine.base.keys import key

minute = key(
    var("minute"),
    num_keys=60,       # valid values are exactly 0..59
    offset=0,
    row_scalar=True,   # one minute value applies to every instrument this row
    dtype="int64",    # assertion, not a conversion
)

result = groupby((minute,), var("x"), cumsum(self_))
```

## Hint meanings

- `num_keys`: Declares a finite consecutive integer domain. When every dynamic
  key is bounded, cpp_stream uses direct mixed-radix indexing instead of hashing.
  Do not use it for absolute timestamps or identifiers whose domain grows.
- `offset`: First value in the bounded domain. For months 1 through 12, use
  `num_keys=12, offset=1`.
- `row_scalar`: Declares that all instrument lanes have the same key on a row.
  cpp_stream resolves that key once and broadcasts the group slot. A false
  declaration changes results, so use it only for genuinely lane-invariant data.
- `dtype`: Verifies the native key dtype and prevents an accidental conversion
  from being hidden. It does not cast the source.
- `monotonic`: Declares a row-scalar key whose equal-value runs are contiguous and
  never return after the value changes. cpp_stream resets the grouped RHS state at
  each change and reuses one slot for that key. A false declaration changes
  results. The hint requires `row_scalar=True`.

## Common patterns

### Bounded calendar key

```python
weekday = key(
    var("weekday"),
    num_keys=7,
    offset=0,
    row_scalar=True,
    dtype="int32",
)
```

### One key per instrument

```python
sector = key(
    var("sector_id"),
    num_keys=32,
    row_scalar=False,
    dtype="int32",
)
```

### Monotonic session epoch

```python
session = key(
    var("session_start0"),
    row_scalar=True,
    dtype="float64",
    monotonic=True,
)
```

Absolute session timestamps are not a bounded key domain, so `num_keys` is
intentionally omitted. Because the session timestamp only advances and never
returns, cpp_stream does not retain a hash-table entry for every historical
session. It resets the grouped RHS at a transition and recycles capacity one.
This is the optimization used by `flows.roll_rets_hints.roll_rets_hints`.

A tuple can combine monotonic epoch keys with ordinary keys. The ordinary keys
retain dense or hashed slots only within the current epoch; all of those slots
are reset together when an epoch key changes.

## Automatic inference

When `row_scalar=None`, cpp_stream infers lane invariance from physical input
shape and expression dependencies. This works automatically for a source stored
as `(rows,)`. A source stored redundantly as `(rows, instruments)` appears
lane-varying to the compiler even when each row happens to contain identical
values; `row_scalar=True` is the explicit assertion for that case.
