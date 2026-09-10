# Tensor types in the strongly typed GP grammar

`cpp_stream` processes one time row at a time.  The streamed time dimension is
therefore not part of the GP value type:

| Physical runtime shape | Per-row shape | GP type rank |
|---|---|---:|
| `(time, instrument)` | `(instrument,)` | row |
| `(time, instrument, level)` | `(instrument, level)` | tensor rank 2 / matrix |
| `(time, instrument, level, channel)` | `(instrument, level, channel)` | tensor rank 3 |
| `(time, instrument, f1, ..., fk)` | `(instrument, f1, ..., fk)` | tensor rank `k + 1` |

The default `book_price` and `book_volume` terminals are matrices assembled from
the existing ten ask and ten bid price or volume columns.  External tensors can
be configured with arbitrary positive feature shapes through `TensorFieldSpec`.

## Tensor semantics

Every active rank has separate numeric, derived, dimensionless, boolean, count,
book-price, and book-volume types.  Rank and semantic type are enforced by DEAP.
Concrete feature extents are validated globally: all values that can meet at a
particular rank, including values produced by reducing a higher-rank tensor,
must have the same feature-prefix shape.

For example, `(instrument, 5, 3)` reduces to `(instrument, 5)`.  It may compose
with another `(instrument, 5)` matrix, but not with an `(instrument, 20)` matrix.

## Shape behavior

| Operator class | Shape behavior | Higher-rank behavior |
|---|---|---|
| Elementwise arithmetic, comparisons, transforms, `where`, `clip` | preserves all feature axes | registered for every configured rank |
| `shift`, `diff`, `ffill`, `cumsum` | preserves all feature axes | lane-wise for every configured rank |
| EWM and rolling lane-wise utilities | preserves all feature axes | registered for every configured rank when the backend utility is lane-wise |
| `vec_*` | reduces the final feature axis exactly once | rank 4 -> rank 3 -> rank 2 -> row through composition |
| Cross-sectional/stateless matrix Ridge | matrix -> row projection | intentionally rank 2 only |
| Temporal matrix Ridge | row target plus matrix regressors -> row projection | intentionally rank 2 only |

The `vec_*` families are `vec_avg`, `vec_choose`, `vec_count`, `vec_ir`,
`vec_kurtosis`, `vec_max`, `vec_min`, `vec_norm`, `vec_percentage`,
`vec_powersum`, `vec_range`, `vec_skewness`, `vec_stddev`, and `vec_sum`.

## Utility-selection hierarchy

The grammar uses the least duplicative implementation available:

1. expose an already valid DSL operation directly;
2. otherwise call the existing implementation in
   `trading_dsl_engine.cpp_stream.python.utils`;
3. add a GP-only wrapper only to bind dependent parameters, enforce a legal
   type signature, or consume a non-row intermediate inside a row-producing
   composition.

No raw Ridge model, free-form `cat`, `groupby`, `einsum`, `emit`, or cache node
is added to the GP type graph.  Those operations may still appear internally in
a predefined composition whose externally visible result has a valid typed
shape.

## External higher-rank input

```python
from flows.gp import GPConfig, TensorFieldSpec, gp_input_types, make_pset

config = GPConfig(
    tensor_fields=(
        TensorFieldSpec("book", "price", (20, 2)),
    ),
    tensor_indices=(0, 1),
)
pset = make_pset(config)
input_types = gp_input_types(config, n_instruments=9)
```

Here `book` has physical runtime shape `(time, 9, 20, 2)` and GP rank 3.
Applying one `vec_avg(book)` produces a rank-2 `(instrument, 20)` matrix;
applying another produces an instrument row.
