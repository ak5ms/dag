# Unified cpp_stream sources

`cpp_stream` has one compilation API and one execution API:

```python
from trading_dsl_engine.cpp_stream import compile_formula

runtime = compile_formula(
    "xs_rank(ewm(close / open, 21))",
    {
        "close": "/data/close.npy",
        "open": "/data/open.npy",
    },
    n_instruments=9,
)
runtime.run(out_path="/data/alpha.bin")
```

There is no `.npy`-specific compiler or runner. Every formula input is inspected and
opened independently. The adapter is inferred from the input object, URI scheme, or
file extension, so one formula may use different source formats at the same time.

## Mixed formats

```python
from trading_dsl_engine.cpp_stream import InputTypeSpec, compile_formula, source

runtime = compile_formula(
    "left + right",
    {
        "left": "/data/left.npy",
        "right": source(
            "/data/right.bin",
            input_type=InputTypeSpec("float64", 9),
        ),
    },
    n_instruments=9,
)
runtime.run(out_path="/data/result.bin")
```

The `.npy` adapter reads dtype and full per-row tensor shape from the header. A raw
`.bin` or `.raw` source has no header, so its dtype and row shape must come from an
`InputTypeSpec`, either in `source(...)` or in `compile_formula(input_types=...)`.
All prepared sources are validated for compatible row counts before the single native
entrypoint is called.

Built-in adapters currently cover:

- C-order `.npy` files, zero-copy mapped;
- headerless `.bin` and `.raw` files, zero-copy mapped with explicit metadata;
- C-contiguous in-memory NumPy arrays.

Any positive C-order per-row tensor shape is valid. `(rows,)` and `(rows, 1)` are row
scalars; `(rows, N)`, `(rows, N, K)`, and higher ranks preserve their complete
per-row shape.

## Replacing bound sources

Sources supplied to `compile_formula` are bound to the returned runtime. A compatible
mapping can be substituted at execution time without recompiling:

```python
runtime.run(
    {
        "close": "/new/day/close.npy",
        "open": "/new/day/open.npy",
    },
    out_path="/data/new_alpha.bin",
)
```

The replacement sources must have the dtype and row shape embedded in the compiled
plan. Format does not have to match the originally bound source. For example, a
compiled `.npy` input may be replaced by a raw file carrying the same
`InputTypeSpec`.

## Custom formats and URI schemes

Parquet, Arrow, object stores, shared-memory feeds, and network streams are source
concerns rather than compiler variants. Add them by registering a `SourceAdapter`:

```python
from trading_dsl_engine.cpp_stream import register_source_adapter

register_source_adapter(MyParquetAdapter())
register_source_adapter(MyNetworkFeedAdapter())
```

Each adapter independently implements:

```python
class SourceAdapter(Protocol):
    name: str

    def accepts(self, item: InputSource) -> bool: ...
    def inspect(self, item, *, expected: InputTypeSpec | None) -> SourceInfo: ...
    def open(self, item, *, expected: InputTypeSpec | None) -> PreparedSource: ...
```

`accepts` may inspect an extension such as `.parquet`, a URI scheme such as
`tcp://`, or a custom Python source object. `inspect` supplies compile-time dtype,
row shape, and row-count metadata. `open` returns a live contiguous buffer pointer
and an owner/close callback for the duration of execution.

This keeps source discovery, decoding, buffering, and lifecycle outside generated
operator code. The generated C++ sees only typed pointers, row counts, and row
widths, regardless of source origin.

A future true incremental network adapter may provide chunks through the same source
layer; it should not add a `compile_network_formula` or `run_network_files` API.

## Dynamic group-key metadata

Source typing also drives group-key optimization. For example, an `int64` row-scalar
`.npy` timestamp remains integral through calendar arithmetic and can route directly
to bounded dense group state. Native integer equality is preserved, including values
above `2^53`.

```python
Key(
    var("minute"),
    num_keys=60,
    offset=0,
    row_scalar=True,
    dtype="int64",
)
```

`dtype` validates the completed expression type; it does not authorize an implicit
source cast.
