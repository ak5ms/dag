# cpp_stream examples

## Full DSL after reductions

Reduction axes follow the complete streamed array shape. For an expression with
per-row shape `(instruments, features)`, axis 0 is time, axis 1 is instruments,
and axis 2 is features.

```python
from trading_dsl_engine.base.dsl import cat, var
from trading_dsl_engine.cpp_stream import compile_formula

pnls = cat(alpha_pnl_1, alpha_pnl_2, alpha_pnl_3, alpha_pnl_4)
pnl = pnls.sum(axis=1)  # per row: reduce instruments, retain four features

sharpe = (
    pnl.cumsum()
    / (pnl ** 2).cumsum().pow(0.5)
).emit("last")

runtime = compile_formula(sharpe, data)  # n_instruments inferred
result = runtime.run()                   # temporary .npy output
values = result.load()                   # shape inferred; no reshape arguments
```

Fixed-width and arbitrary tensor results now support ordinary elementwise DSL
operators plus `cumsum`, `ewm`, `ffill`, and `shift`.

## Composed temporal reductions

A temporal reduction may feed downstream DSL algebra:

```python
sharpe = pnl.sum(axis=[0, 1]) / pnl.std(axis=[0, 1])
runtime = compile_formula(sharpe, data)
result = runtime.run()
scalar = result.load()
```

The temporal reductions expose their cumulative state to the downstream graph on
each row. Because the final expression depends on a temporal reduction,
cpp_stream implicitly retains only its last value. This remains a one-pass stream;
the time axis is not materialized.

An explicit `emit("last")` remains terminal. It means “return the final value of
this temporal expression,” not “perform a reduction.”

## Output formats

`runtime.run()` and `.npy` paths write a valid NumPy file directly. Native C++
maps the payload after the NumPy header, so there is no conversion or full-output
copy:

```python
result = runtime.run(out_path="alpha.npy")
alpha = result.load()  # np.memmap by default
```

Raw output remains available by choosing `.bin` or `.raw`:

```python
result = runtime.run(out_path="alpha.bin")
alpha = result.load()  # uses RunResult.output_shape automatically
```

## Automatic instrument inference

The public compiler inspects source row shapes. The most frequent non-scalar
leading row extent is selected as `n_instruments`. This handles mappings with many
`(rows, instruments)` market arrays and a smaller number of fixed feature arrays.
Ambiguous ties and scalar-only mappings require an explicit `n_instruments`
rather than an unsafe guess.

## Key hints

See [KEY_HINTS.md](KEY_HINTS.md) and `flows.roll_rets_keys`. Key metadata changes
physical routing only; it does not change the expression's mathematical value.
