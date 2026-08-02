# Streaming reductions

Reductions use the same expression API as other operations:

```python
result = compile_formula((x * y).sum(axis=0), data).run(out_path="sum.bin")
row_mean = x.mean(axis=1)
feature_std = cat(x, y).std(axis=[0, 1], ddof=1)
last_cumulative = x.cumsum().emit("last")
```

Axes are interpreted against the logical materialized result `(time, *row_shape)`.
Axis `0` is therefore time. A reduction containing axis `0` is evaluated online and
emits one final output; it never creates the intermediate time-sized result. Axes
that do not contain `0` reduce the current row and continue to emit one result per
input row, so they compose normally with subsequent operations.

Temporal reductions and `emit("last")` are terminal because they remove the streaming
time dimension. Row reductions can appear anywhere in the graph. `sum`, `mean`, and
`std` ignore non-finite observations; empty groups and standard deviations with
`count <= ddof` produce NaN. Standard deviation uses an online Welford accumulator.

`RunResult.rows` remains the number of input rows processed, preserving throughput
reporting. `RunResult.output_rows`, `output_shape`, and `output_mode` describe the
materialized output. A temporal reduction has `output_rows == 1` and stores only one
fixed-size result in the output file.
