# JAX-flat batch lowering: 1M x 9 diagnosis

## Scope and measurement

The runtime diagnostic API deliberately displays the **tick** transition.  For
`cumsum(var("mp_out0.close"))`, its JAXPR/HLO is consequently small: one row of
nine values enters, the cumulative state is updated, and one row exits.  It is
not a representation of the batch executable, so it cannot answer whether a
million-row batch materializes intermediates or uses one loop.

Use the repeatable CPU benchmark below when assessing a lowering change.  It
uses float64, `cpp=False`, synchronizes before timing, and compares the current
operator-wise batch lowering with a single `lax.scan(runtime.tick, ...)`:

```bash
RUN_PERF_TESTS=1 pytest -n 0 -q \
  tests/trading_dsl_engine/jax_flat/test_performance.py \
  -k batch_lowering_strategies_1m_by_9_assets
```

The three cases separate important hypotheses:

| Formula | Hypothesis exercised |
| --- | --- |
| `cumsum(close)` | A primitive prefix scan should favor the specialised whole-sequence kernel. |
| `ewm(cumsum(close), 21)` | Two dependent stateful nodes reveal time-axis intermediate materialization and scan boundaries. |
| `cumsum(xs_sort(close))` | A cross-sectional producer verifies that a remedy remains valid for non-elementwise, multi-instrument operators. |

On this CPU environment (JAX 0.11, X64, 1,000,000 x 9), warm measurements were
0.421 s versus 0.407 s for `cumsum`, 0.770 s versus 0.873 s for the dependent
`ewm(cumsum(...))` chain, and 0.652 s versus 1.039 s for
`cumsum(xs_sort(...))`; each pair is operator-wise batch then tick scan.  These
numbers are directional, not a portable threshold: compiler, device, and
memory bandwidth change them.  The benchmark retains both alternatives and
checks their outputs, rather than encoding an accidental winner as a test
failure.

## Architectural causes

1. **Diagnostics and batch use different execution graphs.**  `get_jaxpr()` and
   `get_compiled_hlo()` lower one `tick`; batch lowers each DAG node through its
   own `scan_batch`.  A compact tick HLO therefore says little about the batch
   schedule.
2. **The batch planner is node-oriented.**  It propagates a complete
   `(time, instrument, ...)` sequence from one node to the next.  Stateful
   chains can consequently create a full intermediate at each dependency.
   The required root output alone is 72 MB at 1M x 9 float64; an input plus the
   cumsum-to-EWM intermediate plus the root output is already at least 216 MB,
   before compiler temporaries and state.
3. **A fixed 65,536-row chunk is assembled into a full device output.**  The
   in-memory path allocates the full result and uses dynamic updates for each
   chunk.  Chunking bounds individual operator working sets, but it does not
   remove root-output allocation or fuse a stateful dependency chain.
4. **Operator-local fast paths trade fusion for algorithm quality.**  Direct
   `cumsum` maps to `jnp.cumsum`, which is faster than a generic tick scan in
   the benchmark.  Replacing all `scan_batch` implementations with one global
   scan would regress this case and cross-sectional work.  This is why a
   formula-specific or graph-level decision is necessary.

## Remediation plan

The first generic scheduling improvement is implemented: batch DAG execution
now computes static use counts and drops each chunk sequence immediately after
its last consumer.  This makes liveness explicit in the traced graph, so XLA
may reuse the chunk buffer rather than retaining every topological node value.
Root outputs and `cache(...)` values are terminal consumers and remain live by
design.  The same release rule is applied to nested groupby RHS graphs.

The next step is a **costed, region-based batch planner**, not a central
per-operator branch:

1. Let every op advertise a compositional batch-lowering capability: a
   whole-sequence implementation, tick-compatible implementation, estimated
   temporary/output shape, and state boundary.  Keep this on `Op`/factory
   metadata so new operators participate without edits to a central switch.
2. Partition the topologically sorted DAG into regions.  Preserve specialised
   sequence regions for prefix/rolling kernels when they win; fuse adjacent
   tick-compatible stateful and stateless nodes into one `lax.scan` region when
   it avoids a large live time-axis intermediate.  Cross-sectional operations
   remain normal nodes inside a region, so this extends to `xs_sort`, groupby,
   matrices, and arity greater than one.
3. Choose the partition with a simple static cost model: live bytes per chunk,
   number of full time-axis materializations, and operation capability.  Make
   the chunk size an explicit planner input rather than an implicit global
   constant.  Validate candidate schedules against the identical streaming
   `tick` ABI.
4. Stream region outputs to the existing disk-backed output path when consumers
   do not require their full history.  Retain full materialization only for a
   root result, `cache(...)`, or a downstream whole-sequence region.  This
   improves 1M-row memory use without altering live state semantics.

The acceptance matrix should retain the three benchmark formulas above and add
representatives for grouped state, lag-history matrices, and n-ary stateless
chains.  For each, compare outputs and final state to a tick scan with finite,
partial-NaN, and all-NaN inputs.  That tests the generic composition boundary,
instead of optimising `cumsum` by name.
