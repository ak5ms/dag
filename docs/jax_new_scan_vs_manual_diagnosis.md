# `jax_new` vs manual `lax.scan` (`xs_rank(close)` with `_xs_rank := jnp.sort`)

## Repro setup
- Formula path: `compile_formula("xs_rank(close)")` through `trading_dsl_engine.jax_new.engine.jit_batch`.
- Manual baseline: `jax.lax.scan(lambda carry, row: (carry, jnp.sort(row)), None, xs)`.
- Input shape used in this diagnosis run: `(4096, 9)` `float64` (smaller than 1y/minute, but enough to expose codegen structure).
- XLA dump flags: `XLA_FLAGS=--xla_dump_to=/tmp/xla_dump`.

## What differs architecturally

### 1) `jax_new` carries extra state through the scan body
The lowered/jaxpr form for `runtime.run_batch` carries two vector states through every timestep (`num_carry=2`), even though this specific workload (`xs_rank(close)` as plain `sort`) is stateless:
- carry A: previous raw input row (`i`)
- carry B: previous sorted row (`j`)

The scan body effectively returns `(i, j, broadcast(j))` each step, and the next step consumes both carry vectors.

Manual scan does not carry vectors for this case (`num_carry=0` when `init=None` and carry is unchanged), so the loop state is lighter.

### 2) `jax_new` emits width-1 matrix output and reshapes
`jax_new` output for this formula is `[..., 9, 1]`, not `[..., 9]`. That introduces extra `broadcast_in_dim` + concat/update plumbing in the loop body versus manual scan.

### 3) `jax_new` unrolls by 2
The `jax_new` scan lowers with `unroll=2`, so each while-iteration executes two sort calls and extra slice/reshape scaffolding for the pair.
Manual scan uses `unroll=1`.

This can help or hurt depending on kernel mix; here it increases body complexity around a tiny per-row sort.

## Evidence from generated code (beyond HLO)

### Buffer assignment footprint
From XLA CPU optimized buffer assignment dumps:
- `jit_jit_batch` (`jax_new`) has output/carry tuple state including `(s64[], f64[9], f64[9], f64[4096,9,1])` and additional 72-byte vector allocations reused as while-phi values.
- manual scan module has simpler while state centered on `(index, output[f64[4096,9]])` with no extra vector carries.

### Object code / asm size proxy
Disassembling emitted object files (`objdump -d`):
- `module_0008.jit_jit_batch.obj-file.part_00.o` (`jax_new`): **67** instruction lines.
- `module_0010.jit__lambda.obj-file.part_00.o` (manual scan): **33** instruction lines.

So the non-sort wrapper/control path in `jax_new` is roughly 2x larger in this artifact, consistent with the observed runtime delta for this microbenchmark.

## Why this is enough to explain ~2x slowdown here
With only 9 elements per row, each row-level `sort` is cheap. The fixed per-iteration scaffolding (carry shuffles, reshape/slice/update, extra tuple plumbing, width-1 output expansion) becomes a dominant cost. `jax_new` has more of that scaffolding than manual scan, and the asm/object footprint reflects it.

## Likely high-impact fix direction
1. Add a stateless fast-path in `jax_new` batch lowering:
   - if all nodes in program are stateless for the formula, lower to `vmap`/single `sort`-axis call or to scan with `num_carry=0`.
2. Avoid forcing `[..., n, 1]` materialization for purely vector outputs in batch path (defer/optional expand).
3. Revisit hardcoded `unroll=2` for small, cheap per-step kernels; make it heuristic.

These should preserve architecture while removing unnecessary loop state traffic.

## Assembly inspection details (requested deep dive)

I re-ran the repro with `XLA_FLAGS=--xla_dump_to=/tmp/xla_dump2` and disassembled the emitted object files with `objdump -d -Mintel`:

- `module_0004.jit_jit_batch.obj-file.part_00.o` (`jax_new` path): **67** instruction lines.
- `module_0006.jit__lambda.obj-file.part_00.o` (manual scan path): **33** instruction lines.

Notably, the DSL object contains both `region_2.4.clone` and `region_2.4` copies of the sort comparator path (duplicated compare/NaN-total-order logic), while the manual-scan object has only one `region_1.2` body for the same comparator sequence.

Representative duplicated fragment in `jax_new` object (appears in both `region_2.4.clone` and `region_2.4`):

```asm
vucomisd xmm1,xmm0
vmovq    rdx,xmm0
cmovne   rsi,rdx
cmovp    rsi,rdx
vucomisd xmm0,xmm0
cmovp    rsi,rdx
...
cmp      r9,r8
setl     BYTE PTR [rdi]
ret
```

This validates that at the native-code layer, the DSL variant has materially larger control/wrapper code around the same scalar comparator kernel, aligning with the runtime gap when per-row work is tiny (`N=9`).

### Tooling
No extra packages were required in this environment; GNU `objdump` was already available and sufficient for assembly inspection.


## Full assembly inventory and interpretation

To avoid sampling bias from a single object, I inspected **all** emitted object files for each compiled entrypoint from the same repro run (`/tmp/xla_dump2`):

- DSL path (`module_0004.jit_jit_batch.obj-file.*.o`): **19 objects**, **2138 total instruction lines**.
- Manual scan path (`module_0006.jit__lambda.obj-file.*.o`): **6 objects**, **137 total instruction lines**.

Largest DSL contributors:
- `call.6_computation_kernel_module.o`: 445
- `call.7_computation_kernel_module.o`: 445
- `bitcast_dynamic-update-slice_fusion_kernel_module.o`: 165
- `multiply_select_fusion_kernel_module.o`: 153
- `xor_select_fusion(.1)_kernel_module.o`: 139 / 141
- `broadcast_select_fusion_kernel_module.o`: 112
- `is-finite_select_fusion_kernel_module.o`: 110

Manual-scan contributors stay small and mostly structural:
- `part_00.o`: 33
- `bitcast_dynamic-update-slice_fusion_kernel_module.o`: 26
- `dynamic-slice_bitcast_fusion_kernel_module.o`: 26
- `wrapped_broadcast_kernel_module.o`: 24

### What this implies architecturally

The gap is not only from the sort comparator itself. It is dominated by **graph shape inflation** around the comparator:
1. Two `call.*_computation_kernel_module` objects in DSL reflect duplicated step scaffolding work around unrolled execution.
2. Multiple select/fusion kernels (`is-finite`, `broadcast_select`, `xor_select`, `multiply_select`) are emitted for generic NaN/shape handling paths that the manual sort baseline does not need.
3. Width-1 output materialization (`[..., n, 1]`) plus dynamic-update-slice/broadcast patterns increases auxiliary kernels.

In short: for tiny row width (`n=9`), fixed wrapper kernels dominate, while manual scan stays close to “sort + minimal loop shell.”

## DSL architectural changes to close the gap

### A) Introduce a stateless-vector batch lowering lane
At compile/lower time, detect formulas whose DAG is stateless and vector-output only. Route them to a dedicated batch lane that:
- bypasses state-carry scan tuples,
- lowers as `vmap(step_fn)` or direct axis op where available,
- emits `[T, N]` and applies `[..., None]` only at external API boundary if required.

This keeps the general state-machine architecture intact while removing scan baggage for stateless formulas.

### B) Separate internal tensor rank from external schema rank
Keep internal runtime values in natural rank (`[N]` for vector per tick, `[T,N]` for batch). Add a cheap view/reshape adapter only at the final output contract boundary.
This avoids per-step broadcast/concat/update overhead solely to maintain `[N,1]` internal shape.

### C) Unroll policy should be cost-model driven
Replace hardcoded `unroll=2` with a heuristic keyed by:
- estimated per-step FLOPs / memory traffic,
- op count in step graph,
- row width `N` and output rank.

For tiny per-step kernels, prefer `unroll=1` to avoid duplicate call scaffolding.

### D) Add op-level lowering hints in registry metadata
Extend op specs with compile-time tags such as:
- `stateful: bool`
- `batch_fusable_axis: Optional[int]`
- `requires_nan_masking: bool`
- `output_rank: scalar|vector|matrix`

Then let lowering choose specialized pipelines without hardcoding operator names in the engine.

### E) Canonicalize NaN semantics once per row when possible
For chains of stateless ops that all require the same finite mask policy, hoist/shared-mask lowering can reduce redundant `isfinite/select/xor_select` kernels.
This preserves semantics while shrinking emitted helper kernels.

### F) Add a “microkernel benchmark gate” in CI for JAX backend
Track representative small-width workloads (like `xs_rank(close)` with `N<=16`) and assert relative slowdown thresholds against a manual-scan reference.
This prevents regressions where wrapper kernels quietly outweigh useful computation.


## Stateful-subgraph vectorization opportunity (e.g. `ewm(xs_rank(close+open), 21)`)

For mixed formulas, we should split the DAG into:
- a **stateless prefix subgraph** (per-timestep pure functions), and
- a **stateful suffix subgraph** (ops like `ewm`, `cumsum`, etc.).

Example: `ewm(xs_rank(add(close, open)), 21)`
- `add` + `xs_rank` are stateless and can be lowered as batch vectorized transforms over `[T, N]`.
- only `ewm` needs scan carry.

A practical lowering plan:
1. During compile/lower, mark each node with `is_stateful` and compute topo-closure.
2. Materialize stateless node outputs in batch once (`vmap`/axis-native op) to produce an intermediate stream.
3. Run `lax.scan` only for the minimal stateful frontier, carrying only required state structs.
4. Re-join with any trailing stateless nodes outside the scan when possible.

This preserves semantics but shrinks scan body and carry traffic for common mixed formulas.

## State-passing diagnosis from assembly/JAX IR

Yes — current `jax_new` batch path passes the **full runtime state tuple** through scan even when much of it is logically stateless for a given formula path.

Signals:
- JAX IR shows `scan(... num_carry=2 ...)` for `xs_rank(close)`-style path even though the core op is stateless.
- Buffer assignment shows while-phi tuples containing multiple vector carry buffers (`f64[9]`, `f64[9]`) plus output tensor.
- Assembly/object inventory includes multiple wrapper/call kernels tied to tuple/shape plumbing, not just comparator work.

So the overhead is not from Python object passing (there is none in JIT), but from **compiled tuple state carried in while-loop SSA form** plus associated reshape/update helper kernels.

### Architectural change to improve state passing

Introduce a compact scan state layout generated per program:
- carry only nodes with `is_stateful=True` **or** nodes required as recurrent dependencies for those stateful nodes,
- keep stateless intermediates as loop-local values (or precomputed batch tensors),
- avoid carrying final-output formatting tensors (`[..., N, 1]`) in scan state.

That directly reduces while-phi arity, buffer alias pressure, and helper-kernel count.
