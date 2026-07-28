# Native C++ runtime guide

This document explains the native JAX-flat runtime for a reader who is comfortable
with Python but still learning C++. The main implementation is
[`src/trading_dsl_engine/jax_flat/ops.cpp`](../src/trading_dsl_engine/jax_flat/ops.cpp).
The Python lowering boundary is
[`engine_cpp.py`](../src/trading_dsl_engine/jax_flat/engine_cpp.py).

## 1. The short mental model

The native runtime is a flattened computation graph. In Python-like pseudocode:

```python
class Runtime:                         # immutable compiled formula
    nodes: list[NodeSpec]
    prepared_nodes: list[PreparedNode]
    output_id: int

class State:                           # mutable values for one stream
    values: list[NodeValue]            # preallocated output/scratch per node
    operator_state: list[...]          # ewm, shift, ridge, groupby, ...

def eval_row(runtime, state, input_rows, output):
    for node_id, spec in enumerate(runtime.nodes):
        run_kernel(spec, state, input_rows, state.values[node_id])
    copy_or_direct_write_root(output)
```

`Runtime` owns facts that do not change between ticks. `State` owns everything that
changes as data arrives. Keeping that distinction is important: parsing a string,
building a lookup table, or allocating a vector belongs in `Runtime` construction;
updating an EWM accumulator belongs in `State` and `eval_row`.

The graph is topologically ordered, so every child has already been evaluated when
its parent runs. This is the C++ equivalent of iterating through a list of Python
callables, except the switch dispatch and numeric loops run without Python.

## 2. Important C++ syntax in this file

| C++ | Rough Python analogy | Meaning here |
| --- | --- | --- |
| `enum class OpCode` | `Enum` | Stable identifier for a native kernel. |
| `struct NodeSpec` | `@dataclass(frozen=True)` | Immutable lowering parameters and child IDs. |
| `std::vector<T>` | `list[T]` backed by contiguous memory | Dynamically sized owned storage. |
| `std::array<T, N>` | fixed-size tuple/list | Compile-time-sized metadata table. |
| `const T&` | borrowed read-only object | No copy and no ownership transfer. |
| `T&` | borrowed mutable object | The callee may update the original. |
| `T*` | address/buffer pointer | Used for NumPy/Eigen-compatible contiguous data. |
| `constexpr` | evaluated like a constant at build time | No runtime table construction. |
| `static_cast<size_t>(x)` | `int(x)` with an explicit target type | Converts enum/index types deliberately. |
| `std::move(x)` | transfer ownership | Avoids copying a vector or graph. |
| `friend class State` | privileged access declaration | Lets `State` use private `Runtime` layout helpers. |

References and pointers do **not** keep an object alive. The owner must outlive every
borrow. This is why tick inputs retain their `py::array_t` owners for the complete
native call rather than retaining only NumPy data pointers.

## 3. From Python formula to native execution

1. Python compilation produces a `StreamingProgram`.
2. `engine_cpp.py` lowers each node to the typed `NativeExecutionPlan`, then to the
   compact tuple ABI consumed by the extension.
3. C++ parses each tuple into a `NodeSpec` and an `OpCode`.
4. `Runtime` assigns state slots and builds any immutable prepared data.
5. `init_state(n_instruments)` allocates all per-stream buffers once.
6. `tick_into` or `run_batch_into` validates Python arrays at the boundary.
7. `eval_row` walks the same flattened transition for a live tick or each batch row.

The batch implementation intentionally calls the tick transition repeatedly. This
prevents a fast batch path from quietly acquiring different NaN or state semantics.
The loop is in C++, with the GIL released, rather than in Python.

## 4. `OpMetadata`: one registry for construction traits

`OpCode` answers **which kernel runs**. `OpMetadata` answers **what generic runtime
machinery that kernel needs**. It is similar to a Python dictionary:

```python
OP_METADATA = {
    OpCode.Einsum: OpMetadata(
        prepare=PrepareKind.Einsum,
        output_rows=OutputRows.EinsumDerived,
    ),
    OpCode.Shift: OpMetadata(state=StateKind.Shift),
}
```

The real C++ table is a `constexpr std::array`, indexed by the integer value of the
enum. All entries begin with common defaults, then exceptional traits are assigned.
The fields are:

### `prepare`

Selects immutable work performed once in `Runtime` construction. Examples:

- `Einsum`: parse subscripts into axis/label positions.
- `FutureBasis`: build the reusable suffix lookup table.
- `None`: no prepared data.

This is not kernel execution. It prevents string parsing and heap allocation on each
row. `PreparedNode` is parallel to `nodes_`, so `prepared_nodes_[node_id]` always
belongs to `nodes_[node_id]`.

### `output_rows`

Describes axis 0 of `NodeValue`:

- `Instruments`: one row per instrument.
- `Fixed`: a scalar or fixed mathematical vector, independent of instrument count.
- `ModelProjection`: depends on the child model family.
- `EinsumDerived`: inferred from the contraction signature and child layouts.

This replaces scattered `if opcode == ...` shape allowlists. The policy can still
contain family-specific shape logic; the important distinction is that generic state
construction dispatches on a declared layout policy.

### `state`

Selects the state storage family. Multiple future opcodes can reuse `Value`, `Shift`,
`Ridge`, or another existing state representation without changing state-slot
indexing. A genuinely different state representation still requires a named struct
and one construction case.

### `direct_root_write`

Allows an eligible root kernel to write directly into the caller's output buffer,
skipping the final scratch-to-output copy. This must remain explicit. A kernel is not
eligible if it or later logic expects the root's `NodeValue::data` scratch buffer to
contain the newly written result.

### `needs_rank_scratch`

Requests runtime-wide reusable ranking scratch. The allocation occurs once when any
node declares the trait; `xs_rank` does not allocate items or normal scores per tick.

## 5. `NodeValue` shape and `rows_kind`

Every node owns a flat `std::vector<double>`, plus enough metadata to interpret it:

```text
rows_kind = instrument-aligned  -> rows = n_instruments
rows_kind = fixed              -> rows = fixed_rows
flat offset                    -> row * width + column
```

This representation covers scalars, instrument vectors, fixed coefficient vectors,
and matrices without a Python object or variant in the hot loop. `width` is the
second dimension. A literal normally has one fixed row and width one; a basis result
normally has `n_instruments` rows and `n_basis` width.

`configure_value_layout` processes nodes in topological order. Consequently, an
einsum may inspect already configured child layouts when deciding whether its single
output label represents instruments or a fixed feature axis.

## 6. Runtime preparation versus streaming state

It helps to ask: “Would two independent streams using the same compiled formula
share this value?”

- **Yes:** put it in `Runtime`/`PreparedNode`. Examples: parsed einsum plan, static
  future-basis table, opcode metadata.
- **No:** put it in `State`. Examples: EWM accumulator, shift ring, ridge sufficient
  statistics, group hash slots.

`Runtime` may be shared conceptually; `State` must not be shared between independent
streams. `init_state` preallocates node values and typed state arrays based on
`StateKind`. `eval_row` then mutates only that state and caller output.

## 7. Tick and batch memory ownership

At the Python boundary, pybind may need to convert a non-contiguous or non-`float64`
array. Such a conversion creates a temporary owning Python object. `tick_input_owners_`
keeps those objects alive until `eval_row` completes; `row_ptrs_` alone would be unsafe.

For batch input, validation happens once. `batch_base_ptrs_` stores each contiguous
array base address, and the inner loop advances it by `t * n_instruments`. Upcoming
rows are prefetched as a compiler/CPU hint. Prefetching is only a hint: correctness
must never depend on it.

The GIL is released only after validation and ownership retention. No `py::...` API
may be called while the GIL is released or from numeric kernel code.

## 8. Adding or changing a native operator

Use this checklist:

1. Add/confirm the pure JAX-flat implementation and batch scan semantics.
2. Add the native `OpCode` and string parsing/lowering mapping.
3. Implement the numeric kernel in `eval_row` or a focused helper.
4. Declare non-default construction traits in `make_op_metadata`:
   - prepared data?
   - non-instrument output layout?
   - specialized state family?
   - reusable scratch?
   - safe direct-root write?
5. If a new preparation or state **representation** is required, add one enum value,
   one named storage struct/field, and one generic dispatch case. Do not add a new
   parallel per-op vector or allowlist.
6. Test native versus `cpp=False`, including NaN and shape behavior.
7. For stateful ops, compare batch results with repeated live ticks.

Operator-specific arithmetic remains operator-specific; construction and storage
traits are what the metadata generalizes. Trying to encode all arithmetic in metadata
would merely create a slower, harder-to-read interpreter.

## 9. Editable `.so` rebuilds

[`src/trading_dsl_engine/_native_build.py`](../src/trading_dsl_engine/_native_build.py)
protects editable checkouts from stale binaries:

1. Start at each extension's translation unit (`engine.cpp` or `eigen_nnqp.cc`).
2. Recursively follow repository-local quoted includes such as `"ops.cpp"` and local
   headers.
3. Hash paths, contents, `setup.py`, `pyproject.toml`, compiler environment, Python
   ABI, and platform.
4. Compare the hash with the ignored JSON stamp beside the `.so`.
5. If stale, acquire a filesystem lock, recheck (another process may have built),
   run one forced build, and atomically replace both stamps only after success.

Installed wheels normally lack the repository `setup.py`; they are treated as
immutable and never attempt a local rebuild.

## 10. Debugging map

| Symptom | First place to inspect |
| --- | --- |
| Formula remains a JAX island | `engine_cpp.py` lowering and `cpp_name`/opcode mapping |
| Wrong output shape | `OpMetadata::output_rows` and `configure_value_layout` |
| State leaks across formulas/groups | `StateKind`, state slot assignment, group state |
| First tick correct, later ticks wrong | state update kernel and ring/count semantics |
| Batch differs from tick loop | `eval_row` inputs or Python wrapper, since transition is shared |
| Crash after dtype/contiguity conversion | pybind owner lifetime before GIL release |
| Stale `.so` | fingerprint stamp and transitive quoted-include closure |
| Extra copy at root | `direct_root_write` safety and final root copy |

When debugging numeric behavior, first force `cpp=False` and compare the same compact
input cartesian product against native execution. Treat NaNs, infinities, zero
denominators, shape-changing outputs, and the first few streaming transitions as
separate regimes.
