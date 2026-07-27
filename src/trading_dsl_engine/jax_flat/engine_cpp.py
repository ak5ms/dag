from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from enum import Enum
import json
from typing import Any, Callable
import os
import tempfile

import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import DSLFunctionRegistry
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.jax_flat.engine import (
    DagNode,
    InnerGraphOp,
    StateFieldRef,
    StateLayout,
    StreamingProgram,
    compile_formula as compile_formula_jax,
)
from trading_dsl_engine.jax_flat.ops import (
    CacheOp,
    CumsumOp,
    EwmOp,
    FFillOp,
    FutureRbfBasisSumOp,
    GroupByOp,
    InputOp,
    InstrumentBasisMeanOp,
    LiteralOp,
    NaryOp,
    Op,
    RbfBasisOp,
    RidgeOp,
    RollingMeanOp,
    RollingOp,
    ShiftOp,
    BufferShiftOp,
)


@dataclass(frozen=True)
class CppPlanNode:
    """One node in the inspectable JAX-flat/native lowering plan."""

    id: int
    operation: str
    backend: str
    island: int
    children: tuple[int, ...]
    reason: str | None = None


@dataclass(frozen=True)
class CppLoweringPlan:
    """Static explanation of the native/JAX partition selected at compile time."""

    nodes: tuple[CppPlanNode, ...]
    outputs: tuple[int, ...]
    input_names: tuple[str, ...]
    unsupported_functions: tuple[str, ...] = ()

    @property
    def missing_cpp_functions(self) -> tuple[str, ...]:
        direct = (
            node.operation
            for node in self.nodes
            if node.backend == "jax" and not (node.operation == "groupby" and node.reason and node.reason.startswith("nested RHS"))
        )
        return tuple(sorted({*direct, *self.unsupported_functions}))

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": self.input_names, "outputs": self.outputs, "missing_cpp_functions": self.missing_cpp_functions, "nodes": [asdict(node) for node in self.nodes]}

    def to_dot(self) -> str:
        lines = ["digraph jax_flat_plan {", '  rankdir="LR";']
        for node in self.nodes:
            color = "#59a14f" if node.backend == "cpp" else "#4e79a7"
            label = f"{node.id}: {node.operation}\\n{node.backend} island {node.island}"
            lines.append(f'  n{node.id} [label="{label}", style=filled, fillcolor="{color}"];')
            lines.extend(f"  n{child} -> n{node.id};" for child in node.children)
        lines.append("}")
        return "\n".join(lines)

    def __str__(self) -> str:
        rows = ["JAX-flat lowering plan:"]
        for node in self.nodes:
            reason = f" ({node.reason})" if node.reason else ""
            rows.append(f"  {node.id:>3}  {node.backend:<3} island={node.island:<2} {node.operation} <- {list(node.children)}{reason}")
        return "\n".join(rows)

    def format(self, format: str = "text") -> str:
        if format == "text":
            return str(self)
        if format == "dot":
            return self.to_dot()
        if format == "json":
            return json.dumps(self.to_dict(), indent=2)
        raise ValueError("format must be 'text', 'json', or 'dot'")


@dataclass(frozen=True)
class CppFlatRuntime:
    """Allocation-conscious native tick-path runtime for supported flat formulas.

    The C++ core owns flattened node specs and preallocates per-node scratch in
    ``init_state``. ``run_batch`` mirrors the JAX-flat runtime API for batch
    execution. ``tick_into`` remains the lowest-allocation row hot path for
    callers that can provide reusable state and output buffers.
    """

    program: StreamingProgram
    core: Any
    supported_ops: tuple[str, ...]
    native_plan: NativeExecutionPlan

    def init_state(self, n_instruments: int):
        return self.core.init_state(n_instruments)

    def tick(self, state, *input_rows):
        return self.core.tick(state, *input_rows)

    def tick_into(self, state, out, *input_rows) -> None:
        self.core.tick_into(state, out, *input_rows)

    def run_batch(self, inputs, states=None, out=None):
        inputs = _normalize_batch_inputs_for_program(self.program, inputs)
        if not inputs:
            raise ValueError("run_batch requires at least one input array")
        n_steps, n_instruments = inputs[0].shape
        for arr in inputs[1:]:
            if arr.shape != (n_steps, n_instruments):
                raise ValueError("All inputs must share aligned shape (time, n_instruments)")
        state = states or self.init_state(n_instruments)
        np_inputs = tuple(np.asarray(arr, dtype=np.float64) for arr in inputs)
        if out is None:
            raw = self.core.run_batch(state, *np_inputs)
            return state, _reshape_cpp_batch_output(self.program, raw, n_steps, n_instruments)
        self.core.run_batch_into(state, out, *np_inputs)
        return state, out

    def run_batch_into(self, state, out, inputs):
        inputs = _normalize_batch_inputs_for_program(self.program, inputs)
        np_inputs = tuple(np.asarray(arr, dtype=np.float64) for arr in inputs)
        self.core.run_batch_into(state, out, *np_inputs)
        return state, out

    def inspect_native_plan(self) -> dict[str, Any]:
        """Return a serialization-friendly plan diagnostic, off the hot path."""
        return self.native_plan.diagnostic()


class BroadcastMode(str, Enum):
    SCALAR = "scalar"
    ELEMENTWISE = "elementwise"
    MATRIX = "matrix"
    REDUCTION = "reduction"
    GROUPED = "grouped"


@dataclass(frozen=True)
class NativeValueType:
    shape: str
    width: int
    dtype: str
    broadcast: BroadcastMode


@dataclass(frozen=True)
class NativePlanNode:
    """A fully resolved node in the portable native execution plan."""

    node_id: int
    opcode: str
    children: tuple[int, ...]
    value_type: NativeValueType
    state_index: int
    live_from: int
    live_until: int
    stateful: bool
    pure: bool
    grouping: tuple[int, ...]
    legacy_spec: tuple[Any, ...]


@dataclass(frozen=True)
class NativeExecutionPlan:
    """Typed lowering boundary between ``StreamingProgram`` and C++.

    The tuple ABI remains as the reference evaluator input during the staged
    migration.  All planning and diagnostics consume this typed form first.
    """

    nodes: tuple[NativePlanNode, ...]
    output_id: int
    state_count: int
    dtype: str
    source_node_count: int
    optimizations: tuple[tuple[str, int], ...]

    def reference_specs(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(node.legacy_spec for node in self.nodes)

    def diagnostic(self) -> dict[str, Any]:
        return {
            "version": 1,
            "dtype": self.dtype,
            "node_count": len(self.nodes),
            "source_node_count": self.source_node_count,
            "optimizations": dict(self.optimizations),
            "state_count": self.state_count,
            "output_id": self.output_id,
            "nodes": tuple(
                {
                    "id": n.node_id,
                    "opcode": n.opcode,
                    "children": n.children,
                    "shape": n.value_type.shape,
                    "width": n.value_type.width,
                    "broadcast": n.value_type.broadcast.value,
                    "state_index": n.state_index,
                    "liveness": (n.live_from, n.live_until),
                    "stateful": n.stateful,
                    "pure": n.pure,
                    "grouping": n.grouping,
                }
                for n in self.nodes
            ),
        }


def compile_formula(formula: str | Expr, dsl_registry: DSLFunctionRegistry | None = None) -> CppFlatRuntime:
    """Compile a supported jax_flat formula to the native C++ tick runtime.

    This backend intentionally covers the allocation-sensitive scalar/vector tick
    subset first. Unsupported operators raise a clear ``NotImplementedError`` so
    callers can keep using the JAX-flat runtime for full DSL coverage.
    """
    from trading_dsl_engine.jax_flat import _cpp_flat

    runtime = compile_formula_jax(formula, dsl_registry=dsl_registry, cpp=False)
    plan, supported = lower_native_plan(runtime.program)
    core = _cpp_flat.make_runtime(plan.reference_specs(), plan.output_id, plan.state_count)
    return CppFlatRuntime(
        program=runtime.program,
        core=core,
        supported_ops=tuple(sorted(set(supported))),
        native_plan=plan,
    )


def lower_native_plan(
    program: StreamingProgram, *, dtype: str = "float64", optimize: bool = True
) -> tuple[NativeExecutionPlan, list[str]]:
    """Lower a program to typed native IR with resolved liveness metadata.

    This intentionally retains the old tuple evaluator as an equivalence
    oracle. Subsequent native stages can consume the typed fields without
    changing streaming state transitions.
    """
    if dtype not in {"float32", "float64"}:
        raise ValueError(f"unsupported native dtype: {dtype}")
    specs, supported, state_count = _cpp_node_specs(program)
    source_node_count = len(specs)
    if optimize:
        specs, output_id, source_ids, optimization_counts = _optimize_native_specs(
            specs, program.outputs[0], program
        )
    else:
        output_id = program.outputs[0]
        source_ids = tuple(range(len(specs)))
        optimization_counts = ()
    last_use = list(range(len(specs)))
    for parent, spec in enumerate(specs):
        for child in spec[1]:
            last_use[child] = max(last_use[child], parent)
    last_use[output_id] = len(specs)
    nodes = []
    reduction_names = {"mean", "xstd", "xs_rank", "xs_sort", "xs_norm"}
    for node_id, (source_id, spec) in enumerate(zip(source_ids, specs, strict=True)):
        dag_node = program.nodes[source_id]
        opcode, children = spec[0], tuple(spec[1])
        width = int(spec[7])
        shape = dag_node.op.output_kind
        if opcode == "group":
            broadcast = BroadcastMode.GROUPED
        elif opcode in reduction_names:
            broadcast = BroadcastMode.REDUCTION
        elif shape == "scalar":
            broadcast = BroadcastMode.SCALAR
        elif shape == "matrix":
            broadcast = BroadcastMode.MATRIX
        else:
            broadcast = BroadcastMode.ELEMENTWISE
        nodes.append(
            NativePlanNode(
                node_id=node_id,
                opcode=opcode,
                children=children,
                value_type=NativeValueType(shape, width, dtype, broadcast),
                state_index=int(spec[3]),
                live_from=node_id,
                live_until=last_use[node_id],
                stateful=dag_node.op.is_stateful,
                pure=not dag_node.op.is_stateful,
                grouping=tuple(spec[8]) if opcode == "group" else (),
                legacy_spec=spec,
            )
        )
    return NativeExecutionPlan(
        tuple(nodes), output_id, state_count, dtype, source_node_count, optimization_counts
    ), supported


def _optimize_native_specs(specs, output_id: int, program: StreamingProgram):
    """Apply semantics-safe DCE, pure CSE, literal folding, and cache aliases."""
    reachable: set[int] = set()
    stack = [output_id]
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        stack.extend(specs[node_id][1])

    remap: dict[int, int] = {}
    key_to_id: dict[Any, int] = {}
    optimized: list[tuple[Any, ...]] = []
    source_ids: list[int] = []
    folded = aliases = common = stateful_common = 0
    literal_values: dict[int, float] = {}

    def replace_spec(spec, **changes):
        values = list(spec)
        indexes = {"opcode": 0, "children": 1, "literal": 4, "width": 7}
        for name, value in changes.items():
            values[indexes[name]] = value
        return tuple(values)

    for old_id, spec in enumerate(specs):
        if old_id not in reachable:
            continue
        children = tuple(remap[child] for child in spec[1])
        spec = replace_spec(spec, children=children)
        opcode = spec[0]
        if opcode == "cache" and len(children) == 1:
            remap[old_id] = children[0]
            aliases += 1
            continue
        folded_value = _fold_native_literal(opcode, children, literal_values)
        if folded_value is not None:
            spec = replace_spec(spec, opcode="literal", children=(), literal=folded_value, width=1)
            opcode = "literal"
            children = ()
            folded += 1
        is_stateful = program.nodes[old_id].op.is_stateful
        is_shareable = opcode != "group"
        # Identical deterministic state transitions over identical children can
        # share one state slot: their on_data/emit sequences are indistinguishable.
        # Group nodes remain unique because they own keyed routing domains.
        key_spec = list(spec)
        if is_stateful:
            key_spec[3] = -1  # physical state slot is assigned after planning
        # repr canonicalizes nested tuple parameters and gives NaNs a stable key.
        key = repr(tuple(key_spec)) if is_shareable else None
        if key is not None and key in key_to_id:
            remap[old_id] = key_to_id[key]
            common += 1
            stateful_common += int(is_stateful)
            continue
        new_id = len(optimized)
        remap[old_id] = new_id
        optimized.append(spec)
        source_ids.append(old_id)
        if opcode == "literal":
            literal_values[new_id] = float(spec[4])
        if key is not None:
            key_to_id[key] = new_id
    new_output = remap[output_id]
    live_after_folding: set[int] = set()
    stack = [new_output]
    while stack:
        node_id = stack.pop()
        if node_id in live_after_folding:
            continue
        live_after_folding.add(node_id)
        stack.extend(optimized[node_id][1])
    compact_remap = {old: new for new, old in enumerate(sorted(live_after_folding))}
    compact_specs = tuple(
        replace_spec(optimized[old], children=tuple(compact_remap[c] for c in optimized[old][1]))
        for old in sorted(live_after_folding)
    )
    compact_sources = tuple(source_ids[old] for old in sorted(live_after_folding))
    dead = len(specs) - len(reachable) + len(optimized) - len(compact_specs)
    counts = (
        ("dead_nodes", dead),
        ("constant_folds", folded),
        ("aliases_removed", aliases),
        ("common_subexpressions", common),
        ("stateful_common_subexpressions", stateful_common),
    )
    return compact_specs, compact_remap[new_output], compact_sources, counts


def _fold_native_literal(opcode: str, children: tuple[int, ...], values: dict[int, float]):
    if not children or any(child not in values for child in children):
        return None
    args = tuple(values[child] for child in children)
    binary = {
        "add": np.add,
        "sub": np.subtract,
        "mul": np.multiply,
        "div": np.divide,
        "pow": np.power,
    }
    unary = {"abs": np.abs, "exp": np.exp, "ln": np.log, "sign": np.sign}
    fn = binary.get(opcode) if len(args) == 2 else unary.get(opcode)
    if fn is None:
        return None
    with np.errstate(all="ignore"):
        return float(fn(*args))


def _reshape_cpp_batch_output(program: StreamingProgram, raw, n_steps: int, n_instruments: int):
    root = program.nodes[program.outputs[0]].op
    arr = np.asarray(raw)
    if root.output_kind == "matrix":
        width = int(root.output_width) if root.output_width is not None else n_instruments
        if isinstance(root, NaryOp) and root.cpp_name == "einsum":
            subscripts = root.cpp_str_param
            output = subscripts.split("->", 1)[1] if "->" in subscripts else ""
            if "i" not in output and len(output) == 2:
                input_terms = subscripts.split("->", 1)[0].split(",")
                row_width = next(
                    (program.nodes[cid].op.output_width for term, cid in zip(input_terms, program.nodes[program.outputs[0]].child_ids) if len(term) == 2 and term[1] == output[0]),
                    None,
                )
                if row_width is not None:
                    return arr.reshape((n_steps, int(row_width), width))
        return arr.reshape((n_steps, n_instruments, width))
    if root.output_kind == "scalar" and arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _normalize_batch_inputs_for_program(program: StreamingProgram, inputs):
    if isinstance(inputs, dict):
        missing = [name for name in program.input_names if name not in inputs]
        if missing:
            raise ValueError(f"Missing jax_flat C++ input(s): {missing}")
        inputs = tuple(inputs[name] for name in program.input_names)
    else:
        inputs = tuple(inputs)
    if len(inputs) != len(program.input_names):
        raise ValueError(f"expected {len(program.input_names)} input array(s), got {len(inputs)}")
    for name, arr in zip(program.input_names, inputs):
        if np.asarray(arr).ndim != 2:
            raise ValueError(f"Expected 2D input for '{name}', got shape {np.asarray(arr).shape}")
    return inputs


def _cpp_spec_tuple(
    name, children=(), input_index=-1, state_index=-1, literal=0.0,
    param=0.0, int_param=0, width=1, feature_widths=(), inner_specs=(),
    inner_output_id=-1, str_param="",
):
    """Canonical Python/native node-spec ABI constructor."""
    return (
        name, tuple(children), input_index, state_index, float(literal),
        float(param), int(int_param), int(1 if width is None else width),
        tuple(int(w) for w in feature_widths), tuple(inner_specs),
        int(inner_output_id), str(str_param),
    )


def _cpp_node_specs(program: StreamingProgram):
    if len(program.outputs) != 1:
        raise NotImplementedError("C++ jax_flat currently supports exactly one output")
    specs = []
    supported = []
    state_count = program.state_layout.total_leaves

    for idx, node in enumerate(program.nodes):
        op = node.op
        field = program.state_layout.node_fields[idx]
        state_index = field.index
        width = op.output_width or 1
        if isinstance(op, InputOp):
            specs.append(_cpp_spec_tuple("input", input_index=op.input_index, width=1))
            supported.append("input")
            continue
        if isinstance(op, LiteralOp):
            specs.append(_cpp_spec_tuple("literal", literal=op.value, width=1))
            supported.append("literal")
            continue
        if isinstance(op, GroupByOp):
            group_spec, group_supported = _cpp_groupby_spec(op, node.child_ids, state_index, _cpp_spec_tuple)
            specs.append(group_spec)
            supported.extend(group_supported)
            continue
        if isinstance(op, CacheOp):
            specs.append(_cpp_spec_tuple("cache", node.child_ids, width=width))
            supported.append("cache")
            continue
        if isinstance(op, NaryOp) and op.cpp_name is not None:
            cpp_width = width
            if op.cpp_name == "outer":
                cpp_width = 0
            if op.cpp_name == "einsum" and op.output_width is None:
                cpp_width = 0
            if op.cpp_name == "get_beta":
                beta_child = program.nodes[node.child_ids[0]].op
                if isinstance(beta_child, RidgeOp):
                    cpp_width = sum(beta_child.feature_widths)
                elif isinstance(beta_child, InstrumentBasisMeanOp):
                    cpp_width = beta_child.feature_width
                else:
                    raise NotImplementedError("C++ jax_flat get_beta expects direct Ridge or InstrumentBasisMean child")
            feature_widths = ()
            if op.cpp_name == "einsum":
                feature_widths = tuple(program.nodes[cid].op.output_width or 1 for cid in node.child_ids)
            specs.append(_cpp_spec_tuple(op.cpp_name, node.child_ids, int_param=op.cpp_int_param, param=op.cpp_param, width=cpp_width, feature_widths=feature_widths, str_param=getattr(op, "cpp_str_param", "")))
            supported.append(op.cpp_name)
            continue
        if isinstance(op, CumsumOp):
            specs.append(_cpp_spec_tuple("cumsum", node.child_ids, state_index=state_index, width=width))
            supported.append("cumsum")
            continue
        if isinstance(op, EwmOp) and op.span is not None and op.ignore_na and not op.adjust:
            min_periods = -1 if op.min_periods is None else int(round(float(op.min_periods)))
            ewm_flags = (1 if op.ignore_na else 0) | (2 if op.adjust else 0)
            specs.append(_cpp_spec_tuple("ewm", node.child_ids, state_index=state_index, param=op.span, int_param=(min_periods + 1) * 4 + ewm_flags, width=width))
            supported.append("ewm")
            continue
        if isinstance(op, RollingMeanOp):
            specs.append(_cpp_spec_tuple("roll_mean", node.child_ids, state_index=state_index, param=op.min_periods, int_param=op.lookback, width=width))
            supported.append("roll_mean")
            continue
        if isinstance(op, FFillOp) and not op.dynamic_limit:
            limit = -1 if op.limit is None else op.limit
            specs.append(_cpp_spec_tuple("ffill", node.child_ids, state_index=state_index, int_param=limit, width=width))
            supported.append("ffill")
            continue
        if isinstance(op, ShiftOp):
            specs.append(_cpp_spec_tuple("shift", node.child_ids, state_index=state_index, int_param=op.max_size, width=width))
            supported.append("shift")
            continue
        if isinstance(op, RbfBasisOp):
            specs.append(_cpp_spec_tuple("rbf_basis", node.child_ids, int_param=op.n_basis, width=op.n_basis))
            supported.append("rbf_basis")
            continue
        if isinstance(op, FutureRbfBasisSumOp):
            specs.append(_cpp_spec_tuple("future_rbf_basis_sum", node.child_ids, int_param=op.n_basis, param=op.n_steps, width=op.n_basis))
            supported.append("future_rbf_basis_sum")
            continue
        if isinstance(op, InstrumentBasisMeanOp):
            specs.append(
                _cpp_spec_tuple(
                    "instrument_basis_mean",
                    node.child_ids,
                    state_index=state_index,
                    int_param=1 if op.has_weights else 0,
                    width=1,
                    feature_widths=(op.feature_width,),
                )
            )
            supported.append("instrument_basis_mean")
            continue
        if isinstance(op, RidgeOp):
            if state_index < 0:
                state_index = state_count
                state_count += 1
            specs.append(_cpp_spec_tuple("ridge", node.child_ids, state_index=state_index, int_param=1 if op.nonneg else 0, width=1, feature_widths=op.feature_widths))
            supported.append("ridge")
            continue
        raise NotImplementedError(f"C++ jax_flat does not yet support node {idx}: {type(op).__name__}")
    return tuple(specs), supported, state_count



def _cpp_groupby_spec(op: GroupByOp, child_ids: tuple[int, ...], state_index: int, spec_tuple):
    inner = op.inner_op
    if not isinstance(inner, InnerGraphOp):
        raise NotImplementedError("C++ jax_flat groupby expects lowered inner graph")
    if inner.n_inputs != 1:
        raise NotImplementedError("C++ jax_flat groupby currently supports RHS graphs over self_ only")
    inner_specs, supported = _cpp_inner_node_specs(inner, spec_tuple)
    return (
        spec_tuple(
            "group",
            child_ids,
            state_index=state_index,
            param=op.capacity,
            int_param=op.n_keys,
            width=op.output_width or 1,
            feature_widths=_encode_universe_groups(op.universe_groups),
            inner_specs=inner_specs,
            inner_output_id=inner.output_id,
        ),
        ["group", *supported],
    )


def _encode_universe_groups(groups: tuple[tuple[int, ...], ...] | None) -> tuple[int, ...]:
    if groups is None:
        return ()
    encoded: list[int] = [len(groups)]
    for group in groups:
        encoded.append(len(group))
        encoded.extend(int(col) for col in group)
    return tuple(encoded)


def _cpp_inner_node_specs(inner: InnerGraphOp, spec_tuple):
    specs = []
    supported = []
    for idx, node in enumerate(inner.nodes):
        op = node.op
        field = inner.state_layout.node_fields[idx]
        state_index = field.index
        width = op.output_width or 1
        if isinstance(op, InputOp):
            specs.append(spec_tuple("input", input_index=op.input_index, width=1))
            supported.append("inner_input")
            continue
        if isinstance(op, LiteralOp):
            specs.append(spec_tuple("literal", literal=op.value, width=1))
            supported.append("inner_literal")
            continue
        if isinstance(op, CacheOp):
            specs.append(spec_tuple("cache", node.child_ids, width=width))
            supported.append("cache")
            continue
        if isinstance(op, NaryOp) and op.cpp_name is not None:
            cpp_width = 0 if op.cpp_name in {"outer", "einsum"} and op.output_width is None else width
            if cpp_width != 1:
                raise NotImplementedError(f"C++ jax_flat groupby inner op {op.cpp_name!r} currently supports vector/scalar width 1")
            specs.append(spec_tuple(op.cpp_name, node.child_ids, param=op.cpp_param, int_param=op.cpp_int_param, width=cpp_width, str_param=getattr(op, "cpp_str_param", "")))
            supported.append(f"inner_{op.cpp_name}")
            continue
        if isinstance(op, CumsumOp):
            specs.append(spec_tuple("cumsum", node.child_ids, state_index=state_index, width=width))
            supported.append("inner_cumsum")
            continue
        if isinstance(op, EwmOp) and op.span is not None and op.ignore_na and not op.adjust:
            min_periods = -1 if op.min_periods is None else int(round(float(op.min_periods)))
            ewm_flags = (1 if op.ignore_na else 0) | (2 if op.adjust else 0)
            specs.append(spec_tuple("ewm", node.child_ids, state_index=state_index, param=op.span, int_param=(min_periods + 1) * 4 + ewm_flags, width=width))
            supported.append("inner_ewm")
            continue
        if isinstance(op, FFillOp) and not op.dynamic_limit:
            limit = -1 if op.limit is None else op.limit
            specs.append(spec_tuple("ffill", node.child_ids, state_index=state_index, int_param=limit, width=width))
            supported.append("inner_ffill")
            continue
        raise NotImplementedError(f"C++ jax_flat groupby inner node {idx} unsupported: {type(op).__name__}")
    return tuple(specs), supported


# --- JAX/C++ island handling for hybrid jax_flat batch execution: start ---

def _try_cpp_hybrid_batch(
    runtime,
    inputs,
    accelerator_cache: dict[tuple[Any, ...], Any],
    warn_callback: Callable[[Any, str], None],
    out_path=False,
):
    """Run full-native or staged native/JAX/native batch execution when possible."""
    if os.getenv("TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL", "0") == "1":
        return None
    full = _try_cpp_full_batch(
        runtime, inputs, accelerator_cache, warn_callback, emit_warning=False, out_path=out_path
    )
    if full is not None:
        return full
    if out_path:
        _try_cpp_full_batch(runtime, inputs, accelerator_cache, warn_callback, emit_warning=True)
        return None

    n_steps, n_instruments = inputs[0].shape
    candidates = _cpp_hybrid_candidates(runtime.program, n_steps, n_instruments)
    if not candidates:
        if any(isinstance(node.op, GroupByOp) for node in runtime.program.nodes):
            _try_cpp_full_batch(runtime, inputs, accelerator_cache, warn_callback, emit_warning=True)
        return None
    try:
        from trading_dsl_engine.jax_flat import _cpp_flat
    except Exception as exc:
        warn_callback(runtime, f"C++ jax_flat hybrid accelerator unavailable ({type(exc).__name__}: {exc}); falling back to JAX-flat")
        return None

    extra_inputs = []
    candidate_programs = []
    for node_id in candidates:
        subprogram, source_input_indices = _subprogram_for_node(runtime.program, node_id)
        try:
            plan, _ = lower_native_plan(subprogram)
        except NotImplementedError:
            continue
        node_specs = plan.reference_specs()
        key = (node_specs, plan.output_id, plan.state_count, n_instruments)
        core = accelerator_cache.get(key)
        if core is None:
            core = _cpp_flat.make_runtime(node_specs, plan.output_id, plan.state_count)
            accelerator_cache[key] = core
        state = core.init_state(n_instruments)
        source_inputs = tuple(np.asarray(inputs[i], dtype=np.float64) for i in source_input_indices)
        raw = core.run_batch(state, *source_inputs)
        extra_inputs.append(np.asarray(_reshape_cpp_batch_output(subprogram, raw, n_steps, n_instruments)))
        candidate_programs.append(node_id)
    if not extra_inputs:
        _try_cpp_full_batch(runtime, inputs, accelerator_cache, warn_callback, emit_warning=True)
        return None

    staged = _try_cpp_staged_output_batch(runtime, inputs, tuple(candidate_programs), tuple(extra_inputs), accelerator_cache)
    if staged is not None:
        return staged

    residual = _program_with_cpp_inputs(runtime.program, tuple(candidate_programs))
    residual_runtime = replace(runtime, program=residual, cpp=False)
    residual_inputs = tuple(inputs) + tuple(jnp.asarray(arr) for arr in extra_inputs)
    return residual_runtime._run_batch_once(residual_inputs, None, False)


def _try_cpp_full_batch(
    runtime,
    inputs,
    accelerator_cache: dict[tuple[Any, ...], Any],
    warn_callback: Callable[[Any, str], None],
    *,
    emit_warning: bool,
    out_path=False,
):
    try:
        from trading_dsl_engine.jax_flat import _cpp_flat
        plan, _ = lower_native_plan(runtime.program)
    except Exception as exc:
        if emit_warning:
            warn_callback(runtime, f"C++ jax_flat accelerator unsupported for this formula: {exc}; falling back to JAX-flat")
        return None

    n_steps, n_instruments = inputs[0].shape
    out = _cpp_batch_output(runtime.program, out_path, n_steps, n_instruments) if out_path else None
    node_specs = plan.reference_specs()
    key = (node_specs, plan.output_id, plan.state_count, n_instruments)
    core = accelerator_cache.get(key)
    if core is None:
        core = _cpp_flat.make_runtime(node_specs, plan.output_id, plan.state_count)
        accelerator_cache[key] = core
    state = core.init_state(n_instruments)
    cpp_inputs = tuple(np.asarray(arr, dtype=np.float64) for arr in inputs)
    if out is not None:
        core.run_batch_into(state, out, *cpp_inputs)
        return state, out
    raw = core.run_batch(state, *cpp_inputs)
    return state, _reshape_cpp_batch_output(runtime.program, raw, n_steps, n_instruments)


def _cpp_batch_output(program: StreamingProgram, out_path, n_steps: int, n_instruments: int):
    root = program.nodes[program.outputs[0]].op
    if root.output_kind == "scalar":
        shape = (n_steps,)
    elif root.output_kind == "vector":
        shape = (n_steps, n_instruments)
    elif root.output_kind == "matrix" and root.output_width is not None:
        shape = (n_steps, n_instruments, int(root.output_width))
    else:
        return None
    if out_path is True:
        fd, out_path = tempfile.mkstemp(prefix="trading_dsl_engine_cpp_out_", suffix=".memmap")
        os.close(fd)
    if not isinstance(out_path, (str, os.PathLike)):
        return None
    return np.memmap(os.fspath(out_path), mode="w+", dtype=np.float64, shape=shape)


def _try_cpp_staged_output_batch(runtime, inputs, upstream_node_ids: tuple[int, ...], upstream_values: tuple[np.ndarray, ...], accelerator_cache: dict[tuple[Any, ...], Any]):
    if len(runtime.program.outputs) != 1:
        return None
    root_id = runtime.program.outputs[0]
    try:
        subprogram, frontier_ids, input_sources = _subprogram_for_node_with_frontier(runtime.program, root_id)
    except NotImplementedError:
        return None
    if not frontier_ids:
        return None
    try:
        from trading_dsl_engine.jax_flat import _cpp_flat
        plan, _ = lower_native_plan(subprogram)
    except Exception:
        return None

    residual = _program_with_cpp_inputs(runtime.program, upstream_node_ids)
    residual_inputs = tuple(inputs) + tuple(jnp.asarray(arr) for arr in upstream_values)
    # The flat runtime root ABI is singular. Evaluate multiple compile-time
    # frontiers as independent batch roots rather than coercing them through a
    # stacked root whose leading dimension is ambiguous with time.
    frontier_values = []
    for frontier_id in frontier_ids:
        frontier_program = replace(residual, outputs=(frontier_id,))
        residual_runtime = replace(runtime, program=frontier_program, cpp=False)
        _, frontier_out = residual_runtime._run_batch_once(residual_inputs, None, False)
        frontier_values.append(np.asarray(frontier_out))
    frontier_values = tuple(frontier_values)

    n_steps, n_instruments = inputs[0].shape
    node_specs = plan.reference_specs()
    key = (node_specs, plan.output_id, plan.state_count, n_instruments)
    core = accelerator_cache.get(key)
    if core is None:
        core = _cpp_flat.make_runtime(node_specs, plan.output_id, plan.state_count)
        accelerator_cache[key] = core
    state = core.init_state(n_instruments)
    frontier_by_id = dict(zip(frontier_ids, frontier_values, strict=True))
    cpp_inputs = tuple(
        np.asarray(inputs[idx] if kind == "input" else frontier_by_id[idx], dtype=np.float64)
        for kind, idx in input_sources
    )
    raw = core.run_batch(state, *cpp_inputs)
    return state, _reshape_cpp_batch_output(subprogram, raw, n_steps, n_instruments)


def _is_cpp_boundary_op(op: Op) -> bool:
    return (
        isinstance(
            op,
            (
                InputOp,
                LiteralOp,
                CacheOp,
                GroupByOp,
                CumsumOp,
                EwmOp,
                FFillOp,
                RollingMeanOp,
                ShiftOp,
                RbfBasisOp,
                FutureRbfBasisSumOp,
                InstrumentBasisMeanOp,
                RidgeOp,
            ),
        )
        or (isinstance(op, NaryOp) and op.cpp_name is not None)
    )


def _subprogram_for_node_with_frontier(program: StreamingProgram, node_id: int) -> tuple[StreamingProgram, tuple[int, ...], tuple[tuple[str, int], ...]]:
    frontier: list[int] = []
    input_remap: dict[int, int] = {}
    remap: dict[int, int] = {}
    nodes: list[DagNode] = []

    def compact_input_index(input_index: int) -> int:
        if input_index not in input_remap:
            input_remap[input_index] = len(input_remap)
        return input_remap[input_index]

    def build(old_id: int, *, force_frontier: bool = False) -> int:
        if old_id in remap:
            return remap[old_id]
        op = program.nodes[old_id].op
        if force_frontier or not _is_cpp_boundary_op(op):
            if old_id not in frontier:
                frontier.append(old_id)
            input_index = compact_input_index(len(program.input_names) + frontier.index(old_id))
            remap[old_id] = len(nodes)
            nodes.append(DagNode(InputOp(input_index, output_kind=op.output_kind, output_width=op.output_width), ()))
            return remap[old_id]
        if isinstance(op, InputOp):
            remap[old_id] = len(nodes)
            nodes.append(DagNode(InputOp(compact_input_index(op.input_index), output_kind=op.output_kind, output_width=op.output_width), ()))
            return remap[old_id]
        child_ids = []
        for child_id in program.nodes[old_id].child_ids:
            child_op = program.nodes[child_id].op
            child_ids.append(build(child_id, force_frontier=not _is_cpp_boundary_op(child_op)))
        remap[old_id] = len(nodes)
        nodes.append(DagNode(op, tuple(child_ids)))
        return remap[old_id]

    root_op = program.nodes[node_id].op
    if not _is_cpp_boundary_op(root_op):
        raise NotImplementedError("root is not C++ lowerable")
    output = build(node_id)
    subprogram = StreamingProgram(
        nodes=tuple(nodes),
        outputs=(output,),
        input_names=tuple(
            program.input_names[i] if i < len(program.input_names) else f"__jax_frontier_{i - len(program.input_names)}"
            for i, _ in sorted(input_remap.items(), key=lambda item: item[1])
        ),
        state_layout=_state_layout_for_nodes(tuple(nodes)),
        metadata=None,
        cache_nodes=(),
    )
    _cpp_node_specs(subprogram)
    input_sources = tuple(
        ("input", i) if i < len(program.input_names) else ("frontier", frontier[i - len(program.input_names)])
        for i, _ in sorted(input_remap.items(), key=lambda item: item[1])
    )
    return subprogram, tuple(frontier), input_sources


def inspect_hybrid_partition(
    program: StreamingProgram, n_rows: int, n_instruments: int, *, itemsize: int = 8
) -> dict[str, Any]:
    """Explain cost-aware native-island decisions without entering a hot path."""
    candidates = _cpp_hybrid_candidates(program, n_rows, n_instruments, apply_cost=False)
    decisions = []
    for node_id in candidates:
        node_count = len(_ancestor_ids(program, node_id))
        work = node_count * n_rows * n_instruments
        frontier_bytes = n_rows * n_instruments * itemsize
        # Descriptor dispatch is deliberately expressed in element-equivalent
        # work units; measured portable-runtime launches are small because the
        # extension and state are already cached.
        launch_cost = 64
        estimated_cost = 2 * (frontier_bytes // max(itemsize, 1)) + launch_cost
        decisions.append(
            {
                "node_id": node_id,
                "node_count": node_count,
                "estimated_work": work,
                "frontier_bytes": frontier_bytes,
                "conversion_copy": True,
                "runtime_launches": 1,
                "estimated_transfer_cost": estimated_cost,
                "accelerate": work >= estimated_cost,
            }
        )
    return {"version": 1, "rows": n_rows, "instruments": n_instruments, "candidates": tuple(decisions)}


def _cpp_hybrid_candidates(
    program: StreamingProgram,
    n_rows: int = 0,
    n_instruments: int = 0,
    *,
    apply_cost: bool = True,
) -> tuple[int, ...]:
    selected: list[int] = []
    covered: set[int] = set()
    for node_id in range(len(program.nodes) - 1, -1, -1):
        if node_id in covered or node_id in program.outputs:
            continue
        op = program.nodes[node_id].op
        ancestors = _ancestor_ids(program, node_id)
        is_projection = isinstance(op, NaryOp) and op.cpp_name in {"get_beta", "get_preds"}
        is_supported_stateless = (
            isinstance(op, NaryOp)
            and op.cpp_name is not None
            and any(program.nodes[cid].op.is_stateful for cid in ancestors if cid != node_id)
        )
        if not (is_projection or is_supported_stateless or isinstance(op, (GroupByOp, CumsumOp, EwmOp, FFillOp, RollingMeanOp, ShiftOp))):
            continue
        if ancestors & covered:
            continue
        try:
            _cpp_node_specs(_subprogram_for_node(program, node_id)[0])
        except Exception:
            continue
        selected.append(node_id)
        covered.update(ancestors)
    selected = sorted(selected)
    if apply_cost and n_rows > 0 and n_instruments > 0:
        diagnostic = inspect_hybrid_partition(program, n_rows, n_instruments)
        accepted = {item["node_id"] for item in diagnostic["candidates"] if item["accelerate"]}
        selected = [node_id for node_id in selected if node_id in accepted]
    return tuple(selected)


def _ancestor_ids(program: StreamingProgram, node_id: int) -> set[int]:
    seen: set[int] = set()
    stack = [node_id]
    while stack:
        idx = stack.pop()
        if idx in seen:
            continue
        seen.add(idx)
        stack.extend(program.nodes[idx].child_ids)
    return seen


def _state_layout_for_nodes(nodes: tuple[DagNode, ...]) -> StateLayout:
    fields = []
    total = 0
    for node in nodes:
        if node.op.is_stateful:
            fields.append(StateFieldRef(total))
            total += 1
        else:
            fields.append(StateFieldRef(-1))
    return StateLayout(tuple(fields), total)


def _subprogram_for_node(program: StreamingProgram, node_id: int) -> tuple[StreamingProgram, tuple[int, ...]]:
    ids = sorted(_ancestor_ids(program, node_id))
    remap = {old: new for new, old in enumerate(ids)}
    input_remap: dict[int, int] = {}
    nodes = []
    for old in ids:
        op = program.nodes[old].op
        if isinstance(op, InputOp):
            if op.input_index not in input_remap:
                input_remap[op.input_index] = len(input_remap)
            op = InputOp(input_remap[op.input_index], output_kind=op.output_kind, output_width=op.output_width)
        nodes.append(DagNode(op, tuple(remap[c] for c in program.nodes[old].child_ids)))
    compact_nodes = tuple(nodes)
    input_names = tuple(program.input_names[i] for i, _ in sorted(input_remap.items(), key=lambda item: item[1]))
    source_input_indices = tuple(i for i, _ in sorted(input_remap.items(), key=lambda item: item[1]))
    return StreamingProgram(nodes=compact_nodes, outputs=(remap[node_id],), input_names=input_names, state_layout=_state_layout_for_nodes(compact_nodes), metadata=None, cache_nodes=()), source_input_indices


def _program_with_cpp_inputs(program: StreamingProgram, node_ids: tuple[int, ...]) -> StreamingProgram:
    replacements = {node_id: len(program.input_names) + i for i, node_id in enumerate(node_ids)}
    nodes = []
    for idx, node in enumerate(program.nodes):
        if idx in replacements:
            op = node.op
            nodes.append(DagNode(InputOp(replacements[idx], output_kind=op.output_kind, output_width=op.output_width), ()))
        else:
            nodes.append(node)
    return replace(program, nodes=tuple(nodes), input_names=program.input_names + tuple(f"__cpp_subgraph_{i}" for i in node_ids), state_layout=_state_layout_for_nodes(tuple(nodes)))

# --- JAX/C++ island handling for hybrid jax_flat batch execution: end ---

def _require_vector_cpp_op(op: Op, name: str) -> None:
    if op.output_kind != "vector" or op.output_width not in (None, 1):
        raise NotImplementedError(f"C++ jax_flat op {name!r} currently supports vector outputs only")


def _operation_name(op: Op) -> str:
    if isinstance(op, NaryOp):
        return op.cpp_name or op.diagnostic_name or getattr(op.fn, "__name__", "stateless")
    if isinstance(op, RollingOp):
        return getattr(op.fn, "__name__", "rolling")
    names = {
        InputOp: "input", LiteralOp: "literal", CacheOp: "cache", CumsumOp: "cumsum",
        EwmOp: "ewm", RollingMeanOp: "roll_mean", ShiftOp: "shift",
        BufferShiftOp: "buffer", FFillOp: "ffill", RbfBasisOp: "rbf_basis",
        FutureRbfBasisSumOp: "future_rbf_basis_sum", InstrumentBasisMeanOp: "InstrumentBasisMean",
        RidgeOp: "Ridge", GroupByOp: "groupby",
    }
    return names.get(type(op), type(op).__name__)


def _synthetic_node_program(program: StreamingProgram, node_id: int) -> StreamingProgram:
    """Build a one-op program whose children are typed JAX frontier inputs."""
    node = program.nodes[node_id]
    inputs = tuple(
        DagNode(InputOp(index, output_kind=program.nodes[child_id].op.output_kind, output_width=program.nodes[child_id].op.output_width), ())
        for index, child_id in enumerate(node.child_ids)
    )
    nodes = inputs + (DagNode(node.op, tuple(range(len(inputs)))),)
    return StreamingProgram(
        nodes=nodes,
        outputs=(len(nodes) - 1,),
        input_names=tuple(f"frontier_{index}" for index in range(len(inputs))),
        state_layout=_state_layout_for_nodes(nodes),
    )


def _node_cpp_support(program: StreamingProgram, node_id: int) -> tuple[bool, str | None]:
    """Ask the real native spec lowerer whether a node accepts JAX frontiers."""
    op = program.nodes[node_id].op
    try:
        # Object projections cannot cross a numeric frontier, so validate their
        # complete ancestry exactly as execution lowering does.
        if isinstance(op, NaryOp) and op.cpp_name in {"get_beta", "get_preds"}:
            candidate = _subprogram_for_node(program, node_id)[0]
        else:
            candidate = _synthetic_node_program(program, node_id)
        _cpp_node_specs(candidate)
    except Exception as exc:
        return False, str(exc)
    return True, None


def _groupby_cpp_gaps(op: GroupByOp) -> tuple[str, ...]:
    inner = op.inner_op
    if not isinstance(inner, InnerGraphOp):
        return ("groupby",)
    gaps = []
    for node_id, node in enumerate(inner.nodes):
        inputs = tuple(
            DagNode(InputOp(index, output_kind=inner.nodes[child_id].op.output_kind, output_width=inner.nodes[child_id].op.output_width), ())
            for index, child_id in enumerate(node.child_ids)
        )
        nodes = inputs + (DagNode(node.op, tuple(range(len(inputs)))),)
        candidate = InnerGraphOp(nodes, len(nodes) - 1, _state_layout_for_nodes(nodes), len(inputs))
        try:
            _cpp_inner_node_specs(candidate, _cpp_spec_tuple)
        except Exception:
            gaps.append(_operation_name(node.op))
    return tuple(sorted(set(gaps)))


def explain_cpp_plan(program: StreamingProgram) -> CppLoweringPlan:
    """Partition a lowered program into connected C++ and JAX execution islands.

    This is compile-time-only introspection; it neither imports the extension nor
    executes/traces JAX.  Consequently it is safe to use in tooling and notebooks.
    """
    group_gaps = {node_id: _groupby_cpp_gaps(node.op) for node_id, node in enumerate(program.nodes) if isinstance(node.op, GroupByOp)}
    support = []
    for node_id, node in enumerate(program.nodes):
        supported, reason = _node_cpp_support(program, node_id)
        if group_gaps.get(node_id):
            supported, reason = False, "nested RHS requires JAX: " + ", ".join(group_gaps[node_id])
        support.append((supported, reason))
    backends = ["cpp" if item[0] else "jax" for item in support]
    parents = list(range(len(program.nodes)))

    def find(node_id: int) -> int:
        while parents[node_id] != node_id:
            parents[node_id] = parents[parents[node_id]]
            node_id = parents[node_id]
        return node_id

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for node_id, node in enumerate(program.nodes):
        for child_id in node.child_ids:
            if backends[child_id] == backends[node_id]:
                union(node_id, child_id)

    island_numbers: dict[tuple[str, int], int] = {}
    plan_nodes = []
    for node_id, node in enumerate(program.nodes):
        _, reason = support[node_id]
        backend = backends[node_id]
        component = (backend, find(node_id))
        if component not in island_numbers:
            island_numbers[component] = sum(key[0] == backend for key in island_numbers)
        island = island_numbers[component]
        plan_nodes.append(CppPlanNode(node_id, _operation_name(node.op), backend, island, node.child_ids, reason))
    nested_gaps = tuple(sorted({name for gaps in group_gaps.values() for name in gaps}))
    return CppLoweringPlan(tuple(plan_nodes), program.outputs, program.input_names, nested_gaps)
