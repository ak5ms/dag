from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable
import os

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
    ShiftOp,
)


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


def compile_formula(formula: str | Expr, dsl_registry: DSLFunctionRegistry | None = None) -> CppFlatRuntime:
    """Compile a supported jax_flat formula to the native C++ tick runtime.

    This backend intentionally covers the allocation-sensitive scalar/vector tick
    subset first. Unsupported operators raise a clear ``NotImplementedError`` so
    callers can keep using the JAX-flat runtime for full DSL coverage.
    """
    from trading_dsl_engine.jax_flat import _cpp_flat

    runtime = compile_formula_jax(formula, dsl_registry=dsl_registry, cpp=False)
    node_specs, supported, state_count = _cpp_node_specs(runtime.program)
    core = _cpp_flat.make_runtime(node_specs, runtime.program.outputs[0], state_count)
    return CppFlatRuntime(program=runtime.program, core=core, supported_ops=tuple(sorted(set(supported))))


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


def _cpp_node_specs(program: StreamingProgram):
    if len(program.outputs) != 1:
        raise NotImplementedError("C++ jax_flat currently supports exactly one output")
    specs = []
    supported = []
    state_count = program.state_layout.total_leaves

    def spec_tuple(
        name,
        children=(),
        input_index=-1,
        state_index=-1,
        literal=0.0,
        param=0.0,
        int_param=0,
        width=1,
        feature_widths=(),
        inner_specs=(),
        inner_output_id=-1,
        str_param="",
    ):
        return (
            name,
            tuple(children),
            input_index,
            state_index,
            float(literal),
            float(param),
            int(int_param),
            int(1 if width is None else width),
            tuple(int(w) for w in feature_widths),
            tuple(inner_specs),
            int(inner_output_id),
            str(str_param),
        )

    for idx, node in enumerate(program.nodes):
        op = node.op
        field = program.state_layout.node_fields[idx]
        state_index = field.index
        width = op.output_width or 1
        if isinstance(op, InputOp):
            specs.append(spec_tuple("input", input_index=op.input_index, width=1))
            supported.append("input")
            continue
        if isinstance(op, LiteralOp):
            specs.append(spec_tuple("literal", literal=op.value, width=1))
            supported.append("literal")
            continue
        if isinstance(op, GroupByOp):
            group_spec, group_supported = _cpp_groupby_spec(op, node.child_ids, state_index, spec_tuple)
            specs.append(group_spec)
            supported.extend(group_supported)
            continue
        if isinstance(op, CacheOp):
            specs.append(spec_tuple("cache", node.child_ids, width=width))
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
            specs.append(spec_tuple(op.cpp_name, node.child_ids, int_param=op.cpp_int_param, param=op.cpp_param, width=cpp_width, feature_widths=feature_widths, str_param=getattr(op, "cpp_str_param", "")))
            supported.append(op.cpp_name)
            continue
        if isinstance(op, CumsumOp):
            specs.append(spec_tuple("cumsum", node.child_ids, state_index=state_index, width=width))
            supported.append("cumsum")
            continue
        if isinstance(op, EwmOp) and op.span is not None:
            min_periods = -1 if op.min_periods is None else int(round(float(op.min_periods)))
            ewm_flags = (1 if op.ignore_na else 0) | (2 if op.adjust else 0)
            specs.append(spec_tuple("ewm", node.child_ids, state_index=state_index, param=op.span, int_param=(min_periods + 1) * 4 + ewm_flags, width=width))
            supported.append("ewm")
            continue
        if isinstance(op, RollingMeanOp):
            specs.append(spec_tuple("roll_mean", node.child_ids, state_index=state_index, param=op.min_periods, int_param=op.lookback, width=width))
            supported.append("roll_mean")
            continue
        if isinstance(op, FFillOp) and not op.dynamic_limit:
            limit = -1 if op.limit is None else op.limit
            specs.append(spec_tuple("ffill", node.child_ids, state_index=state_index, int_param=limit, width=width))
            supported.append("ffill")
            continue
        if isinstance(op, ShiftOp):
            specs.append(spec_tuple("shift", node.child_ids, state_index=state_index, int_param=op.max_size, width=width))
            supported.append("shift")
            continue
        if isinstance(op, RbfBasisOp):
            specs.append(spec_tuple("rbf_basis", node.child_ids, int_param=op.n_basis, width=op.n_basis))
            supported.append("rbf_basis")
            continue
        if isinstance(op, FutureRbfBasisSumOp):
            specs.append(spec_tuple("future_rbf_basis_sum", node.child_ids, int_param=op.n_basis, param=op.n_steps, width=op.n_basis))
            supported.append("future_rbf_basis_sum")
            continue
        if isinstance(op, InstrumentBasisMeanOp):
            specs.append(
                spec_tuple(
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
            specs.append(spec_tuple("ridge", node.child_ids, state_index=state_index, width=1, feature_widths=op.feature_widths))
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
        if isinstance(op, EwmOp) and op.span is not None:
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

def _try_cpp_hybrid_batch(runtime, inputs, accelerator_cache: dict[tuple[Any, ...], Any], warn_callback: Callable[[Any, str], None]):
    """Run full-native or staged native/JAX/native batch execution when possible."""
    if os.getenv("TRADING_DSL_ENGINE_DISABLE_CPP_ACCEL", "0") == "1":
        return None
    full = _try_cpp_full_batch(runtime, inputs, accelerator_cache, warn_callback, emit_warning=False)
    if full is not None:
        return full

    candidates = _cpp_hybrid_candidates(runtime.program)
    if not candidates:
        if any(isinstance(node.op, GroupByOp) for node in runtime.program.nodes):
            _try_cpp_full_batch(runtime, inputs, accelerator_cache, warn_callback, emit_warning=True)
        return None
    try:
        from trading_dsl_engine.jax_flat import _cpp_flat
    except Exception as exc:
        warn_callback(runtime, f"C++ jax_flat hybrid accelerator unavailable ({type(exc).__name__}: {exc}); falling back to JAX-flat")
        return None

    n_steps, n_instruments = inputs[0].shape
    extra_inputs = []
    candidate_programs = []
    for node_id in candidates:
        subprogram = _subprogram_for_node(runtime.program, node_id)
        try:
            node_specs, _, state_count = _cpp_node_specs(subprogram)
        except NotImplementedError:
            continue
        key = (node_specs, subprogram.outputs[0], state_count, n_instruments)
        core = accelerator_cache.get(key)
        if core is None:
            core = _cpp_flat.make_runtime(node_specs, subprogram.outputs[0], state_count)
            accelerator_cache[key] = core
        state = core.init_state(n_instruments)
        raw = core.run_batch(state, *tuple(np.asarray(arr, dtype=np.float64) for arr in inputs))
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


def _try_cpp_full_batch(runtime, inputs, accelerator_cache: dict[tuple[Any, ...], Any], warn_callback: Callable[[Any, str], None], *, emit_warning: bool):
    try:
        from trading_dsl_engine.jax_flat import _cpp_flat
        node_specs, _, state_count = _cpp_node_specs(runtime.program)
    except Exception as exc:
        if emit_warning:
            warn_callback(runtime, f"C++ jax_flat accelerator unsupported for this formula: {exc}; falling back to JAX-flat")
        return None

    n_steps, n_instruments = inputs[0].shape
    key = (node_specs, runtime.program.outputs[0], state_count, n_instruments)
    core = accelerator_cache.get(key)
    if core is None:
        core = _cpp_flat.make_runtime(node_specs, runtime.program.outputs[0], state_count)
        accelerator_cache[key] = core
    state = core.init_state(n_instruments)
    raw = core.run_batch(state, *tuple(np.asarray(arr, dtype=np.float64) for arr in inputs))
    return state, _reshape_cpp_batch_output(runtime.program, raw, n_steps, n_instruments)


def _try_cpp_staged_output_batch(runtime, inputs, upstream_node_ids: tuple[int, ...], upstream_values: tuple[np.ndarray, ...], accelerator_cache: dict[tuple[Any, ...], Any]):
    if len(runtime.program.outputs) != 1:
        return None
    root_id = runtime.program.outputs[0]
    try:
        subprogram, frontier_ids = _subprogram_for_node_with_frontier(runtime.program, root_id)
    except NotImplementedError:
        return None
    if not frontier_ids:
        return None
    try:
        from trading_dsl_engine.jax_flat import _cpp_flat
        node_specs, _, state_count = _cpp_node_specs(subprogram)
    except Exception:
        return None

    residual = _program_with_cpp_inputs(runtime.program, upstream_node_ids)
    residual = replace(residual, outputs=frontier_ids)
    residual_runtime = replace(runtime, program=residual, cpp=False)
    residual_inputs = tuple(inputs) + tuple(jnp.asarray(arr) for arr in upstream_values)
    _, frontier_out = residual_runtime._run_batch_once(residual_inputs, None, False)
    frontier_values = _split_frontier_outputs(frontier_out, len(frontier_ids))

    n_steps, n_instruments = inputs[0].shape
    key = (node_specs, subprogram.outputs[0], state_count, n_instruments)
    core = accelerator_cache.get(key)
    if core is None:
        core = _cpp_flat.make_runtime(node_specs, subprogram.outputs[0], state_count)
        accelerator_cache[key] = core
    state = core.init_state(n_instruments)
    original_inputs = tuple(np.asarray(arr, dtype=np.float64) for arr in inputs)
    frontier_inputs = tuple(np.asarray(arr, dtype=np.float64) for arr in frontier_values)
    raw = core.run_batch(state, *(original_inputs + frontier_inputs))
    return state, _reshape_cpp_batch_output(subprogram, raw, n_steps, n_instruments)


def _split_frontier_outputs(frontier_out, n_frontiers: int) -> tuple[np.ndarray, ...]:
    if n_frontiers == 1:
        return (np.asarray(frontier_out),)
    arr = np.asarray(frontier_out)
    if arr.ndim >= 1 and arr.shape[0] == n_frontiers:
        return tuple(np.asarray(arr[i]) for i in range(n_frontiers))
    return tuple(np.asarray(x) for x in frontier_out)


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


def _subprogram_for_node_with_frontier(program: StreamingProgram, node_id: int) -> tuple[StreamingProgram, tuple[int, ...]]:
    frontier: list[int] = []
    remap: dict[int, int] = {}
    nodes: list[DagNode] = []

    def build(old_id: int, *, force_frontier: bool = False) -> int:
        if old_id in remap:
            return remap[old_id]
        op = program.nodes[old_id].op
        if force_frontier or not _is_cpp_boundary_op(op):
            if old_id not in frontier:
                frontier.append(old_id)
            input_index = len(program.input_names) + frontier.index(old_id)
            remap[old_id] = len(nodes)
            nodes.append(DagNode(InputOp(input_index, output_kind=op.output_kind, output_width=op.output_width), ()))
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
        input_names=program.input_names + tuple(f"__jax_frontier_{idx}" for idx in frontier),
        state_layout=_state_layout_for_nodes(tuple(nodes)),
        metadata=None,
        cache_nodes=(),
    )
    _cpp_node_specs(subprogram)
    return subprogram, tuple(frontier)


def _cpp_hybrid_candidates(program: StreamingProgram) -> tuple[int, ...]:
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
            _cpp_node_specs(_subprogram_for_node(program, node_id))
        except Exception:
            continue
        selected.append(node_id)
        covered.update(ancestors)
    return tuple(sorted(selected))


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


def _subprogram_for_node(program: StreamingProgram, node_id: int) -> StreamingProgram:
    ids = sorted(_ancestor_ids(program, node_id))
    remap = {old: new for new, old in enumerate(ids)}
    nodes = tuple(DagNode(program.nodes[old].op, tuple(remap[c] for c in program.nodes[old].child_ids)) for old in ids)
    return StreamingProgram(nodes=nodes, outputs=(remap[node_id],), input_names=program.input_names, state_layout=_state_layout_for_nodes(nodes), metadata=None, cache_nodes=())


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

