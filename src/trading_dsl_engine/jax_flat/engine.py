from __future__ import annotations

"""Public JAX-flat engine with planned batch execution.

The original implementation is retained in :mod:`engine_legacy` so private
symbols used by the native accelerator and older callers remain available. This
module replaces only batch planning/lowering and compilation entry points.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from functools import partial
import os
import time
import warnings
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.jax_flat import engine_legacy as _legacy
from trading_dsl_engine.jax_flat.engine_legacy import *  # noqa: F401,F403
from trading_dsl_engine.jax_flat.ops import (
    CacheOp,
    CumsumOp,
    EwmOp,
    EwmState,
    FFillOp,
    GroupByOp,
    InputOp,
    LiteralOp,
    NaryOp,
    RollingMeanOp,
    RollingOp,
    ShiftOp,
)


class ExecutionKind(str, Enum):
    STATELESS = "stateless"
    PREFIX = "prefix"
    AFFINE = "affine"
    SEQUENTIAL = "sequential"
    LOOKBACK = "lookback"
    BLOCKER = "blocker"
    HOST_NATIVE = "host_native"


@dataclass(frozen=True)
class ExecutionRegion:
    kind: ExecutionKind
    node_ids: tuple[int, ...]


@dataclass(frozen=True)
class ExecutionPlan:
    regions: tuple[ExecutionRegion, ...]
    chunk_size: int
    strategy: str
    pad_safe: bool


@dataclass(frozen=True)
class EwmBranchPlan:
    base_node_id: int
    levels: tuple[tuple[int, ...], ...]
    tail_nodes: frozenset[int]

    @property
    def breadth(self) -> int:
        return len(self.levels[0])

    @property
    def depth(self) -> int:
        return len(self.levels)


def classify_op(op: Any) -> ExecutionKind:
    if isinstance(op, (InputOp, LiteralOp, NaryOp)):
        return ExecutionKind.STATELESS
    if isinstance(op, (CumsumOp, FFillOp)):
        return ExecutionKind.PREFIX
    if isinstance(op, EwmOp):
        if op.span is not None and op.ignore_na and not op.adjust:
            return ExecutionKind.AFFINE
        return ExecutionKind.SEQUENTIAL
    if isinstance(op, (RollingMeanOp, RollingOp, ShiftOp)):
        return ExecutionKind.LOOKBACK
    if isinstance(op, GroupByOp):
        return ExecutionKind.BLOCKER
    if isinstance(op, CacheOp):
        return ExecutionKind.HOST_NATIVE
    return ExecutionKind.SEQUENTIAL if getattr(op, "is_stateful", False) else ExecutionKind.STATELESS


def _build_regions(program: StreamingProgram) -> tuple[ExecutionRegion, ...]:
    regions: list[ExecutionRegion] = []
    current_kind: ExecutionKind | None = None
    current_nodes: list[int] = []
    for node_id, node in enumerate(program.nodes):
        kind = classify_op(node.op)
        hard_boundary = kind in {ExecutionKind.BLOCKER, ExecutionKind.HOST_NATIVE}
        if current_nodes and (hard_boundary or kind != current_kind):
            regions.append(ExecutionRegion(current_kind, tuple(current_nodes)))
            current_nodes = []
        if hard_boundary:
            regions.append(ExecutionRegion(kind, (node_id,)))
            current_kind = None
        else:
            current_kind = kind
            current_nodes.append(node_id)
    if current_nodes:
        regions.append(ExecutionRegion(current_kind, tuple(current_nodes)))
    return tuple(regions)


def _consumer_lists(program: StreamingProgram):
    consumers: list[list[int]] = [[] for _ in program.nodes]
    for consumer_id, node in enumerate(program.nodes):
        for child_id in node.child_ids:
            consumers[child_id].append(consumer_id)
    return tuple(tuple(items) for items in consumers)


def _compatible_branch_ewm(node: DagNode) -> bool:
    op = node.op
    return (
        isinstance(op, EwmOp)
        and len(node.child_ids) == 1
        and op.span is not None
        and op.ignore_na
        and not op.adjust
        and op.output_kind == "vector"
    )


def _ewm_path(program: StreamingProgram, output_id: int):
    path_reversed: list[int] = []
    node_id = output_id
    while _compatible_branch_ewm(program.nodes[node_id]):
        path_reversed.append(node_id)
        node_id = program.nodes[node_id].child_ids[0]
    return node_id, tuple(reversed(path_reversed))


def _detect_ewm_branch_plan(program: StreamingProgram) -> EwmBranchPlan | None:
    if len(program.outputs) < 2:
        return None
    walked = tuple(_ewm_path(program, output_id) for output_id in program.outputs)
    raw_bases = tuple(item[0] for item in walked)
    paths = tuple(item[1] for item in walked)
    if not paths or any(not path for path in paths) or len(set(raw_bases)) != 1:
        return None
    common = 0
    shortest = min(map(len, paths))
    while common < shortest and all(path[common] == paths[0][common] for path in paths[1:]):
        common += 1
    base_node_id = paths[0][common - 1] if common else raw_bases[0]
    tails = tuple(path[common:] for path in paths)
    if any(not tail for tail in tails) or len({len(tail) for tail in tails}) != 1:
        return None
    levels = tuple(tuple(tail[level] for tail in tails) for level in range(len(tails[0])))
    if any(len(set(level)) != len(level) for level in levels):
        return None
    if tuple(levels[-1]) != tuple(program.outputs):
        return None
    tail_nodes = frozenset(node_id for level in levels for node_id in level)
    if tail_nodes.intersection(program.cache_nodes):
        return None
    consumers = _consumer_lists(program)
    for branch_i in range(len(program.outputs)):
        for level_i, level in enumerate(levels):
            node_id = level[branch_i]
            expected = () if level_i + 1 == len(levels) else (levels[level_i + 1][branch_i],)
            if consumers[node_id] != expected:
                return None
            expected_child = base_node_id if level_i == 0 else levels[level_i - 1][branch_i]
            if program.nodes[node_id].child_ids != (expected_child,):
                return None
    return EwmBranchPlan(base_node_id, levels, tail_nodes)


def _pad_safe(program: StreamingProgram) -> bool:
    for node in program.nodes:
        kind = classify_op(node.op)
        if kind in {ExecutionKind.STATELESS, ExecutionKind.PREFIX, ExecutionKind.AFFINE, ExecutionKind.LOOKBACK}:
            continue
        if isinstance(node.op, CacheOp):
            continue
        return False
    return True


def _program_counts(program: StreamingProgram) -> tuple[int, int]:
    stateful = stateless = 0
    for node in program.nodes:
        if isinstance(node.op, (InputOp, LiteralOp, CacheOp)):
            continue
        if node.op.is_stateful:
            stateful += 1
        else:
            stateless += 1
    return stateful, stateless


def _automatic_chunk_size(program: StreamingProgram) -> int:
    branch_plan = _detect_ewm_branch_plan(program)
    stateful, stateless = _program_counts(program)
    if branch_plan is not None:
        total_depth = _legacy._stateful_depth(program) if hasattr(_legacy, "_stateful_depth") else stateful
        return 8_192 if branch_plan.breadth == 4 and total_depth < 8 else 4_096
    if stateful == 0:
        return 32_768
    if stateless == 0 and all(isinstance(node.op, (InputOp, LiteralOp, CacheOp, EwmOp)) for node in program.nodes):
        return 4_096
    return 65_536


def build_execution_plan(program: StreamingProgram) -> ExecutionPlan:
    regions = _build_regions(program)
    branch_plan = _detect_ewm_branch_plan(program)
    pad_safe = _pad_safe(program)
    if branch_plan is not None:
        strategy = "ewm_branch_batch"
    elif any(region.kind == ExecutionKind.BLOCKER for region in regions):
        strategy = "compound"
    else:
        strategy = "node_batch" if pad_safe else "compound"
    return ExecutionPlan(regions, _automatic_chunk_size(program), strategy, pad_safe)


def _stack_ewm_states(states) -> EwmState:
    return EwmState(
        value=jnp.stack(tuple(state.value for state in states), axis=0),
        weight=jnp.stack(tuple(state.weight for state in states), axis=0),
        initialized=jnp.stack(tuple(state.initialized for state in states), axis=0),
        count=jnp.stack(tuple(state.count for state in states), axis=0),
    )


def _unstack_ewm_state(state: EwmState, index: int) -> EwmState:
    return EwmState(state.value[index], state.weight[index], state.initialized[index], state.count[index])


def _scan_batched_ewm_level(ops, state: EwmState, values):
    spans = jnp.asarray(tuple(float(op.span) for op in ops), dtype=values.dtype)[:, None]
    alpha = 2.0 / (spans + 1.0)
    old_wt_factor = 1.0 - alpha
    min_periods = jnp.asarray(
        tuple(-1 if op.min_periods is None else int(round(float(op.min_periods))) for op in ops),
        dtype=jnp.int64,
    )[:, None]

    def step(carry: EwmState, x_t):
        value, weight, initialized, count = carry.value, carry.weight, carry.initialized, carry.count
        valid = jnp.isfinite(x_t)
        decayed_weight = jnp.where(initialized & valid, weight * old_wt_factor, weight)
        normalized = (decayed_weight * value + alpha * x_t) / (decayed_weight + alpha)
        half = decayed_weight * value + (1.0 - decayed_weight) * x_t
        weighted = jnp.where(jnp.isclose(alpha, 0.5), half, normalized)
        next_value = jnp.where(valid, jnp.where(initialized, weighted, x_t), value)
        next_weight = jnp.where(valid, jnp.ones_like(decayed_weight), decayed_weight)
        next_initialized = initialized | valid
        next_count = count + valid.astype(jnp.int64)
        enough = (min_periods < 0) | (next_count >= min_periods)
        out = jnp.where(next_initialized & enough, next_value, jnp.nan)
        return EwmState(next_value, next_weight, next_initialized, next_count), out

    return jax.lax.scan(step, state, values, unroll=1)


def _evaluate_node_batch(runtime, state_leaves, inputs, batch_start, omitted=frozenset()):
    n_steps = inputs[0].shape[0]
    values = [jnp.asarray(0.0)] * len(runtime.program.nodes)
    new_state = list(state_leaves)
    for node_id, node in enumerate(runtime.program.nodes):
        if node_id in omitted:
            continue
        op = node.op
        if isinstance(op, InputOp):
            values[node_id] = inputs[op.input_index]
            continue
        if isinstance(op, LiteralOp):
            values[node_id] = jnp.full((n_steps,), op.value, dtype=jnp.float64)
            continue
        child_values = tuple(values[child_id] for child_id in node.child_ids)
        field = runtime.program.state_layout.node_fields[node_id]
        node_state = None if field.index < 0 else state_leaves[field.index]
        next_state, value = (
            op.scan_batch_with_start(node_state, batch_start, *child_values)
            if isinstance(op, CacheOp)
            else op.scan_batch(node_state, *child_values)
        )
        if field.index >= 0:
            new_state[field.index] = next_state
        values[node_id] = value
    return values, new_state


def _node_batch_branch_impl(runtime, plan: EwmBranchPlan, state_leaves, inputs, batch_start):
    values, new_state = _evaluate_node_batch(runtime, state_leaves, inputs, batch_start, plan.tail_nodes)
    breadth = plan.breadth
    base = values[plan.base_node_id]
    branch_values = jnp.broadcast_to(base[:, None, :], (base.shape[0], breadth, base.shape[1]))
    for level in plan.levels:
        fields = tuple(runtime.program.state_layout.node_fields[node_id] for node_id in level)
        states = _stack_ewm_states(tuple(state_leaves[field.index] for field in fields))
        ops = tuple(runtime.program.nodes[node_id].op for node_id in level)
        next_states, branch_values = _scan_batched_ewm_level(ops, states, branch_values)
        for branch_i, field in enumerate(fields):
            new_state[field.index] = _unstack_ewm_state(next_states, branch_i)
    outputs = tuple(branch_values[:, branch_i, :] for branch_i in range(breadth))
    cache_outputs = tuple(values[node_id] for node_id in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


@partial(jax.jit, donate_argnums=(1,))
def _planned_node_batch_chunk(runtime, state_leaves, inputs, batch_start):
    plan = _detect_ewm_branch_plan(runtime.program)
    if plan is not None:
        return _node_batch_branch_impl(runtime, plan, state_leaves, inputs, batch_start)
    values, new_state = _evaluate_node_batch(runtime, state_leaves, inputs, batch_start)
    outputs = tuple(values[node_id] for node_id in runtime.program.outputs)
    cache_outputs = tuple(values[node_id] for node_id in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


def _tick_program(runtime, state_leaves, input_rows):
    values = [jnp.asarray(0.0)] * len(runtime.program.nodes)
    new_state = list(state_leaves)
    for node_id, node in enumerate(runtime.program.nodes):
        op = node.op
        if isinstance(op, InputOp):
            values[node_id] = input_rows[op.input_index]
            continue
        if isinstance(op, LiteralOp):
            values[node_id] = jnp.asarray(op.value, dtype=jnp.float64)
            continue
        child_values = tuple(values[child_id] for child_id in node.child_ids)
        field = runtime.program.state_layout.node_fields[node_id]
        node_state = None if field.index < 0 else state_leaves[field.index]
        next_state, value = op.tick(node_state, *child_values)
        if field.index >= 0:
            new_state[field.index] = next_state
        values[node_id] = value
    return tuple(new_state), tuple(values[node_id] for node_id in runtime.program.outputs)


@partial(jax.jit, donate_argnums=(1,))
def _planned_compound_chunk(runtime, state_leaves, inputs, valid_length):
    n_steps = inputs[0].shape[0]
    row_indices = jnp.arange(n_steps, dtype=jnp.int32)
    templates = tuple(_value_template(runtime.program.nodes[node_id].op, inputs[0].shape[1]) for node_id in runtime.program.outputs)

    def step(state, xs):
        rows, row_id = xs[:-1], xs[-1]
        return jax.lax.cond(
            row_id < valid_length,
            lambda _: _tick_program(runtime, state, rows),
            lambda _: (state, tuple(_invalid_like(template) for template in templates)),
            operand=None,
        )

    return jax.lax.scan(step, state_leaves, (*inputs, row_indices), unroll=1)


def _value_template(op, n_instruments: int):
    if op.output_kind == "scalar":
        return jnp.asarray(0.0, dtype=jnp.float64)
    if op.output_kind == "vector":
        return jnp.zeros((n_instruments,), dtype=jnp.float64)
    if op.output_kind == "matrix" and op.output_width is not None:
        return jnp.zeros((n_instruments, int(op.output_width)), dtype=jnp.float64)
    raise ValueError(f"Cannot infer output shape for {op.output_kind!r}")


def _invalid_like(value):
    value = jnp.asarray(value)
    if jnp.issubdtype(value.dtype, jnp.inexact):
        return jnp.full_like(value, jnp.nan)
    if jnp.issubdtype(value.dtype, jnp.bool_):
        return jnp.zeros_like(value, dtype=bool)
    return jnp.zeros_like(value)


def _pad_chunk(array, start: int, stop: int, chunk_size: int):
    source = np.asarray(array[start:stop], dtype=np.float64)
    if source.shape[0] == chunk_size:
        return source
    target = np.full((chunk_size,) + source.shape[1:], np.nan, dtype=np.float64)
    target[: source.shape[0]] = source
    return target


def _allocate_outputs(runtime, n_steps: int, n_instruments: int, out_path):
    templates = tuple(_value_template(runtime.program.nodes[node_id].op, n_instruments) for node_id in runtime.program.outputs)
    multiple = len(templates) > 1
    names = runtime.program.output_names or tuple(f"output_{i}" for i in range(len(templates)))
    arrays = []
    for name, template in zip(names, templates, strict=True):
        shape = (n_steps,) + tuple(np.asarray(template).shape)
        if out_path is False or out_path is None:
            arrays.append(np.empty(shape, dtype=np.asarray(template).dtype))
            continue
        if out_path is True:
            path = _legacy._fresh_memmap_path(f"trading_dsl_engine_jax_flat_{name}_")
        elif isinstance(out_path, str):
            if not multiple:
                path = out_path
            else:
                root, ext = os.path.splitext(out_path)
                path = f"{root}.{name}{ext or '.memmap'}"
        else:
            raise ValueError("out_path must be False, True, None, or a filesystem path")
        arrays.append(np.memmap(path, mode="w+", dtype=np.asarray(template).dtype, shape=shape))
    return tuple(arrays)


def _materialize_pending(item, output_arrays):
    start, valid_length, values = item
    stop = start + valid_length
    for target, value in zip(output_arrays, values, strict=True):
        target[start:stop] = np.asarray(jax.device_get(value))[:valid_length]
        if isinstance(target, np.memmap):
            target.flush()


def _format_outputs(runtime, arrays):
    if len(arrays) == 1:
        return arrays[0]
    return dict(zip(runtime.program.output_names, arrays, strict=True))


def _run_planned_jax_batch(runtime, inputs, states, out_path):
    n_steps, n_instruments = inputs[0].shape[:2]
    state = runtime.init_state(n_instruments) if states is None else states
    plan = build_execution_plan(runtime.program)
    configured = int(os.environ.get("TRADING_DSL_JAX_FLAT_BATCH_CHUNK_SIZE", "0"))
    chunk_size = min(n_steps, configured or plan.chunk_size)
    max_in_flight = max(1, int(os.environ.get("TRADING_DSL_JAX_FLAT_MAX_IN_FLIGHT", "2")))
    output_arrays = _allocate_outputs(runtime, n_steps, n_instruments, out_path)
    pending = deque()

    for start in range(0, n_steps, chunk_size):
        stop = min(start + chunk_size, n_steps)
        valid_length = stop - start
        chunk_inputs = tuple(jnp.asarray(_pad_chunk(array, start, stop, chunk_size)) for array in inputs)
        if plan.strategy == "compound":
            state, chunk_outputs = _planned_compound_chunk(runtime, state, chunk_inputs, jnp.asarray(valid_length, jnp.int32))
        else:
            state, (chunk_outputs, _) = _planned_node_batch_chunk(
                runtime, state, chunk_inputs, jnp.asarray(start, dtype=jnp.int64)
            )
        pending.append((start, valid_length, chunk_outputs))
        if len(pending) >= max_in_flight:
            _materialize_pending(pending.popleft(), output_arrays)
    while pending:
        _materialize_pending(pending.popleft(), output_arrays)
    jax.block_until_ready(state)
    return state, _format_outputs(runtime, output_arrays)


# Preserve the legacy class object so native modules importing it keep working,
# then install the planned methods directly on that public class.
JaxFlatRuntime = _legacy.JaxFlatRuntime


@partial(jax.jit, donate_argnums=(1,))
def _planned_tick(self, state_leaves, *input_rows):
    next_state, outputs = _tick_program(self, state_leaves, input_rows)
    return next_state, outputs[0] if len(outputs) == 1 else outputs


def _planned_run_batch_once(self, inputs, states=None, out_path: str | bool = False):
    inputs = _legacy._normalize_batch_inputs(self, inputs)
    if not inputs:
        raise ValueError("run_batch requires at least one input array")
    n_steps, n_instruments = inputs[0].shape[:2]
    if any(array.shape[:2] != (n_steps, n_instruments) for array in inputs[1:]):
        raise ValueError("All inputs must share aligned shape (time, n_instruments)")
    if self.program.cache_nodes:
        self.clear_cached_values()
    if self.cpp and len(self.program.outputs) == 1 and not self.program.cache_nodes and states is None and not out_path:
        try:
            from trading_dsl_engine.jax_flat.engine_cpp import _try_cpp_hybrid_batch
        except Exception as exc:
            _legacy._warn_cpp_fallback(self, f"C++ jax_flat accelerator unavailable ({type(exc).__name__}: {exc}); falling back to JAX-flat")
        else:
            hybrid = _try_cpp_hybrid_batch(self, inputs, _legacy._CPP_ACCELERATOR_CACHE, _legacy._warn_cpp_fallback)
            if hybrid is not None:
                return hybrid
    return _run_planned_jax_batch(self, inputs, states, out_path)


JaxFlatRuntime.tick = _planned_tick
JaxFlatRuntime._run_batch_once = _planned_run_batch_once


StreamingProgram = _legacy.StreamingProgram
if "output_names" not in getattr(StreamingProgram, "__dataclass_fields__", {}):
    # The legacy dataclass cannot be mutated structurally. Multi-output names are
    # attached after construction; they are static Python metadata and therefore
    # do not participate in JAX pytrees.
    StreamingProgram.output_names = ()


def _attach_output_names(program: StreamingProgram, names: Sequence[str]):
    object.__setattr__(program, "output_names", tuple(names))
    return program


def compile_formula(
    formula,
    cpp: bool = True,
    dsl_registry=None,
    metadata_config=None,
    runtimes=None,
):
    runtime = _legacy.compile_formula(
        formula,
        cpp=cpp,
        dsl_registry=dsl_registry,
        metadata_config=metadata_config,
        runtimes=runtimes,
    )
    _attach_output_names(runtime.program, ("output",))
    return runtime


def compile_features(
    formulas: Mapping[str, Any],
    *,
    cpp: bool = False,
    dsl_registry=None,
    runtimes=None,
):
    if not formulas:
        raise ValueError("compile_features requires at least one named formula")
    external_cache_names, external_cache_values = _legacy._external_cache_inputs(runtimes)
    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    input_names: list[str] = []
    outputs: list[int] = []
    names: list[str] = []
    for name, formula in formulas.items():
        expr = _legacy.parse_formula(formula) if isinstance(formula, str) else formula
        expr = _legacy._normalize_static_jax_flat_kwargs(expr)
        expr = _legacy._expand_dsl(expr, dsl_registry or _legacy.DEFAULT_DSL_REGISTRY)
        expr = _legacy._normalize_static_jax_flat_kwargs(expr)
        outputs.append(_legacy._compile_node(expr, memo, nodes, input_names, external_cache_names))
        names.append(str(name))
    node_tuple = tuple(nodes)
    cache_nodes = tuple(index for index, node in enumerate(node_tuple) if isinstance(node.op, CacheOp))
    program = StreamingProgram(
        nodes=node_tuple,
        outputs=tuple(outputs),
        input_names=tuple(input_names),
        state_layout=_legacy._build_state_layout(node_tuple),
        metadata=None,
        cache_nodes=cache_nodes,
        cache_expr_keys=(),
        external_cache_inputs=external_cache_values or None,
    )
    _attach_output_names(program, names)
    return JaxFlatRuntime(program=program, cpp=cpp)


# Private compatibility exports used elsewhere in the package.
for _name in dir(_legacy):
    if _name.startswith("_") and _name not in globals():
        globals()[_name] = getattr(_legacy, _name)


__all__ = [
    *getattr(_legacy, "__all__", ()),
    "ExecutionKind",
    "ExecutionRegion",
    "ExecutionPlan",
    "build_execution_plan",
    "classify_op",
    "compile_formula",
    "compile_features",
]
