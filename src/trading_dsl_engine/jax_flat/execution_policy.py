from __future__ import annotations

"""Conservative execution planner for the public JAX-flat runtime.

The compiler and runtime class remain defined in ``engine.py``. This module
only replaces the JAX batch fallback for DAGs whose state, output shape, and
padding semantics are understood. All blockers, caches, object outputs, native
boundaries, and unsupported operators retain the original execution path.
"""

from collections import deque
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import partial
import os
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.jax_flat import engine
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
    masked_tail: bool


@dataclass(frozen=True)
class EwmBranchPlan:
    base_node_id: int
    levels: tuple[tuple[int, ...], ...]
    tail_nodes: frozenset[int]

    @property
    def breadth(self) -> int:
        return len(self.levels[0])


_PLANNED_OPS = (
    InputOp,
    LiteralOp,
    NaryOp,
    EwmOp,
    CumsumOp,
    FFillOp,
    ShiftOp,
    RollingMeanOp,
    RollingOp,
)


def classify_op(op: Any) -> ExecutionKind:
    if isinstance(op, (InputOp, LiteralOp, NaryOp)):
        return ExecutionKind.STATELESS
    if isinstance(op, (CumsumOp, FFillOp)):
        return ExecutionKind.PREFIX
    if isinstance(op, EwmOp):
        if op.span is not None and op.ignore_na and not op.adjust:
            return ExecutionKind.AFFINE
        return ExecutionKind.SEQUENTIAL
    if isinstance(op, (ShiftOp, RollingMeanOp, RollingOp)):
        return ExecutionKind.LOOKBACK
    if isinstance(op, GroupByOp):
        return ExecutionKind.BLOCKER
    if isinstance(op, CacheOp):
        return ExecutionKind.HOST_NATIVE
    return ExecutionKind.SEQUENTIAL if getattr(op, "is_stateful", False) else ExecutionKind.STATELESS


def _regions(program) -> tuple[ExecutionRegion, ...]:
    result: list[ExecutionRegion] = []
    current_kind: ExecutionKind | None = None
    current_nodes: list[int] = []
    for node_id, node in enumerate(program.nodes):
        kind = classify_op(node.op)
        boundary = kind in {ExecutionKind.BLOCKER, ExecutionKind.HOST_NATIVE}
        if current_nodes and (boundary or kind != current_kind):
            result.append(ExecutionRegion(current_kind, tuple(current_nodes)))
            current_nodes = []
        if boundary:
            result.append(ExecutionRegion(kind, (node_id,)))
            current_kind = None
        else:
            current_kind = kind
            current_nodes.append(node_id)
    if current_nodes:
        result.append(ExecutionRegion(current_kind, tuple(current_nodes)))
    return tuple(result)


def _consumer_lists(program):
    consumers: list[list[int]] = [[] for _ in program.nodes]
    for consumer_id, node in enumerate(program.nodes):
        for child_id in node.child_ids:
            consumers[child_id].append(consumer_id)
    return tuple(tuple(items) for items in consumers)


def _stateful_depth(program) -> int:
    depths: list[int] = []
    for node in program.nodes:
        parent = max((depths[child_id] for child_id in node.child_ids), default=0)
        depths.append(parent + int(bool(node.op.is_stateful)))
    return max((depths[node_id] for node_id in program.outputs), default=0)


def _output_shape_known(op) -> bool:
    return op.output_kind in {"scalar", "vector"} or (
        op.output_kind == "matrix" and op.output_width is not None
    )


def _eligible(program) -> bool:
    if program.cache_nodes:
        return False
    if any(not isinstance(node.op, _PLANNED_OPS) for node in program.nodes):
        return False
    if any(isinstance(node.op, (GroupByOp, CacheOp)) for node in program.nodes):
        return False
    return all(_output_shape_known(program.nodes[node_id].op) for node_id in program.outputs)


def _compatible_branch_ewm(node) -> bool:
    op = node.op
    return (
        isinstance(op, EwmOp)
        and len(node.child_ids) == 1
        and op.span is not None
        and op.ignore_na
        and not op.adjust
        and op.output_kind == "vector"
    )


def _ewm_path(program, output_id: int):
    path: list[int] = []
    node_id = output_id
    while _compatible_branch_ewm(program.nodes[node_id]):
        path.append(node_id)
        node_id = program.nodes[node_id].child_ids[0]
    return node_id, tuple(reversed(path))


def _branch_plan(program) -> EwmBranchPlan | None:
    if len(program.outputs) < 2:
        return None
    walked = tuple(_ewm_path(program, node_id) for node_id in program.outputs)
    bases = tuple(base for base, _ in walked)
    paths = tuple(path for _, path in walked)
    if len(set(bases)) != 1 or any(not path for path in paths):
        return None

    common = 0
    shortest = min(map(len, paths))
    while common < shortest and all(path[common] == paths[0][common] for path in paths[1:]):
        common += 1
    base_node_id = paths[0][common - 1] if common else bases[0]
    tails = tuple(path[common:] for path in paths)
    if any(not tail for tail in tails) or len({len(tail) for tail in tails}) != 1:
        return None

    levels = tuple(tuple(tail[level] for tail in tails) for level in range(len(tails[0])))
    if tuple(levels[-1]) != tuple(program.outputs):
        return None
    if any(len(set(level)) != len(level) for level in levels):
        return None

    consumers = _consumer_lists(program)
    tail_nodes = frozenset(node_id for level in levels for node_id in level)
    for branch_id in range(len(program.outputs)):
        for level_id, level in enumerate(levels):
            node_id = level[branch_id]
            expected_child = base_node_id if level_id == 0 else levels[level_id - 1][branch_id]
            expected_consumers = () if level_id + 1 == len(levels) else (levels[level_id + 1][branch_id],)
            if program.nodes[node_id].child_ids != (expected_child,):
                return None
            if consumers[node_id] != expected_consumers:
                return None
    return EwmBranchPlan(base_node_id, levels, tail_nodes)


def build_execution_plan(program) -> ExecutionPlan:
    stateful = sum(bool(node.op.is_stateful) for node in program.nodes)
    stateless = sum(
        not node.op.is_stateful and not isinstance(node.op, (InputOp, LiteralOp))
        for node in program.nodes
    )
    branch = _branch_plan(program)
    if branch is not None:
        chunk_size = 8_192 if branch.breadth == 4 and _stateful_depth(program) < 8 else 4_096
        strategy = "ewm_branch_batch"
    elif stateful == 0:
        chunk_size = 32_768
        strategy = "node_batch"
    elif stateless == 0 and all(isinstance(node.op, (InputOp, LiteralOp, EwmOp)) for node in program.nodes):
        chunk_size = 4_096
        strategy = "pair_fused_node_batch"
    else:
        chunk_size = 65_536
        strategy = "node_batch"
    return ExecutionPlan(
        regions=_regions(program),
        chunk_size=chunk_size,
        strategy=strategy,
        masked_tail=bool(stateful),
    )


def _value_template(op, n_assets: int):
    if op.output_kind == "scalar":
        return jnp.asarray(0.0, dtype=jnp.float64)
    if op.output_kind == "vector":
        return jnp.zeros((n_assets,), dtype=jnp.float64)
    return jnp.zeros((n_assets, int(op.output_width)), dtype=jnp.float64)


def _invalid_like(value):
    value = jnp.asarray(value)
    if jnp.issubdtype(value.dtype, jnp.inexact):
        return jnp.full_like(value, jnp.nan)
    if jnp.issubdtype(value.dtype, jnp.bool_):
        return jnp.zeros_like(value, dtype=bool)
    return jnp.zeros_like(value)


def _paired_ewm(program, node_id: int, consumers) -> int | None:
    node = program.nodes[node_id]
    if not isinstance(node.op, EwmOp) or len(node.child_ids) != 1:
        return None
    if node_id in program.outputs or len(consumers[node_id]) != 1:
        return None
    consumer_id = consumers[node_id][0]
    consumer = program.nodes[consumer_id]
    if isinstance(consumer.op, EwmOp) and consumer.child_ids == (node_id,):
        return consumer_id
    return None


def _scan_ewm_pair(first_op, second_op, first_state, second_state, values):
    def step(carry, value):
        state1, state2 = carry
        state1, first_value = first_op.tick(state1, value)
        state2, second_value = second_op.tick(state2, first_value)
        return (state1, state2), second_value

    return jax.lax.scan(step, (first_state, second_state), values, unroll=1)


def _associative_ewm(op: EwmOp, state: EwmState, values):
    alpha = jnp.asarray(2.0 / (float(op.span) + 1.0), dtype=values.dtype)
    decay = 1.0 - alpha
    valid = jnp.isfinite(values)
    items = (
        valid,
        jnp.where(valid, decay, 1.0),
        jnp.where(valid, alpha * values, 0.0),
        jnp.where(valid, values, 0.0),
        valid.astype(jnp.int64),
    )

    def combine(left, right):
        has1, a1, b1, u1, count1 = left
        has2, a2, b2, u2, count2 = right
        return (
            has1 | has2,
            a2 * a1,
            a2 * b1 + b2,
            jnp.where(has1, a2 * u1 + b2, u2),
            count1 + count2,
        )

    has, a, b, uninitialized, counts = jax.lax.associative_scan(combine, items, axis=0)
    output_values = jnp.where(state.initialized, a * state.value + b, jnp.where(has, uninitialized, state.value))
    initialized = state.initialized | has
    count = state.count + counts
    enough = True if op.min_periods is None else count >= int(round(float(op.min_periods)))
    outputs = jnp.where(initialized & enough, output_values, jnp.nan)
    final_initialized = initialized[-1]
    return EwmState(
        value=output_values[-1],
        weight=jnp.where(final_initialized, jnp.ones_like(state.weight), state.weight),
        initialized=final_initialized,
        count=count[-1],
    ), outputs


def _scan_node(op, state, child_values):
    threshold = int(os.environ.get("TRADING_DSL_JAX_FLAT_ASSOCIATIVE_EWM_MIN_WIDTH", "512"))
    if (
        isinstance(op, EwmOp)
        and op.span is not None
        and op.ignore_na
        and not op.adjust
        and len(child_values) == 1
        and child_values[0].shape[-1] >= threshold
    ):
        return _associative_ewm(op, state, child_values[0])
    return op.scan_batch(state, *child_values)


def _evaluate(runtime, state_leaves, inputs, batch_start, omitted=frozenset()):
    n_steps = inputs[0].shape[0]
    values = [jnp.asarray(0.0)] * len(runtime.program.nodes)
    new_state = list(state_leaves)
    consumers = _consumer_lists(runtime.program)
    skipped: set[int] = set()

    for node_id, node in enumerate(runtime.program.nodes):
        if node_id in omitted or node_id in skipped:
            continue
        op = node.op
        if isinstance(op, InputOp):
            values[node_id] = inputs[op.input_index]
            continue
        if isinstance(op, LiteralOp):
            values[node_id] = jnp.full((n_steps,), op.value, dtype=jnp.float64)
            continue

        pair_id = _paired_ewm(runtime.program, node_id, consumers)
        if pair_id is not None and pair_id not in omitted:
            pair_node = runtime.program.nodes[pair_id]
            first_field = runtime.program.state_layout.node_fields[node_id]
            second_field = runtime.program.state_layout.node_fields[pair_id]
            (next_first, next_second), pair_values = _scan_ewm_pair(
                op,
                pair_node.op,
                state_leaves[first_field.index],
                state_leaves[second_field.index],
                values[node.child_ids[0]],
            )
            new_state[first_field.index] = next_first
            new_state[second_field.index] = next_second
            values[pair_id] = pair_values
            skipped.add(pair_id)
            continue

        child_values = tuple(values[child_id] for child_id in node.child_ids)
        field = runtime.program.state_layout.node_fields[node_id]
        node_state = None if field.index < 0 else state_leaves[field.index]
        next_state, value = _scan_node(op, node_state, child_values)
        if field.index >= 0:
            new_state[field.index] = next_state
        values[node_id] = value
    return values, new_state


def _stack_states(states) -> EwmState:
    return EwmState(
        value=jnp.stack(tuple(state.value for state in states)),
        weight=jnp.stack(tuple(state.weight for state in states)),
        initialized=jnp.stack(tuple(state.initialized for state in states)),
        count=jnp.stack(tuple(state.count for state in states)),
    )


def _unstack_state(state: EwmState, index: int) -> EwmState:
    return EwmState(state.value[index], state.weight[index], state.initialized[index], state.count[index])


def _batched_tick(ops, state: EwmState, values):
    spans = jnp.asarray(tuple(float(op.span) for op in ops), dtype=values.dtype)[:, None]
    alpha = 2.0 / (spans + 1.0)
    decay = 1.0 - alpha
    minimum = jnp.asarray(
        tuple(-1 if op.min_periods is None else int(round(float(op.min_periods))) for op in ops),
        dtype=jnp.int64,
    )[:, None]
    valid = jnp.isfinite(values)
    decayed_weight = jnp.where(state.initialized & valid, state.weight * decay, state.weight)
    normalized = (decayed_weight * state.value + alpha * values) / (decayed_weight + alpha)
    half = decayed_weight * state.value + (1.0 - decayed_weight) * values
    weighted = jnp.where(jnp.isclose(alpha, 0.5), half, normalized)
    next_value = jnp.where(valid, jnp.where(state.initialized, weighted, values), state.value)
    next_weight = jnp.where(valid, jnp.ones_like(decayed_weight), decayed_weight)
    initialized = state.initialized | valid
    count = state.count + valid.astype(jnp.int64)
    output = jnp.where(initialized & ((minimum < 0) | (count >= minimum)), next_value, jnp.nan)
    return EwmState(next_value, next_weight, initialized, count), output


def _branch_evaluate(runtime, plan: EwmBranchPlan, state_leaves, inputs, batch_start):
    values, new_state = _evaluate(runtime, state_leaves, inputs, batch_start, plan.tail_nodes)
    base = values[plan.base_node_id]
    branch_values = jnp.broadcast_to(base[:, None, :], (base.shape[0], plan.breadth, base.shape[1]))

    level_id = 0
    while level_id < len(plan.levels):
        first_level = plan.levels[level_id]
        first_fields = tuple(runtime.program.state_layout.node_fields[node_id] for node_id in first_level)
        first_states = _stack_states(tuple(state_leaves[field.index] for field in first_fields))
        first_ops = tuple(runtime.program.nodes[node_id].op for node_id in first_level)

        if level_id + 1 < len(plan.levels):
            second_level = plan.levels[level_id + 1]
            second_fields = tuple(runtime.program.state_layout.node_fields[node_id] for node_id in second_level)
            second_states = _stack_states(tuple(state_leaves[field.index] for field in second_fields))
            second_ops = tuple(runtime.program.nodes[node_id].op for node_id in second_level)

            def step(carry, value):
                state1, state2 = carry
                state1, first_value = _batched_tick(first_ops, state1, value)
                state2, second_value = _batched_tick(second_ops, state2, first_value)
                return (state1, state2), second_value

            (next_first, next_second), branch_values = jax.lax.scan(
                step, (first_states, second_states), branch_values, unroll=1
            )
            for branch_id, field in enumerate(first_fields):
                new_state[field.index] = _unstack_state(next_first, branch_id)
            for branch_id, field in enumerate(second_fields):
                new_state[field.index] = _unstack_state(next_second, branch_id)
            level_id += 2
        else:
            def step(state, value):
                return _batched_tick(first_ops, state, value)

            next_first, branch_values = jax.lax.scan(step, first_states, branch_values, unroll=1)
            for branch_id, field in enumerate(first_fields):
                new_state[field.index] = _unstack_state(next_first, branch_id)
            level_id += 1

    return tuple(new_state), tuple(branch_values[:, branch_id, :] for branch_id in range(plan.breadth))


def _chunk_impl(runtime, state_leaves, inputs, batch_start):
    branch = _branch_plan(runtime.program)
    if branch is not None:
        return _branch_evaluate(runtime, branch, state_leaves, inputs, batch_start)
    values, new_state = _evaluate(runtime, state_leaves, inputs, batch_start)
    return tuple(new_state), tuple(values[node_id] for node_id in runtime.program.outputs)


@partial(jax.jit, donate_argnums=(1,))
def _chunk_donate(runtime, state_leaves, inputs, batch_start):
    return _chunk_impl(runtime, state_leaves, inputs, batch_start)


@jax.jit
def _chunk(runtime, state_leaves, inputs, batch_start):
    return _chunk_impl(runtime, state_leaves, inputs, batch_start)


def _tick(runtime, state_leaves, rows):
    values = [jnp.asarray(0.0)] * len(runtime.program.nodes)
    new_state = list(state_leaves)
    for node_id, node in enumerate(runtime.program.nodes):
        op = node.op
        if isinstance(op, InputOp):
            values[node_id] = rows[op.input_index]
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


@jax.jit
def _masked_tail(runtime, state_leaves, inputs, valid_length, invalid_outputs):
    indices = jnp.arange(inputs[0].shape[0], dtype=jnp.int32)

    def step(state, item):
        rows, index = item[:-1], item[-1]
        return jax.lax.cond(
            index < valid_length,
            lambda _: _tick(runtime, state, rows),
            lambda _: (state, invalid_outputs),
            operand=None,
        )

    return jax.lax.scan(step, state_leaves, (*inputs, indices), unroll=1)


def _pad(array, start: int, stop: int, size: int):
    source = np.asarray(array[start:stop], dtype=np.float64)
    if source.shape[0] == size:
        return source
    target = np.full((size,) + source.shape[1:], np.nan, dtype=np.float64)
    target[: source.shape[0]] = source
    return target


def _prepare(inputs, start: int, stop: int, size: int):
    return tuple(jnp.asarray(_pad(array, start, stop, size)) for array in inputs)


def _names(program) -> tuple[str, ...]:
    return tuple(getattr(program, "output_names", ())) or tuple(
        f"output_{index}" for index in range(len(program.outputs))
    )


def _attach_names(program, names: Sequence[str]):
    object.__setattr__(program, "output_names", tuple(names))


def _format(runtime, outputs):
    if len(outputs) == 1:
        return outputs[0]
    return dict(zip(_names(runtime.program), outputs, strict=True))


def _allocate(runtime, n_steps: int, n_assets: int, out_path):
    arrays = []
    multiple = len(runtime.program.outputs) > 1
    for name, node_id in zip(_names(runtime.program), runtime.program.outputs, strict=True):
        template = _value_template(runtime.program.nodes[node_id].op, n_assets)
        shape = (n_steps,) + tuple(np.asarray(template).shape)
        if out_path is False or out_path is None:
            arrays.append(np.empty(shape, dtype=np.asarray(template).dtype))
        else:
            if out_path is True:
                path = engine._fresh_memmap_path(f"trading_dsl_engine_jax_flat_{name}_")
            elif multiple:
                root, extension = os.path.splitext(out_path)
                path = f"{root}.{name}{extension or '.memmap'}"
            else:
                path = out_path
            arrays.append(np.memmap(path, mode="w+", dtype=np.asarray(template).dtype, shape=shape))
    return tuple(arrays)


def _write(item, arrays):
    start, valid, outputs = item
    for target, output in zip(arrays, outputs, strict=True):
        target[start:start + valid] = np.asarray(jax.device_get(output))[:valid]
        if isinstance(target, np.memmap):
            target.flush()


def _run(runtime, inputs, states, out_path):
    n_steps, n_assets = inputs[0].shape[:2]
    state = runtime.init_state(n_assets) if states is None else states
    plan = build_execution_plan(runtime.program)
    chunk_size = min(n_steps, plan.chunk_size, engine._BATCH_CHUNK_SIZE)
    host_output = bool(out_path) or engine._has_memmap_input(inputs)
    arrays = _allocate(runtime, n_steps, n_assets, out_path) if host_output else ()
    max_in_flight = max(1, int(os.environ.get("TRADING_DSL_JAX_FLAT_MAX_IN_FLIGHT", "2")))
    pending = deque()
    device_chunks = []

    templates = tuple(
        _invalid_like(_value_template(runtime.program.nodes[node_id].op, n_assets))
        for node_id in runtime.program.outputs
    )
    starts = tuple(range(0, n_steps, chunk_size))
    executor = ThreadPoolExecutor(max_workers=1) if engine._has_memmap_input(inputs) else None
    future: Future | None = None
    if executor is not None and starts:
        future = executor.submit(_prepare, inputs, starts[0], min(starts[0] + chunk_size, n_steps), chunk_size)

    try:
        for chunk_id, start in enumerate(starts):
            stop = min(start + chunk_size, n_steps)
            valid = stop - start
            if future is None:
                chunk_inputs = _prepare(inputs, start, stop, chunk_size)
            else:
                chunk_inputs = future.result()
                next_id = chunk_id + 1
                if next_id < len(starts):
                    next_start = starts[next_id]
                    future = executor.submit(
                        _prepare, inputs, next_start, min(next_start + chunk_size, n_steps), chunk_size
                    )
                else:
                    future = None

            if valid < chunk_size and plan.masked_tail:
                state, outputs = _masked_tail(
                    runtime, state, chunk_inputs, jnp.asarray(valid, dtype=jnp.int32), templates
                )
            else:
                donate = runtime.program.state_layout.total_leaves <= 1 and not (
                    chunk_id == 0 and states is not None
                )
                kernel = _chunk_donate if donate else _chunk
                state, outputs = kernel(
                    runtime, state, chunk_inputs, jnp.asarray(start, dtype=jnp.int64)
                )

            if host_output:
                pending.append((start, valid, outputs))
                if len(pending) >= max_in_flight:
                    _write(pending.popleft(), arrays)
            else:
                device_chunks.append((valid, outputs))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if host_output:
        while pending:
            _write(pending.popleft(), arrays)
        output = _format(runtime, arrays)
    else:
        outputs = tuple(
            jnp.concatenate(
                tuple(chunk_outputs[index][:valid] for valid, chunk_outputs in device_chunks), axis=0
            )
            for index in range(len(runtime.program.outputs))
        )
        output = _format(runtime, outputs)
    jax.block_until_ready(state)
    return state, output


_LEGACY_RUN_BATCH_ONCE = engine.JaxFlatRuntime._run_batch_once


def _planned_run_batch_once(self, inputs, states=None, out_path: str | bool = False):
    if not _eligible(self.program):
        return _LEGACY_RUN_BATCH_ONCE(self, inputs, states, out_path)

    normalized = engine._normalize_batch_inputs(self, inputs)
    if not normalized:
        raise ValueError("run_batch requires at least one input array")
    n_steps, n_assets = normalized[0].shape[:2]
    if any(array.shape[:2] != (n_steps, n_assets) for array in normalized[1:]):
        raise ValueError("All inputs must share aligned shape (time, n_instruments)")

    if self.cpp and len(self.program.outputs) == 1 and states is None and not out_path:
        try:
            from trading_dsl_engine.jax_flat.engine_cpp import _try_cpp_hybrid_batch
        except Exception as exc:
            engine._warn_cpp_fallback(
                self,
                f"C++ jax_flat accelerator unavailable ({type(exc).__name__}: {exc}); falling back to JAX-flat",
            )
        else:
            hybrid = _try_cpp_hybrid_batch(
                self, normalized, engine._CPP_ACCELERATOR_CACHE, engine._warn_cpp_fallback
            )
            if hybrid is not None:
                return hybrid

    return _run(self, normalized, states, out_path)


engine.JaxFlatRuntime._run_batch_once = _planned_run_batch_once
engine.ExecutionKind = ExecutionKind
engine.ExecutionRegion = ExecutionRegion
engine.ExecutionPlan = ExecutionPlan
engine.classify_op = classify_op
engine.build_execution_plan = build_execution_plan


def compile_features(
    formulas: Mapping[str, Any],
    *,
    dsl_registry=None,
    cpp: bool = False,
    runtimes=None,
):
    if not formulas:
        raise ValueError("compile_features requires at least one named formula")

    external_names, external_values = engine._external_cache_inputs(runtimes)
    nodes = []
    memo = {}
    input_names: list[str] = []
    outputs: list[int] = []
    names: list[str] = []
    for name, formula in formulas.items():
        expression = engine.parse_formula(formula) if isinstance(formula, str) else formula
        expression = engine._normalize_static_jax_flat_kwargs(expression)
        expression = engine._expand_dsl(expression, dsl_registry or engine.DEFAULT_DSL_REGISTRY)
        expression = engine._normalize_static_jax_flat_kwargs(expression)
        outputs.append(engine._compile_node(expression, memo, nodes, input_names, external_names))
        names.append(str(name))

    node_tuple = tuple(nodes)
    cache_nodes = tuple(
        node_id for node_id, node in enumerate(node_tuple) if isinstance(node.op, CacheOp)
    )
    program = engine.StreamingProgram(
        nodes=node_tuple,
        outputs=tuple(outputs),
        input_names=tuple(input_names),
        state_layout=engine._build_state_layout(node_tuple),
        metadata=None,
        cache_nodes=cache_nodes,
        cache_expr_keys=(),
        external_cache_inputs=external_values or None,
    )
    _attach_names(program, names)
    return engine.JaxFlatRuntime(program=program, cpp=cpp)


__all__ = [
    "ExecutionKind",
    "ExecutionRegion",
    "ExecutionPlan",
    "build_execution_plan",
    "classify_op",
    "compile_features",
]
