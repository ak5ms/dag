from __future__ import annotations

"""JAX-flat compiler and planned batch executor.

The original implementation lives in ``engine_legacy``. This module preserves
its public/private symbols while replacing the JAX batch fallback with a
region-aware, cache-tiled executor. Existing ``compile_formula`` and
``run_batch`` behavior remains the default API.
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
    masked_tail: bool


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
            assert current_kind is not None
            regions.append(ExecutionRegion(current_kind, tuple(current_nodes)))
            current_nodes = []
        if hard_boundary:
            regions.append(ExecutionRegion(kind, (node_id,)))
            current_kind = None
        else:
            current_kind = kind
            current_nodes.append(node_id)
    if current_nodes:
        assert current_kind is not None
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


def _padding_preserves_state(program: StreamingProgram) -> bool:
    for node in program.nodes:
        op = node.op
        if not getattr(op, "is_stateful", False):
            continue
        if isinstance(op, CumsumOp):
            continue
        if isinstance(op, FFillOp):
            continue
        if isinstance(op, EwmOp) and op.ignore_na:
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


def _stateful_depth(program: StreamingProgram) -> int:
    depths: list[int] = []
    for node in program.nodes:
        parent_depth = max((depths[child_id] for child_id in node.child_ids), default=0)
        depths.append(parent_depth + int(bool(node.op.is_stateful)))
    return max((depths[output_id] for output_id in program.outputs), default=0)


def _automatic_chunk_size(program: StreamingProgram) -> int:
    branch_plan = _detect_ewm_branch_plan(program)
    stateful, stateless = _program_counts(program)
    if branch_plan is not None:
        return 8_192 if branch_plan.breadth == 4 and _stateful_depth(program) < 8 else 4_096
    if stateful == 0:
        return 32_768
    if stateless == 0 and all(
        isinstance(node.op, (InputOp, LiteralOp, CacheOp, EwmOp))
        for node in program.nodes
    ):
        return 4_096
    return 65_536


def build_execution_plan(program: StreamingProgram) -> ExecutionPlan:
    regions = _build_regions(program)
    branch_plan = _detect_ewm_branch_plan(program)
    pad_safe = _padding_preserves_state(program)
    has_blocker = any(
        region.kind in {ExecutionKind.BLOCKER, ExecutionKind.HOST_NATIVE}
        for region in regions
    )
    if branch_plan is not None:
        strategy = "ewm_branch_batch"
    elif has_blocker:
        strategy = "legacy_boundary"
    else:
        strategy = "node_batch"
    return ExecutionPlan(
        regions=regions,
        chunk_size=_automatic_chunk_size(program),
        strategy=strategy,
        pad_safe=pad_safe,
        masked_tail=not pad_safe,
    )


def _stack_ewm_states(states) -> EwmState:
    return EwmState(
        value=jnp.stack(tuple(state.value for state in states), axis=0),
        weight=jnp.stack(tuple(state.weight for state in states), axis=0),
        initialized=jnp.stack(tuple(state.initialized for state in states), axis=0),
        count=jnp.stack(tuple(state.count for state in states), axis=0),
    )


def _unstack_ewm_state(state: EwmState, index: int) -> EwmState:
    return EwmState(
        value=state.value[index],
        weight=state.weight[index],
        initialized=state.initialized[index],
        count=state.count[index],
    )


def _scan_batched_ewm_level(ops, state: EwmState, values):
    spans = jnp.asarray(tuple(float(op.span) for op in ops), dtype=values.dtype)[:, None]
    alpha = 2.0 / (spans + 1.0)
    old_wt_factor = 1.0 - alpha
    min_periods = jnp.asarray(
        tuple(-1 if op.min_periods is None else int(round(float(op.min_periods))) for op in ops),
        dtype=jnp.int64,
    )[:, None]

    def step(carry: EwmState, x_t):
        value, weight, initialized, count = (
            carry.value,
            carry.weight,
            carry.initialized,
            carry.count,
        )
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


def _scan_affine_ewm_associative(op: EwmOp, state: EwmState, values):
    alpha = jnp.asarray(2.0 / (float(op.span) + 1.0), dtype=values.dtype)
    decay = 1.0 - alpha
    valid = jnp.isfinite(values)
    has = valid
    a = jnp.where(valid, decay, 1.0)
    b = jnp.where(valid, alpha * values, 0.0)
    uninitialized_value = jnp.where(valid, values, 0.0)
    counts = valid.astype(jnp.int64)

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

    has_prefix, a_prefix, b_prefix, u_prefix, count_prefix = jax.lax.associative_scan(
        combine,
        (has, a, b, uninitialized_value, counts),
        axis=0,
    )
    initialized0 = state.initialized
    values_initialized = a_prefix * state.value + b_prefix
    values_uninitialized = jnp.where(has_prefix, u_prefix, state.value)
    output_values = jnp.where(initialized0, values_initialized, values_uninitialized)
    initialized = initialized0 | has_prefix
    count = state.count + count_prefix
    enough = True if op.min_periods is None else count >= int(round(float(op.min_periods)))
    outputs = jnp.where(initialized & enough, output_values, jnp.nan)
    final_initialized = initialized[-1]
    final_state = EwmState(
        value=output_values[-1],
        weight=jnp.where(final_initialized, jnp.ones_like(state.weight), state.weight),
        initialized=final_initialized,
        count=count[-1],
    )
    return final_state, outputs


def _paired_ewm_consumer(program: StreamingProgram, node_id: int, consumers) -> int | None:
    node = program.nodes[node_id]
    if not isinstance(node.op, EwmOp) or len(node.child_ids) != 1:
        return None
    if node_id in program.outputs or node_id in program.cache_nodes:
        return None
    if len(consumers[node_id]) != 1:
        return None
    consumer_id = consumers[node_id][0]
    consumer = program.nodes[consumer_id]
    if not isinstance(consumer.op, EwmOp) or consumer.child_ids != (node_id,):
        return None
    return consumer_id


def _scan_ewm_pair(first_op, second_op, first_state, second_state, values):
    def step(carry, x_t):
        state1, state2 = carry
        state1, first_value = first_op.tick(state1, x_t)
        state2, second_value = second_op.tick(state2, first_value)
        return (state1, state2), second_value

    return jax.lax.scan(
        step,
        (first_state, second_state),
        values,
        unroll=1,
    )


def _scan_node(op, node_state, child_values):
    associative_min_width = int(
        os.environ.get("TRADING_DSL_JAX_FLAT_ASSOCIATIVE_EWM_MIN_WIDTH", "512")
    )
    if (
        isinstance(op, EwmOp)
        and op.span is not None
        and op.ignore_na
        and not op.adjust
        and len(child_values) == 1
        and child_values[0].shape[-1] >= associative_min_width
    ):
        return _scan_affine_ewm_associative(op, node_state, child_values[0])
    return op.scan_batch(node_state, *child_values)


def _evaluate_node_batch(runtime, state_leaves, inputs, batch_start, omitted=frozenset()):
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

        pair_id = _paired_ewm_consumer(runtime.program, node_id, consumers)
        if pair_id is not None and pair_id not in omitted:
            pair_node = runtime.program.nodes[pair_id]
            first_field = runtime.program.state_layout.node_fields[node_id]
            second_field = runtime.program.state_layout.node_fields[pair_id]
            source = values[node.child_ids[0]]
            (next_first, next_second), pair_value = _scan_ewm_pair(
                op,
                pair_node.op,
                state_leaves[first_field.index],
                state_leaves[second_field.index],
                source,
            )
            new_state[first_field.index] = next_first
            new_state[second_field.index] = next_second
            values[pair_id] = pair_value
            skipped.add(pair_id)
            continue

        child_values = tuple(values[child_id] for child_id in node.child_ids)
        field = runtime.program.state_layout.node_fields[node_id]
        node_state = None if field.index < 0 else state_leaves[field.index]
        next_state, value = (
            op.scan_batch_with_start(node_state, batch_start, *child_values)
            if isinstance(op, CacheOp)
            else _scan_node(op, node_state, child_values)
        )
        if field.index >= 0:
            new_state[field.index] = next_state
        values[node_id] = value
    return values, new_state


def _node_batch_branch_impl(runtime, plan: EwmBranchPlan, state_leaves, inputs, batch_start):
    values, new_state = _evaluate_node_batch(
        runtime,
        state_leaves,
        inputs,
        batch_start,
        plan.tail_nodes,
    )
    breadth = plan.breadth
    base = values[plan.base_node_id]
    branch_values = jnp.broadcast_to(
        base[:, None, :],
        (base.shape[0], breadth, base.shape[1]),
    )

    level_index = 0
    while level_index < len(plan.levels):
        first_level = plan.levels[level_index]
        first_fields = tuple(
            runtime.program.state_layout.node_fields[node_id]
            for node_id in first_level
        )
        first_states = _stack_ewm_states(
            tuple(state_leaves[field.index] for field in first_fields)
        )
        first_ops = tuple(runtime.program.nodes[node_id].op for node_id in first_level)

        if level_index + 1 < len(plan.levels):
            second_level = plan.levels[level_index + 1]
            second_fields = tuple(
                runtime.program.state_layout.node_fields[node_id]
                for node_id in second_level
            )
            second_states = _stack_ewm_states(
                tuple(state_leaves[field.index] for field in second_fields)
            )
            second_ops = tuple(runtime.program.nodes[node_id].op for node_id in second_level)

            def pair_step(carry, x_t):
                state1, state2 = carry
                state1, first_values = _batched_ewm_tick(first_ops, state1, x_t)
                state2, second_values = _batched_ewm_tick(second_ops, state2, first_values)
                return (state1, state2), second_values

            (next_first, next_second), branch_values = jax.lax.scan(
                pair_step,
                (first_states, second_states),
                branch_values,
                unroll=1,
            )
            for branch_i, field in enumerate(first_fields):
                new_state[field.index] = _unstack_ewm_state(next_first, branch_i)
            for branch_i, field in enumerate(second_fields):
                new_state[field.index] = _unstack_ewm_state(next_second, branch_i)
            level_index += 2
        else:
            next_first, branch_values = _scan_batched_ewm_level(
                first_ops,
                first_states,
                branch_values,
            )
            for branch_i, field in enumerate(first_fields):
                new_state[field.index] = _unstack_ewm_state(next_first, branch_i)
            level_index += 1

    outputs = tuple(branch_values[:, branch_i, :] for branch_i in range(breadth))
    cache_outputs = tuple(values[node_id] for node_id in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


def _batched_ewm_tick(ops, state: EwmState, values):
    spans = jnp.asarray(tuple(float(op.span) for op in ops), dtype=values.dtype)[:, None]
    alpha = 2.0 / (spans + 1.0)
    old_wt_factor = 1.0 - alpha
    min_periods = jnp.asarray(
        tuple(-1 if op.min_periods is None else int(round(float(op.min_periods))) for op in ops),
        dtype=jnp.int64,
    )[:, None]
    valid = jnp.isfinite(values)
    decayed_weight = jnp.where(
        state.initialized & valid,
        state.weight * old_wt_factor,
        state.weight,
    )
    normalized = (decayed_weight * state.value + alpha * values) / (decayed_weight + alpha)
    half = decayed_weight * state.value + (1.0 - decayed_weight) * values
    weighted = jnp.where(jnp.isclose(alpha, 0.5), half, normalized)
    next_value = jnp.where(valid, jnp.where(state.initialized, weighted, values), state.value)
    next_weight = jnp.where(valid, jnp.ones_like(decayed_weight), decayed_weight)
    next_initialized = state.initialized | valid
    next_count = state.count + valid.astype(jnp.int64)
    enough = (min_periods < 0) | (next_count >= min_periods)
    output = jnp.where(next_initialized & enough, next_value, jnp.nan)
    return EwmState(next_value, next_weight, next_initialized, next_count), output


def _planned_node_batch_impl(runtime, state_leaves, inputs, batch_start):
    branch_plan = _detect_ewm_branch_plan(runtime.program)
    if branch_plan is not None:
        return _node_batch_branch_impl(
            runtime,
            branch_plan,
            state_leaves,
            inputs,
            batch_start,
        )
    values, new_state = _evaluate_node_batch(
        runtime,
        state_leaves,
        inputs,
        batch_start,
    )
    outputs = tuple(values[node_id] for node_id in runtime.program.outputs)
    cache_outputs = tuple(values[node_id] for node_id in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


@partial(jax.jit, donate_argnums=(1,))
def _planned_node_batch_chunk(runtime, state_leaves, inputs, batch_start):
    return _planned_node_batch_impl(runtime, state_leaves, inputs, batch_start)


@jax.jit
def _planned_node_batch_chunk_nodonate(runtime, state_leaves, inputs, batch_start):
    return _planned_node_batch_impl(runtime, state_leaves, inputs, batch_start)


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
    outputs = tuple(values[node_id] for node_id in runtime.program.outputs)
    cache_outputs = tuple(values[node_id] for node_id in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


@partial(jax.jit, donate_argnums=(1,))
def _planned_masked_tail_chunk(
    runtime,
    state_leaves,
    inputs,
    valid_length,
    invalid_outputs,
    invalid_caches,
):
    row_indices = jnp.arange(inputs[0].shape[0], dtype=jnp.int32)

    def step(state, xs):
        rows, row_id = xs[:-1], xs[-1]
        return jax.lax.cond(
            row_id < valid_length,
            lambda _: _tick_program(runtime, state, rows),
            lambda _: (state, (invalid_outputs, invalid_caches)),
            operand=None,
        )

    return jax.lax.scan(
        step,
        state_leaves,
        (*inputs, row_indices),
        unroll=1,
    )


@jax.jit
def _planned_masked_tail_chunk_nodonate(
    runtime,
    state_leaves,
    inputs,
    valid_length,
    invalid_outputs,
    invalid_caches,
):
    row_indices = jnp.arange(inputs[0].shape[0], dtype=jnp.int32)

    def step(state, xs):
        rows, row_id = xs[:-1], xs[-1]
        return jax.lax.cond(
            row_id < valid_length,
            lambda _: _tick_program(runtime, state, rows),
            lambda _: (state, (invalid_outputs, invalid_caches)),
            operand=None,
        )

    return jax.lax.scan(
        step,
        state_leaves,
        (*inputs, row_indices),
        unroll=1,
    )


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


def _prepare_chunk(inputs, start: int, stop: int, chunk_size: int):
    return tuple(
        jnp.asarray(_pad_chunk(array, start, stop, chunk_size))
        for array in inputs
    )


def _output_names(program: StreamingProgram) -> tuple[str, ...]:
    names = getattr(program, "output_names", ())
    if names:
        return tuple(names)
    return tuple(f"output_{index}" for index in range(len(program.outputs)))


def _attach_output_names(program: StreamingProgram, names: Sequence[str]):
    object.__setattr__(program, "output_names", tuple(names))
    return program


def _allocate_host_outputs(runtime, n_steps: int, n_instruments: int, out_path):
    templates = tuple(
        _value_template(runtime.program.nodes[node_id].op, n_instruments)
        for node_id in runtime.program.outputs
    )
    multiple = len(templates) > 1
    arrays = []
    for name, template in zip(_output_names(runtime.program), templates, strict=True):
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
                root, extension = os.path.splitext(out_path)
                path = f"{root}.{name}{extension or '.memmap'}"
        else:
            raise ValueError("out_path must be False, True, None, or a filesystem path")
        arrays.append(
            np.memmap(
                path,
                mode="w+",
                dtype=np.asarray(template).dtype,
                shape=shape,
            )
        )
    return tuple(arrays)


def _materialize_pending(item, output_arrays):
    start, valid_length, values = item
    stop = start + valid_length
    for target, value in zip(output_arrays, values, strict=True):
        target[start:stop] = np.asarray(jax.device_get(value))[:valid_length]
        if isinstance(target, np.memmap):
            target.flush()


def _format_outputs(runtime, outputs):
    if len(outputs) == 1:
        return outputs[0]
    return dict(zip(_output_names(runtime.program), outputs, strict=True))


def _concatenate_device_outputs(runtime, chunks):
    outputs = tuple(
        jnp.concatenate(
            tuple(chunk_values[output_index][:valid_length] for valid_length, chunk_values in chunks),
            axis=0,
        )
        for output_index in range(len(runtime.program.outputs))
    )
    return _format_outputs(runtime, outputs)


def _run_planned_jax_batch(runtime, inputs, states, out_path):
    n_steps, n_instruments = inputs[0].shape[:2]
    state = runtime.init_state(n_instruments) if states is None else states
    plan = build_execution_plan(runtime.program)

    configured_chunk_size = int(
        os.environ.get("TRADING_DSL_JAX_FLAT_BATCH_CHUNK_SIZE", "0")
    )
    chunk_size = min(n_steps, configured_chunk_size or plan.chunk_size)
    max_in_flight = max(
        1,
        int(os.environ.get("TRADING_DSL_JAX_FLAT_MAX_IN_FLIGHT", "2")),
    )
    host_output = (
        bool(out_path)
        or _legacy._has_memmap_input(inputs)
    )
    output_arrays = (
        _allocate_host_outputs(runtime, n_steps, n_instruments, out_path)
        if host_output
        else ()
    )
    pending = deque()
    device_chunks = []
    first_chunk = True

    output_templates = tuple(
        _value_template(runtime.program.nodes[node_id].op, n_instruments)
        for node_id in runtime.program.outputs
    )
    cache_templates = tuple(
        _value_template(runtime.program.nodes[node_id].op, n_instruments)
        for node_id in runtime.program.cache_nodes
    )
    invalid_outputs = tuple(_invalid_like(value) for value in output_templates)
    invalid_caches = tuple(_invalid_like(value) for value in cache_templates)

    starts = tuple(range(0, n_steps, chunk_size))
    use_prefetch = _legacy._has_memmap_input(inputs)
    executor = ThreadPoolExecutor(max_workers=1) if use_prefetch else None
    future: Future | None = None
    if executor is not None and starts:
        first_start = starts[0]
        future = executor.submit(
            _prepare_chunk,
            inputs,
            first_start,
            min(first_start + chunk_size, n_steps),
            chunk_size,
        )

    try:
        for chunk_index, start in enumerate(starts):
            stop = min(start + chunk_size, n_steps)
            valid_length = stop - start
            if future is not None:
                chunk_inputs = future.result()
                next_index = chunk_index + 1
                if next_index < len(starts):
                    next_start = starts[next_index]
                    future = executor.submit(
                        _prepare_chunk,
                        inputs,
                        next_start,
                        min(next_start + chunk_size, n_steps),
                        chunk_size,
                    )
                else:
                    future = None
            else:
                chunk_inputs = _prepare_chunk(inputs, start, stop, chunk_size)

            donate = not (first_chunk and states is not None)
            if valid_length < chunk_size and plan.masked_tail:
                kernel = (
                    _planned_masked_tail_chunk
                    if donate
                    else _planned_masked_tail_chunk_nodonate
                )
                state, (chunk_outputs, _) = kernel(
                    runtime,
                    state,
                    chunk_inputs,
                    jnp.asarray(valid_length, dtype=jnp.int32),
                    invalid_outputs,
                    invalid_caches,
                )
            else:
                kernel = (
                    _planned_node_batch_chunk
                    if donate
                    else _planned_node_batch_chunk_nodonate
                )
                state, (chunk_outputs, _) = kernel(
                    runtime,
                    state,
                    chunk_inputs,
                    jnp.asarray(start, dtype=jnp.int64),
                )
            first_chunk = False

            if host_output:
                pending.append((start, valid_length, chunk_outputs))
                if len(pending) >= max_in_flight:
                    _materialize_pending(pending.popleft(), output_arrays)
            else:
                device_chunks.append((valid_length, chunk_outputs))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)

    if host_output:
        while pending:
            _materialize_pending(pending.popleft(), output_arrays)
        output = _format_outputs(runtime, output_arrays)
    else:
        output = _concatenate_device_outputs(runtime, tuple(device_chunks))

    jax.block_until_ready(state)
    return state, output


JaxFlatRuntime = _legacy.JaxFlatRuntime
StreamingProgram = _legacy.StreamingProgram
_LEGACY_RUN_BATCH_ONCE = JaxFlatRuntime._run_batch_once
_LEGACY_DOUBLE_GROUPBY_CAPACITIES = _legacy._double_groupby_capacities


def _planned_run_batch_once(self, inputs, states=None, out_path: str | bool = False):
    inputs = _legacy._normalize_batch_inputs(self, inputs)
    if not inputs:
        raise ValueError("run_batch requires at least one input array")

    n_steps, n_instruments = inputs[0].shape[:2]
    if any(array.shape[:2] != (n_steps, n_instruments) for array in inputs[1:]):
        raise ValueError("All inputs must share aligned shape (time, n_instruments)")

    plan = build_execution_plan(self.program)
    if self.program.cache_nodes or plan.strategy == "legacy_boundary":
        return _LEGACY_RUN_BATCH_ONCE(self, inputs, states, out_path)

    if (
        self.cpp
        and len(self.program.outputs) == 1
        and states is None
        and not out_path
    ):
        try:
            from trading_dsl_engine.jax_flat.engine_cpp import _try_cpp_hybrid_batch
        except Exception as exc:
            _legacy._warn_cpp_fallback(
                self,
                "C++ jax_flat accelerator unavailable "
                f"({type(exc).__name__}: {exc}); falling back to JAX-flat",
            )
        else:
            hybrid = _try_cpp_hybrid_batch(
                self,
                inputs,
                _legacy._CPP_ACCELERATOR_CACHE,
                _legacy._warn_cpp_fallback,
            )
            if hybrid is not None:
                return hybrid

    return _run_planned_jax_batch(self, inputs, states, out_path)


def _planned_double_groupby_capacities(runtime):
    output_names = _output_names(runtime.program)
    next_runtime = _LEGACY_DOUBLE_GROUPBY_CAPACITIES(runtime)
    if next_runtime is not runtime:
        _attach_output_names(next_runtime.program, output_names)
    return next_runtime


JaxFlatRuntime._run_batch_once = _planned_run_batch_once
_legacy._double_groupby_capacities = _planned_double_groupby_capacities


def compile_formula(
    formula,
    dsl_registry=None,
    cpp: bool = True,
    metadata=None,
    type_relations=(),
    runtimes=None,
):
    runtime = _legacy.compile_formula(
        formula,
        dsl_registry=dsl_registry,
        cpp=cpp,
        metadata=metadata,
        type_relations=type_relations,
        runtimes=runtimes,
    )
    _attach_output_names(runtime.program, ("output",))
    return runtime


def compile_features(
    formulas: Mapping[str, Any],
    *,
    dsl_registry=None,
    cpp: bool = False,
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
        expr = _legacy._expand_dsl(
            expr,
            dsl_registry or _legacy.DEFAULT_DSL_REGISTRY,
        )
        expr = _legacy._normalize_static_jax_flat_kwargs(expr)
        outputs.append(
            _legacy._compile_node(
                expr,
                memo,
                nodes,
                input_names,
                external_cache_names,
            )
        )
        names.append(str(name))

    node_tuple = tuple(nodes)
    cache_nodes = tuple(
        node_id
        for node_id, node in enumerate(node_tuple)
        if isinstance(node.op, CacheOp)
    )
    cache_key_by_node = {
        node_id: key
        for key, node_id in memo.items()
        if key[0] == "call" and key[1] == "cache"
    }
    cache_expr_keys = tuple(
        cache_key_by_node[node_id][2][0]
        for node_id in cache_nodes
    )
    program = StreamingProgram(
        nodes=node_tuple,
        outputs=tuple(outputs),
        input_names=tuple(input_names),
        state_layout=_legacy._build_state_layout(node_tuple),
        metadata=None,
        cache_nodes=cache_nodes,
        cache_expr_keys=cache_expr_keys,
        external_cache_inputs=external_cache_values or None,
    )
    _attach_output_names(program, names)
    return JaxFlatRuntime(program=program, cpp=cpp)


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
