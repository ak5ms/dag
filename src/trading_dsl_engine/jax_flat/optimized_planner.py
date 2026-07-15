from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import os

import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat import optimized as _optimized
from trading_dsl_engine.jax_flat.ops import CacheOp, CumsumOp, EwmOp, EwmState, InputOp, LiteralOp


# Preserve undecorated implementations before installing the CPU planner hooks.
_BASE_NODE_BATCH_CHUNK = _optimized._node_batch_chunk
_BASE_NODE_BATCH_IMPL = getattr(_BASE_NODE_BATCH_CHUNK, "__wrapped__", _BASE_NODE_BATCH_CHUNK)
_BASE_RUN_BATCH_ONCE = _optimized.OptimizedJaxFlatRuntime._run_batch_once
_BASE_EXECUTION_STRATEGY = _optimized.OptimizedJaxFlatRuntime.execution_strategy


@dataclass(frozen=True)
class EwmBranchPlan:
    """Same-depth EWM tails that can be evaluated over one breadth axis."""

    base_node_id: int
    levels: tuple[tuple[int, ...], ...]
    tail_nodes: frozenset[int]

    @property
    def breadth(self) -> int:
        return len(self.levels[0])

    @property
    def depth(self) -> int:
        return len(self.levels)


def _replace_parallel_ops_cpu(program):
    """Keep associative EWM out of the default CPU plan.

    The tree-prefix lowering remains available in optimized.py for experiments,
    but it was slower and allocated more temporary memory for the aligned CPU
    shapes measured by the benchmark suite.
    """

    return program


def _node_batch_pad_safe(program) -> bool:
    for node in program.nodes:
        op = node.op
        if not op.is_stateful:
            continue
        if isinstance(op, CumsumOp):
            continue
        if isinstance(op, EwmOp) and op.ignore_na:
            continue
        return False
    return True


def _choose_cpu_strategy(program, requested: str) -> str:
    if requested not in {"auto", "compound", "node_batch"}:
        raise ValueError("strategy must be 'auto', 'compound', or 'node_batch'")
    if requested != "auto":
        return requested

    # A full-DAG compound scan minimizes temporary arrays but leaves narrow time
    # recurrences effectively single-core. Cache-sized node batches preserve the
    # wider CPU kernels; homogeneous EWM branches are packed below.
    return "node_batch" if _node_batch_pad_safe(program) else "compound"


def _consumer_lists(program):
    consumers: list[list[int]] = [[] for _ in program.nodes]
    for consumer_id, node in enumerate(program.nodes):
        for child_id in node.child_ids:
            consumers[child_id].append(consumer_id)
    return tuple(tuple(items) for items in consumers)


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
    path_reversed: list[int] = []
    node_id = output_id
    while _compatible_branch_ewm(program.nodes[node_id]):
        path_reversed.append(node_id)
        node_id = program.nodes[node_id].child_ids[0]
    return node_id, tuple(reversed(path_reversed))


def _detect_ewm_branch_plan(program) -> EwmBranchPlan | None:
    """Find multiple output EWM chains with a common already-computed prefix.

    The lowering is conservative. Every output must end in a same-depth chain of
    static-span, ignore-na, unadjusted EWMs. Tail nodes may only feed the next
    node in their branch and may not be cache roots.
    """

    if len(program.outputs) < 2:
        return None

    walked = tuple(_ewm_path(program, output_id) for output_id in program.outputs)
    raw_bases = tuple(item[0] for item in walked)
    paths = tuple(item[1] for item in walked)
    if not paths or any(not path for path in paths):
        return None
    if len(set(raw_bases)) != 1:
        return None

    common = 0
    shortest = min(map(len, paths))
    while common < shortest and all(path[common] == paths[0][common] for path in paths[1:]):
        common += 1

    base_node_id = paths[0][common - 1] if common else raw_bases[0]
    tails = tuple(path[common:] for path in paths)
    if any(not tail for tail in tails):
        return None
    if len({len(tail) for tail in tails}) != 1:
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

    return EwmBranchPlan(base_node_id=base_node_id, levels=levels, tail_nodes=tail_nodes)


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
        value, weight, initialized, count = carry.value, carry.weight, carry.initialized, carry.count
        valid = jnp.isfinite(x_t)
        decayed_weight = jnp.where(initialized & valid, weight * old_wt_factor, weight)
        normalized = (decayed_weight * value + alpha * x_t) / (decayed_weight + alpha)
        alpha_half = jnp.isclose(alpha, 0.5)
        half_alpha_weighted = decayed_weight * value + (1.0 - decayed_weight) * x_t
        weighted = jnp.where(alpha_half, half_alpha_weighted, normalized)
        next_value = jnp.where(valid, jnp.where(initialized, weighted, x_t), value)
        next_weight = jnp.where(valid, jnp.ones_like(decayed_weight), decayed_weight)
        next_initialized = initialized | valid
        next_count = count + valid.astype(jnp.int64)
        enough = (min_periods < 0) | (next_count >= min_periods)
        out = jnp.where(next_initialized & enough, next_value, jnp.nan)
        return EwmState(next_value, next_weight, next_initialized, next_count), out

    return jax.lax.scan(step, state, values, unroll=1)


def _node_batch_chunk_ewm_branches_impl(runtime, plan: EwmBranchPlan, state_leaves, inputs, batch_start):
    n_steps = inputs[0].shape[0]
    values = [jnp.asarray(0.0)] * len(runtime.program.nodes)
    new_state = list(state_leaves)

    # Evaluate the shared DAG while omitting the branch-tail nodes. Detection
    # guarantees that no non-tail node consumes an omitted value.
    for node_id, node in enumerate(runtime.program.nodes):
        if node_id in plan.tail_nodes:
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

    breadth = plan.breadth
    base = values[plan.base_node_id]
    branch_values = jnp.broadcast_to(base[:, None, :], (base.shape[0], breadth, base.shape[1]))

    for level in plan.levels:
        fields = tuple(runtime.program.state_layout.node_fields[node_id] for node_id in level)
        level_states = tuple(state_leaves[field.index] for field in fields)
        stacked_state = _stack_ewm_states(level_states)
        ops = tuple(runtime.program.nodes[node_id].op for node_id in level)
        next_stacked, branch_values = _scan_batched_ewm_level(ops, stacked_state, branch_values)
        for branch_i, field in enumerate(fields):
            new_state[field.index] = _unstack_ewm_state(next_stacked, branch_i)

    outputs = tuple(branch_values[:, branch_i, :] for branch_i in range(breadth))
    cache_outputs = tuple(values[node_id] for node_id in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


def _paired_ewm_consumer(program, node_id: int, consumers) -> int | None:
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


def _scan_ewm_pair(first_op, second_op, first_state, second_state, inputs):
    def step(carry, x_t):
        state1, state2 = carry
        state1, y1_t = first_op.tick(state1, x_t)
        state2, y2_t = second_op.tick(state2, y1_t)
        return (state1, state2), y2_t

    return jax.lax.scan(step, (first_state, second_state), inputs, unroll=1)


def _node_batch_chunk_pair_fused_impl(runtime, state_leaves, inputs, batch_start):
    n_steps = inputs[0].shape[0]
    values = [jnp.asarray(0.0)] * len(runtime.program.nodes)
    new_state = list(state_leaves)
    consumers = _consumer_lists(runtime.program)
    skipped: set[int] = set()

    for node_id, node in enumerate(runtime.program.nodes):
        if node_id in skipped:
            continue
        op = node.op
        if isinstance(op, InputOp):
            values[node_id] = inputs[op.input_index]
            continue
        if isinstance(op, LiteralOp):
            values[node_id] = jnp.full((n_steps,), op.value, dtype=jnp.float64)
            continue

        pair_id = _paired_ewm_consumer(runtime.program, node_id, consumers)
        if pair_id is not None:
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
            else op.scan_batch(node_state, *child_values)
        )
        if field.index >= 0:
            new_state[field.index] = next_state
        values[node_id] = value

    outputs = tuple(values[node_id] for node_id in runtime.program.outputs)
    cache_outputs = tuple(values[node_id] for node_id in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


@partial(jax.jit, donate_argnums=(1,))
def _node_batch_chunk_cpu(runtime, state_leaves, inputs, batch_start: jax.Array):
    plan = _detect_ewm_branch_plan(runtime.program)
    if plan is not None:
        return _node_batch_chunk_ewm_branches_impl(runtime, plan, state_leaves, inputs, batch_start)
    if os.environ.get("TRADING_DSL_JAX_FLAT_PAIR_FUSION", "0") == "1":
        return _node_batch_chunk_pair_fused_impl(runtime, state_leaves, inputs, batch_start)
    return _BASE_NODE_BATCH_IMPL(runtime, state_leaves, inputs, batch_start)


def _program_counts(program):
    stateful = 0
    stateless = 0
    for node in program.nodes:
        op = node.op
        if isinstance(op, (InputOp, LiteralOp, CacheOp)):
            continue
        if op.is_stateful:
            stateful += 1
        else:
            stateless += 1
    return stateful, stateless


def _cpu_chunk_size(program) -> int:
    plan = _detect_ewm_branch_plan(program)
    stateful, stateless = _program_counts(program)
    if plan is not None:
        total_depth = _optimized._stateful_depth(program)
        return 8_192 if plan.breadth == 4 and total_depth < 8 else 4_096
    if stateful == 0:
        return 32_768
    if stateless == 0 and all(
        isinstance(node.op, (InputOp, LiteralOp, CacheOp, EwmOp)) for node in program.nodes
    ):
        return 4_096
    return 65_536


def _run_batch_once_cpu(self, runtime, inputs, states, out_path):
    auto_tiling = os.environ.get("TRADING_DSL_JAX_FLAT_AUTO_TILE", "1") != "0"
    if auto_tiling and int(self.chunk_size) == int(_optimized._DEFAULT_CHUNK_SIZE):
        tuned = replace(self, chunk_size=_cpu_chunk_size(runtime.program))
        return _BASE_RUN_BATCH_ONCE(tuned, runtime, inputs, states, out_path)
    return _BASE_RUN_BATCH_ONCE(self, runtime, inputs, states, out_path)


def _execution_strategy_cpu(self) -> str:
    if self.strategy == "auto" and _detect_ewm_branch_plan(self.program) is not None:
        return "ewm_branch_batch"
    return _BASE_EXECUTION_STRATEGY(self)


_optimized._replace_parallel_ops = _replace_parallel_ops_cpu
_optimized._choose_strategy = _choose_cpu_strategy
_optimized._node_batch_chunk = _node_batch_chunk_cpu
_optimized.OptimizedJaxFlatRuntime._run_batch_once = _run_batch_once_cpu
_optimized.OptimizedJaxFlatRuntime.execution_strategy = _execution_strategy_cpu
