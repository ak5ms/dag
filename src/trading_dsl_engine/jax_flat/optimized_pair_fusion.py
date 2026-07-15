from __future__ import annotations

from functools import partial
import os

import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat import optimized as _optimized
from trading_dsl_engine.jax_flat import optimized_planner as _planner
from trading_dsl_engine.jax_flat.ops import CacheOp, EwmOp, EwmState, InputOp, LiteralOp


_BASE_EXECUTION_STRATEGY = _optimized.OptimizedJaxFlatRuntime.execution_strategy


def _level_params(ops, dtype):
    spans = jnp.asarray(tuple(float(op.span) for op in ops), dtype=dtype)[:, None]
    alpha = 2.0 / (spans + 1.0)
    min_periods = jnp.asarray(
        tuple(-1 if op.min_periods is None else int(round(float(op.min_periods))) for op in ops),
        dtype=jnp.int64,
    )[:, None]
    return alpha, 1.0 - alpha, min_periods


def _batched_ewm_tick(params, state: EwmState, x_t):
    alpha, old_wt_factor, min_periods = params
    value, weight, initialized, count = state.value, state.weight, state.initialized, state.count
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


def _scan_batched_ewm_pair(ops1, ops2, state1, state2, values):
    params1 = _level_params(ops1, values.dtype)
    params2 = _level_params(ops2, values.dtype)

    def step(carry, x_t):
        first_state, second_state = carry
        first_state, first_value = _batched_ewm_tick(params1, first_state, x_t)
        second_state, second_value = _batched_ewm_tick(params2, second_state, first_value)
        return (first_state, second_state), second_value

    return jax.lax.scan(step, (state1, state2), values, unroll=1)


def _node_batch_chunk_ewm_branches_pair_impl(runtime, plan, state_leaves, inputs, batch_start):
    n_steps = inputs[0].shape[0]
    values = [jnp.asarray(0.0)] * len(runtime.program.nodes)
    new_state = list(state_leaves)

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

    level_i = 0
    while level_i < len(plan.levels):
        level1 = plan.levels[level_i]
        fields1 = tuple(runtime.program.state_layout.node_fields[node_id] for node_id in level1)
        state1 = _planner._stack_ewm_states(tuple(state_leaves[field.index] for field in fields1))
        ops1 = tuple(runtime.program.nodes[node_id].op for node_id in level1)

        if level_i + 1 < len(plan.levels):
            level2 = plan.levels[level_i + 1]
            fields2 = tuple(runtime.program.state_layout.node_fields[node_id] for node_id in level2)
            state2 = _planner._stack_ewm_states(tuple(state_leaves[field.index] for field in fields2))
            ops2 = tuple(runtime.program.nodes[node_id].op for node_id in level2)
            (next1, next2), branch_values = _scan_batched_ewm_pair(
                ops1,
                ops2,
                state1,
                state2,
                branch_values,
            )
            for branch_i, field in enumerate(fields1):
                new_state[field.index] = _planner._unstack_ewm_state(next1, branch_i)
            for branch_i, field in enumerate(fields2):
                new_state[field.index] = _planner._unstack_ewm_state(next2, branch_i)
            level_i += 2
            continue

        next1, branch_values = _planner._scan_batched_ewm_level(ops1, state1, branch_values)
        for branch_i, field in enumerate(fields1):
            new_state[field.index] = _planner._unstack_ewm_state(next1, branch_i)
        level_i += 1

    outputs = tuple(branch_values[:, branch_i, :] for branch_i in range(breadth))
    cache_outputs = tuple(values[node_id] for node_id in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


def _pure_ewm_program(program) -> bool:
    saw_ewm = False
    for node in program.nodes:
        op = node.op
        if isinstance(op, (InputOp, LiteralOp, CacheOp)):
            continue
        if not isinstance(op, EwmOp):
            return False
        saw_ewm = True
    return saw_ewm


@partial(jax.jit, donate_argnums=(1,))
def _node_batch_chunk_cpu_pair_fused(runtime, state_leaves, inputs, batch_start):
    pair_fusion = os.environ.get("TRADING_DSL_JAX_FLAT_PAIR_FUSION", "1") != "0"
    plan = _planner._detect_ewm_branch_plan(runtime.program)
    if plan is not None:
        if pair_fusion:
            return _node_batch_chunk_ewm_branches_pair_impl(
                runtime,
                plan,
                state_leaves,
                inputs,
                batch_start,
            )
        return _planner._node_batch_chunk_ewm_branches_impl(
            runtime,
            plan,
            state_leaves,
            inputs,
            batch_start,
        )
    if pair_fusion and _pure_ewm_program(runtime.program):
        return _planner._node_batch_chunk_pair_fused_impl(runtime, state_leaves, inputs, batch_start)
    return _planner._BASE_NODE_BATCH_IMPL(runtime, state_leaves, inputs, batch_start)


def _execution_strategy_pair_fused(self) -> str:
    if self.strategy == "auto":
        plan = _planner._detect_ewm_branch_plan(self.program)
        if plan is not None:
            return "ewm_branch_pair_batch"
        if _pure_ewm_program(self.program):
            return "pair_fused_node_batch"
    return _BASE_EXECUTION_STRATEGY(self)


_optimized._node_batch_chunk = _node_batch_chunk_cpu_pair_fused
_optimized.OptimizedJaxFlatRuntime.execution_strategy = _execution_strategy_pair_fused
