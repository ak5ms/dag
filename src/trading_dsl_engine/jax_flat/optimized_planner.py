from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat import optimized as _optimized
from trading_dsl_engine.jax_flat.ops import CacheOp, CumsumOp, EwmOp, InputOp, LiteralOp


def _replace_parallel_ops_cpu(program):
    """Keep associative EWM out of the default CPU plan.

    The lowering remains available in optimized.py for experimentation, but local
    CPU HLO/runtime measurements show that the tree scan is slower and allocates
    substantially more temporary memory than the sequential scan for the current
    aligned feature shapes.
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

    # The full-DAG compound scan minimizes temporary arrays, but a narrow
    # time recurrence underutilizes a many-core CPU. The default therefore uses
    # node-batch execution, augmented below with shallow two-node EWM fusion.
    return "node_batch" if _node_batch_pad_safe(program) else "compound"


def _consumer_lists(program):
    consumers: list[list[int]] = [[] for _ in program.nodes]
    for consumer_id, node in enumerate(program.nodes):
        for child_id in node.child_ids:
            consumers[child_id].append(consumer_id)
    return tuple(tuple(items) for items in consumers)


def _paired_ewm_consumer(program, node_id: int, consumers) -> int | None:
    """Return a sole downstream EWM that can be fused with node_id.

    Pair fusion is deliberately conservative: both EWMs must use a single data
    child (static span/min_periods), and the first result must not be observable
    as a root or cache value. This preserves exact tick semantics for all EWM
    modes while avoiding a full time-axis intermediate between the pair.
    """
    node = program.nodes[node_id]
    if not isinstance(node.op, EwmOp) or len(node.child_ids) != 1:
        return None
    if node_id in program.outputs or node_id in program.cache_nodes:
        return None
    if len(consumers[node_id]) != 1:
        return None

    consumer_id = consumers[node_id][0]
    consumer = program.nodes[consumer_id]
    if not isinstance(consumer.op, EwmOp):
        return None
    if consumer.child_ids != (node_id,):
        return None
    return consumer_id


def _scan_ewm_pair(first_op, second_op, first_state, second_state, inputs):
    def step(carry, x_t):
        state1, state2 = carry
        state1, y1_t = first_op.tick(state1, x_t)
        state2, y2_t = second_op.tick(state2, y1_t)
        return (state1, state2), y2_t

    return jax.lax.scan(
        step,
        (first_state, second_state),
        inputs,
        unroll=1,
    )


@partial(jax.jit, donate_argnums=(1,))
def _node_batch_chunk_pair_fused(runtime, state_leaves, inputs, batch_start: jax.Array):
    """Node-batch lowering with conservative two-EWM recurrence fusion.

    A depth-D EWM chain now creates ceil(D / 2) scans and approximately half as
    many block-sized intermediates. Independent/stateless nodes retain their
    specialized scan_batch or vectorized implementations.
    """
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
            first_state = state_leaves[first_field.index]
            second_state = state_leaves[second_field.index]
            source = values[node.child_ids[0]]
            (next_first, next_second), pair_value = _scan_ewm_pair(
                op,
                pair_node.op,
                first_state,
                second_state,
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


_optimized._replace_parallel_ops = _replace_parallel_ops_cpu
_optimized._choose_strategy = _choose_cpu_strategy
_optimized._node_batch_chunk = _node_batch_chunk_pair_fused
