from __future__ import annotations

"""Runtime policies that depend on compiled carry layout.

JAX donation is enabled only when a chunk has at most one stateful slot. Fused
and branch-batched recurrences return multiple live state buffers; donating that
carry can alias distinct returned slots on CPU even when the chunk output itself
is numerically correct.
"""

from trading_dsl_engine.jax_flat import engine


_DONATED_NODE_BATCH = engine._planned_node_batch_chunk
_DONATED_MASKED_TAIL = engine._planned_masked_tail_chunk


def donation_safe(program) -> bool:
    return program.state_layout.total_leaves <= 1


def _node_batch_chunk(runtime, state_leaves, inputs, batch_start):
    if donation_safe(runtime.program):
        return _DONATED_NODE_BATCH(runtime, state_leaves, inputs, batch_start)
    return engine._planned_node_batch_chunk_nodonate(
        runtime,
        state_leaves,
        inputs,
        batch_start,
    )


def _masked_tail_chunk(
    runtime,
    state_leaves,
    inputs,
    valid_length,
    invalid_outputs,
    invalid_caches,
):
    if donation_safe(runtime.program):
        return _DONATED_MASKED_TAIL(
            runtime,
            state_leaves,
            inputs,
            valid_length,
            invalid_outputs,
            invalid_caches,
        )
    return engine._planned_masked_tail_chunk_nodonate(
        runtime,
        state_leaves,
        inputs,
        valid_length,
        invalid_outputs,
        invalid_caches,
    )


engine._planned_node_batch_chunk = _node_batch_chunk
engine._planned_masked_tail_chunk = _masked_tail_chunk
