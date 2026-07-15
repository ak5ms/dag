from __future__ import annotations

"""Carry- and tail-safety policies for the planned executor."""

from dataclasses import replace

from trading_dsl_engine.jax_flat import engine


_DONATED_NODE_BATCH = engine._planned_node_batch_chunk
_DONATED_MASKED_TAIL = engine._planned_masked_tail_chunk
_BASE_BUILD_EXECUTION_PLAN = engine.build_execution_plan


def donation_safe(program) -> bool:
    """Donate only simple carries with one stateful slot."""
    return program.state_layout.total_leaves <= 1


def build_execution_plan(program):
    """Require an explicitly masked tail for every stateful DAG.

    NaN padding is not compositionally safe. For example, an ``ignore_na`` EWM
    holds and emits its last finite value on a padded row, so a downstream EWM
    would still update. Masking the whole DAG prevents every state transition
    after ``valid_length`` while retaining a fixed compiled chunk shape.
    """
    plan = _BASE_BUILD_EXECUTION_PLAN(program)
    if program.state_layout.total_leaves:
        return replace(plan, pad_safe=False, masked_tail=True)
    return replace(plan, pad_safe=True, masked_tail=False)


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


engine.build_execution_plan = build_execution_plan
engine._planned_node_batch_chunk = _node_batch_chunk
engine._planned_masked_tail_chunk = _masked_tail_chunk
