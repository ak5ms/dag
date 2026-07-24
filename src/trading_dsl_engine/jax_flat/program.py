"""JAX-flat streaming program IR and groupby inner-graph wrapper.

Compile-time lowering produces immutable ``StreamingProgram`` graphs. Runtime
execution (live tick and batch scan) consumes these types but does not mutate
them. ``StateLayout`` maps each DAG node to a state-leaf index (``-1`` for
stateless nodes).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import os

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.metadata import FormulaMetadata
from trading_dsl_engine.jax_flat.ops import InputOp, LiteralOp, Op

class MemmapPathTracker:
    def __init__(self):
        self.memmaps: list[np.memmap] = []

    def add(self, memmap: np.memmap) -> None:
        self.memmaps.append(memmap)

    def cleanup(self) -> None:
        memmaps = list(self.memmaps)
        self.memmaps.clear()
        for memmap in memmaps:
            path = memmap.filename
            mapped = getattr(memmap, "_mmap", None)
            try:
                memmap.flush()
                if mapped is not None:
                    mapped.close()
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError:
                pass

    def __del__(self):
        self.cleanup()


class CacheWriteTarget:
    def __init__(self, array: np.ndarray):
        self.array = array

    def write(self, start, value) -> None:
        start_i = int(np.asarray(start))
        value_np = np.asarray(value)
        self.array[start_i : start_i + value_np.shape[0]] = value_np
        if isinstance(self.array, np.memmap):
            self.array.flush()


class JitCompileTracker:
    """Mutable, identity-hashable runtime-only JIT trace counter."""

    def __init__(self):
        self.count = 0

    def record(self) -> None:
        self.count += 1

    def reset(self) -> None:
        self.count = 0


@dataclass(frozen=True)
class DagNode:
    op: Any
    child_ids: tuple[int, ...]


@dataclass(frozen=True)
class StateFieldRef:
    index: int


@dataclass(frozen=True)
class StateLayout:
    node_fields: tuple[StateFieldRef, ...]
    total_leaves: int


@dataclass(frozen=True)
class StreamingProgram:
    nodes: tuple[DagNode, ...]
    outputs: tuple[int, ...]
    input_names: tuple[str, ...]
    state_layout: StateLayout
    metadata: FormulaMetadata | None = None
    cache_nodes: tuple[int, ...] = ()
    cache_expr_keys: tuple[tuple[Any, ...], ...] = ()
    external_cache_inputs: dict[str, np.ndarray] | None = None


@dataclass(frozen=True)
class InnerGraphOp(Op):
    """A groupby-local operator DAG.

    The flat runtime normally stores state per top-level DAG node. A groupby RHS,
    however, must keep any nested stateful operators inside the group bucket. This
    op wraps the RHS sub-DAG so GroupByOp can allocate one complete RHS state per
    universe/dynamic-key slot.
    """

    nodes: tuple[DagNode, ...]
    output_id: int
    state_layout: StateLayout
    n_inputs: int
    is_stateful: bool = False
    output_kind: str = "vector"

    def init_state(self, sample: jax.Array):
        states = []
        for node in self.nodes:
            if node.op.is_stateful:
                states.append(node.op.init_state(sample))
        return tuple(states)

    def tick(self, state_leaves, *input_values: jax.Array):
        values: list[jax.Array] = [jnp.array(0.0)] * len(self.nodes)
        new_state = list(()) if state_leaves is None else list(state_leaves)

        for idx, node in enumerate(self.nodes):
            op = node.op
            if isinstance(op, InputOp):
                values[idx] = input_values[op.input_index]
                continue
            if isinstance(op, LiteralOp):
                values[idx] = jnp.asarray(op.value, dtype=jnp.float64)
                continue

            child_values = tuple(values[cid] for cid in node.child_ids)
            field = self.state_layout.node_fields[idx]
            node_state = None if field.index < 0 else state_leaves[field.index]
            next_state, value = op.tick(node_state, *child_values)
            if field.index >= 0:
                new_state[field.index] = next_state
            values[idx] = value

        return tuple(new_state), values[self.output_id]

    def scan_batch(self, state_leaves, *input_sequences: jax.Array):
        n_steps = input_sequences[0].shape[0]
        values: list[Any] = [jnp.array(0.0)] * len(self.nodes)
        new_state = list(()) if state_leaves is None else list(state_leaves)

        for idx, node in enumerate(self.nodes):
            op = node.op
            if isinstance(op, InputOp):
                values[idx] = input_sequences[op.input_index]
                continue
            if isinstance(op, LiteralOp):
                values[idx] = jnp.full((n_steps,), op.value, dtype=jnp.float64)
                continue

            child_values = tuple(values[cid] for cid in node.child_ids)
            field = self.state_layout.node_fields[idx]
            node_state = None if field.index < 0 else state_leaves[field.index]
            next_state, value = op.scan_batch(node_state, *child_values)
            if field.index >= 0:
                new_state[field.index] = next_state
            values[idx] = value

        next_states = tuple(new_state) if state_leaves is not None else None
        return next_states, values[self.output_id]

