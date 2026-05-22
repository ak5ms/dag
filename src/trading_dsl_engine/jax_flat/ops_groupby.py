from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat.ops import Op

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GroupByState:
    keys: tuple[jax.Array, ...]
    occupied: tuple[jax.Array, ...]
    inner_state: tuple[Any, ...]


def _tree_broadcast_slots(tree: Any, capacity: int):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(jnp.asarray(leaf), (capacity,) + jnp.asarray(leaf).shape),
        tree,
    )


def _tree_take_slot(tree: Any, slot: jax.Array):
    return jax.tree_util.tree_map(lambda leaf: jnp.take(leaf, slot, axis=0), tree)


def _tree_set_slot(tree: Any, slot: jax.Array, value: Any):
    return jax.tree_util.tree_map(lambda dst, src: dst.at[slot].set(src), tree, value)


def _key_matches(stored_keys: jax.Array, occupied: jax.Array, key: jax.Array) -> jax.Array:
    equal_or_both_nan = (stored_keys == key[None, :]) | (jnp.isnan(stored_keys) & jnp.isnan(key[None, :]))
    return occupied & jnp.all(equal_or_both_nan, axis=1)

def _empty_output_like(template: Any, width: int):
    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        suffix = leaf.shape[1:] if leaf.ndim > 0 else ()
        return jnp.full((width,) + suffix, jnp.nan, dtype=leaf.dtype)

    return jax.tree_util.tree_map(alloc, template)


def _first_available_slot(matches: jax.Array, occupied: jax.Array) -> jax.Array:
    has_match = jnp.any(matches)
    # If capacity is exhausted, argmax(~occupied) is 0. This preserves a valid
    # compiled program; callers should size capacity to the expected key domain.
    first_free = jnp.argmax(~occupied)
    return jnp.where(has_match, jnp.argmax(matches), first_free)


def _slice_group_arg(value: jax.Array, idx: jax.Array) -> jax.Array:
    value = jnp.asarray(value)
    if value.ndim == 0:
        return value
    return jnp.take(value, idx, axis=0)


def _mask_group_arg(value: jax.Array, idx: jax.Array, mask: jax.Array) -> jax.Array:
    group_value = _slice_group_arg(value, idx)
    if jnp.asarray(group_value).ndim == 0:
        return group_value
    mask_shape = mask.shape + (1,) * (group_value.ndim - 1)
    return jnp.where(jnp.reshape(mask, mask_shape), group_value, jnp.nan)


def _same_key_vector(left: jax.Array, right: jax.Array) -> jax.Array:
    equal_or_both_nan = (left == right) | (jnp.isnan(left) & jnp.isnan(right))
    return jnp.all(equal_or_both_nan)


def _group_output_like(template: Any, group_width: int):
    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        suffix = leaf.shape[1:] if leaf.ndim > 0 else ()
        return jnp.full((group_width,) + suffix, jnp.nan, dtype=leaf.dtype)

    return jax.tree_util.tree_map(alloc, template)


def _align_group_output(value: Any, group_width: int):
    def align_leaf(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.ndim == 0:
            return jnp.broadcast_to(leaf, (group_width,))
        if leaf.shape[0] == 1:
            return jnp.broadcast_to(leaf[0], (group_width,) + leaf.shape[1:])
        return leaf

    return jax.tree_util.tree_map(align_leaf, value)


def _mask_group_output(old: Any, new: Any, mask: jax.Array):
    def mask_leaf(dst, src):
        mask_shape = mask.shape + (1,) * (jnp.asarray(src).ndim - 1)
        return jnp.where(jnp.reshape(mask, mask_shape), src, dst)

    return jax.tree_util.tree_map(mask_leaf, old, new)


def _scatter_group_output(out: Any, idx: jax.Array, group_value: Any):
    return jax.tree_util.tree_map(lambda dst, src: dst.at[idx].set(src), out, group_value)


@dataclass(frozen=True)
class GroupByOp(Op):
    """Dynamic key x static universe groupby.

    For each universe group, evaluate the inner op once per distinct dynamic-key
    tuple present in that group.

    The inner op receives the full group-shaped input with nonmatching columns
    masked to NaN. Only matching output positions are scattered back.

    This matches the numba backend contract: absent key/universe combinations
    are skipped, not ticked with an all-NaN frame.
    """

    inner_op: Op
    n_keys: int
    universe_groups: tuple[tuple[int, ...], ...]
    capacity: int = 4096
    output_kind: str = "vector"
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        dtype = jnp.asarray(sample).dtype
        keys = []
        occupied = []
        inner_states = []

        for group in self.universe_groups:
            group_width = len(group)
            group_sample = jnp.zeros((group_width,), dtype=dtype)
            inner_state = self.inner_op.init_state(group_sample) if self.inner_op.is_stateful else None

            keys.append(jnp.full((self.capacity, self.n_keys), jnp.nan, dtype=jnp.float64))
            occupied.append(jnp.zeros((self.capacity,), dtype=bool))
            inner_states.append(_tree_broadcast_slots(inner_state, self.capacity) if inner_state is not None else None)

        return GroupByState(
            keys=tuple(keys),
            occupied=tuple(occupied),
            inner_state=tuple(inner_states),
        )

    def _key_matrix_for_group(self, idx: jax.Array, key_cols: tuple[jax.Array, ...]) -> jax.Array:
        if self.n_keys == 0:
            return jnp.zeros((idx.shape[0], 0), dtype=jnp.float64)

        return jnp.stack(
            tuple(jnp.take(jnp.asarray(key_col, dtype=jnp.float64), idx, axis=0) for key_col in key_cols),
            axis=1,
        )

    def _group_template(self, group_state: Any, idx: jax.Array, args: tuple[jax.Array, ...]):
        sample_state = _tree_take_slot(group_state, jnp.asarray(0, dtype=jnp.int32)) if group_state is not None else None
        sample_args = tuple(_slice_group_arg(arg, idx) for arg in args)
        _, template = self.inner_op.tick(sample_state, *sample_args)
        return template

    def _tick_group(
        self,
        keys: jax.Array,
        occupied: jax.Array,
        inner_state: Any,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        group_width = idx.shape[0]
        template = self._group_template(inner_state, idx, args)
        group_out0 = _group_output_like(template, group_width)

        def body(member_pos, carry):
            keys_c, occupied_c, inner_state_c, group_out_c = carry
            key = key_matrix[member_pos]

            already_processed = jnp.asarray(False)
            for prev_pos in range(group_width):
                already_processed = jnp.where(
                    prev_pos < member_pos,
                    already_processed | _same_key_vector(key_matrix[prev_pos], key),
                    already_processed,
                )

            def process(process_carry):
                keys_p, occupied_p, inner_state_p, group_out_p = process_carry

                matches = _key_matches(keys_p, occupied_p, key)
                slot = _first_available_slot(matches, occupied_p)

                mask = jax.vmap(lambda row_key: _same_key_vector(row_key, key))(key_matrix)
                group_args = tuple(_mask_group_arg(arg, idx, mask) for arg in args)

                selected_state = _tree_take_slot(inner_state_p, slot) if inner_state_p is not None else None
                updated_state, local_out = self.inner_op.tick(selected_state, *group_args)

                inner_state_next = (
                    _tree_set_slot(inner_state_p, slot, updated_state)
                    if inner_state_p is not None and updated_state is not None
                    else inner_state_p
                )

                aligned_out = _align_group_output(local_out, group_width)
                group_out_next = _mask_group_output(group_out_p, aligned_out, mask)

                keys_next = keys_p.at[slot].set(key)
                occupied_next = occupied_p.at[slot].set(True)

                return keys_next, occupied_next, inner_state_next, group_out_next

            return jax.lax.cond(
                already_processed,
                lambda x: x,
                process,
                (keys_c, occupied_c, inner_state_c, group_out_c),
            )

        return jax.lax.fori_loop(
            0,
            group_width,
            body,
            (keys, occupied, inner_state, group_out0),
        )

    def tick(self, state: GroupByState, *child_values: jax.Array):
        key_cols = tuple(jnp.asarray(child_values[i]) for i in range(self.n_keys))
        args = tuple(jnp.asarray(v) for v in child_values[self.n_keys:])

        width = args[0].shape[0] if args and args[0].ndim > 0 else key_cols[0].shape[0]

        first_idx = jnp.asarray(self.universe_groups[0], dtype=jnp.int64)
        first_template = self._group_template(state.inner_state[0], first_idx, args)
        out = _empty_output_like(first_template, width)

        new_keys = []
        new_occupied = []
        new_inner_states = []

        for group_i, group in enumerate(self.universe_groups):
            idx = jnp.asarray(group, dtype=jnp.int64)
            key_matrix = self._key_matrix_for_group(idx, key_cols)

            keys_i, occupied_i, inner_state_i, group_out = self._tick_group(
                state.keys[group_i],
                state.occupied[group_i],
                state.inner_state[group_i],
                idx,
                key_matrix,
                args,
            )

            out = _scatter_group_output(out, idx, group_out)

            new_keys.append(keys_i)
            new_occupied.append(occupied_i)
            new_inner_states.append(inner_state_i)

        return (
            GroupByState(
                keys=tuple(new_keys),
                occupied=tuple(new_occupied),
                inner_state=tuple(new_inner_states),
            ),
            out,
        )

