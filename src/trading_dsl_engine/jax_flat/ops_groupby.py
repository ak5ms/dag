from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat.ops import Op

jax.config.update("jax_enable_x64", True)


_AUTO_REGISTERED_DATACLASS_TYPES: set[type] = set()


def _is_dataclass_instance(value: Any) -> bool:
    return is_dataclass(value) and not isinstance(value, type)


def _ensure_dataclass_pytree(value: Any) -> Any:
    """Register plain dataclass instances as JAX pytrees on first use."""
    if _is_dataclass_instance(value):
        cls = type(value)
        for field in fields(value):
            _ensure_dataclass_pytree(getattr(value, field.name))

        if cls not in _AUTO_REGISTERED_DATACLASS_TYPES:
            leaves, _ = jax.tree_util.tree_flatten(value)
            if len(leaves) == 1 and leaves[0] is value:
                try:
                    jax.tree_util.register_dataclass(cls)
                except ValueError:
                    pass
            _AUTO_REGISTERED_DATACLASS_TYPES.add(cls)
        return value

    if isinstance(value, tuple):
        for item in value:
            _ensure_dataclass_pytree(item)
    elif isinstance(value, list):
        for item in value:
            _ensure_dataclass_pytree(item)
    elif isinstance(value, dict):
        for item in value.values():
            _ensure_dataclass_pytree(item)
    return value


def _tree_has_group_axis(tree: Any, group_width: int) -> bool:
    tree = _ensure_dataclass_pytree(tree)
    leaves = jax.tree_util.tree_leaves(tree)
    return any(jnp.asarray(leaf).ndim > 0 and jnp.asarray(leaf).shape[0] == group_width for leaf in leaves)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GroupByState:
    keys: tuple[jax.Array, ...]
    occupied: tuple[jax.Array, ...]
    table_slots: tuple[jax.Array, ...]
    counts: tuple[jax.Array, ...]
    cached_keys: tuple[jax.Array, ...]
    cached_slots: tuple[jax.Array, ...]
    cache_valid: tuple[jax.Array, ...]
    cached_inner_state: tuple[Any, ...]
    inner_state: tuple[Any, ...]


def _tree_broadcast_slots(tree: Any, capacity: int):
    tree = _ensure_dataclass_pytree(tree)
    return jax.tree_util.tree_map(
        lambda leaf: jnp.broadcast_to(jnp.asarray(leaf), (capacity,) + jnp.asarray(leaf).shape),
        tree,
    )


def _tree_take_slot(tree: Any, slot: jax.Array):
    tree = _ensure_dataclass_pytree(tree)
    return jax.tree_util.tree_map(lambda leaf: jnp.take(leaf, slot, axis=0), tree)


def _tree_set_slot(tree: Any, slot: jax.Array, value: Any):
    tree = _ensure_dataclass_pytree(tree)
    value = _ensure_dataclass_pytree(value)
    return jax.tree_util.tree_map(lambda dst, src: dst.at[slot].set(src), tree, value)


def _hash_mix(value: jax.Array) -> jax.Array:
    value = value ^ (value >> jnp.uint64(33))
    value = value * jnp.uint64(0xff51afd7ed558ccd)
    value = value ^ (value >> jnp.uint64(33))
    value = value * jnp.uint64(0xc4ceb9fe1a85ec53)
    return value ^ (value >> jnp.uint64(33))


def _hash_key(key: jax.Array) -> jax.Array:
    key = jnp.asarray(key, dtype=jnp.float64)
    bits = jax.lax.bitcast_convert_type(key, jnp.uint64)
    nan_bits = jnp.full(bits.shape, jnp.uint64(0x7ff8000000000000), dtype=jnp.uint64)
    bits = jnp.where(jnp.isnan(key), nan_bits, bits)
    positions = jnp.arange(bits.shape[0], dtype=jnp.uint64)
    mixed = _hash_mix(bits + positions * jnp.uint64(0x9e3779b97f4a7c15))
    return jnp.bitwise_xor.reduce(mixed, initial=jnp.uint64(0x9e3779b97f4a7c15))


def _lookup_or_insert_slot(
    keys: jax.Array,
    occupied: jax.Array,
    table_slots: jax.Array,
    count: jax.Array,
    key: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    capacity = keys.shape[0]
    table_capacity = table_slots.shape[0]
    start = jnp.asarray(_hash_key(key) % jnp.uint64(table_capacity), dtype=jnp.int32)

    def cond(carry):
        probe, done, _, _ = carry
        return (probe < table_capacity) & ~done

    def body(carry):
        probe, _, slot, insert_bucket = carry
        bucket = (start + probe) % table_capacity
        candidate = table_slots[bucket]
        empty = candidate < 0
        safe_candidate = jnp.maximum(candidate, jnp.asarray(0, dtype=jnp.int32))
        matched = (~empty) & _same_key_vector(keys[safe_candidate], key)
        done = empty | matched
        next_slot = jnp.where(matched, candidate, slot)
        next_insert_bucket = jnp.where(empty, bucket, insert_bucket)
        return probe + 1, done, next_slot, next_insert_bucket

    _, _, found_slot, insert_bucket = jax.lax.while_loop(
        cond,
        body,
        (
            jnp.asarray(0, dtype=jnp.int32),
            jnp.asarray(False),
            jnp.asarray(-1, dtype=jnp.int32),
            start,
        ),
    )

    has_match = found_slot >= 0
    next_free = jnp.minimum(count, jnp.asarray(capacity - 1, dtype=jnp.int32))
    slot = jnp.where(has_match, found_slot, next_free)
    keys_next = jax.lax.cond(has_match, lambda x: x, lambda x: x.at[slot].set(key), keys)
    occupied_next = jax.lax.cond(has_match, lambda x: x, lambda x: x.at[slot].set(True), occupied)
    table_next = jax.lax.cond(has_match, lambda x: x, lambda x: x.at[insert_bucket].set(slot), table_slots)
    count_next = jnp.where(has_match, count, jnp.minimum(count + 1, jnp.asarray(capacity, dtype=jnp.int32)))
    return slot, keys_next, occupied_next, table_next, count_next


def _empty_output_like(template: Any, width: int):
    template = _ensure_dataclass_pytree(template)

    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        suffix = leaf.shape[1:] if leaf.ndim > 0 else ()
        return jnp.full((width,) + suffix, jnp.nan, dtype=leaf.dtype)

    return jax.tree_util.tree_map(alloc, template)


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


def _all_same_group_key(key_matrix: jax.Array) -> jax.Array:
    first = key_matrix[0]
    return jnp.all(jax.vmap(lambda row_key: _same_key_vector(row_key, first))(key_matrix))


def _flush_cached_inner_state(
    inner_state: Any,
    cached_inner_state: Any,
    cached_slot: jax.Array,
    cache_valid: jax.Array,
):
    if inner_state is None or cached_inner_state is None:
        return inner_state
    return jax.lax.cond(
        cache_valid,
        lambda state: _tree_set_slot(state, cached_slot, cached_inner_state),
        lambda state: state,
        inner_state,
    )


def _group_output_like(template: Any, group_width: int):
    template = _ensure_dataclass_pytree(template)

    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        suffix = leaf.shape[1:] if leaf.ndim > 0 else ()
        return jnp.full((group_width,) + suffix, jnp.nan, dtype=leaf.dtype)

    return jax.tree_util.tree_map(alloc, template)


def _align_group_output(value: Any, group_width: int):
    value = _ensure_dataclass_pytree(value)

    def align_leaf(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.ndim == 0:
            return jnp.broadcast_to(leaf, (group_width,))
        if leaf.shape[0] == 1:
            return jnp.broadcast_to(leaf[0], (group_width,) + leaf.shape[1:])
        return leaf

    return jax.tree_util.tree_map(align_leaf, value)


def _mask_group_output(old: Any, new: Any, mask: jax.Array):
    old = _ensure_dataclass_pytree(old)
    new = _ensure_dataclass_pytree(new)

    def mask_leaf(dst, src):
        mask_shape = mask.shape + (1,) * (jnp.asarray(src).ndim - 1)
        return jnp.where(jnp.reshape(mask, mask_shape), src, dst)

    return jax.tree_util.tree_map(mask_leaf, old, new)


def _scatter_group_output(out: Any, idx: jax.Array, group_value: Any):
    out = _ensure_dataclass_pytree(out)
    group_value = _ensure_dataclass_pytree(group_value)
    return jax.tree_util.tree_map(lambda dst, src: dst.at[idx].set(src), out, group_value)


def _empty_batch_output_like(template: Any, n_steps: int, width: int):
    template = _ensure_dataclass_pytree(template)

    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        suffix = leaf.shape[2:] if leaf.ndim > 1 else ()
        return jnp.full((n_steps, width) + suffix, jnp.nan, dtype=leaf.dtype)

    return jax.tree_util.tree_map(alloc, template)


def _empty_batch_group_output_like(template: Any, n_steps: int, group_width: int):
    template = _ensure_dataclass_pytree(template)

    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        suffix = leaf.shape[1:] if leaf.ndim > 0 else ()
        return jnp.full((n_steps, group_width) + suffix, jnp.nan, dtype=leaf.dtype)

    return jax.tree_util.tree_map(alloc, template)


def _set_time_output(out: Any, t: jax.Array, value: Any):
    out = _ensure_dataclass_pytree(out)
    value = _ensure_dataclass_pytree(value)
    return jax.tree_util.tree_map(lambda dst, src: dst.at[t].set(src), out, value)


def _scatter_batch_group_output(out: Any, idx: jax.Array, group_values: Any):
    out = _ensure_dataclass_pytree(out)
    group_values = _ensure_dataclass_pytree(group_values)
    return jax.tree_util.tree_map(lambda dst, src: dst.at[:, idx].set(src), out, group_values)


def _slice_member_arg(value: jax.Array, idx: jax.Array, member_pos: jax.Array) -> jax.Array:
    group_value = _slice_group_arg(value, idx)
    if jnp.asarray(group_value).ndim == 0:
        return group_value
    return jnp.take(group_value, member_pos, axis=0)


def _set_member_output(out: Any, member_pos: jax.Array, value: Any):
    out = _ensure_dataclass_pytree(out)
    value = _ensure_dataclass_pytree(value)

    def set_leaf(dst, src):
        return dst.at[member_pos].set(src)

    return jax.tree_util.tree_map(set_leaf, out, value)


@dataclass(frozen=True)
class GroupByOp(Op):
    inner_op: Op
    n_keys: int
    universe_groups: tuple[tuple[int, ...], ...] | None = None
    capacity: int = 4096
    hash_capacity: int = 8192
    output_kind: str = "vector"
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        dtype = jnp.asarray(sample).dtype
        keys = []
        occupied = []
        table_slots = []
        counts = []
        cached_keys = []
        cached_slots = []
        cache_valid = []
        cached_inner_states = []
        inner_states = []
        hash_capacity = max(self.hash_capacity, self.capacity)

        groups = self.universe_groups
        if groups is None:
            groups = (tuple(range(int(jnp.asarray(sample).shape[0]))),)

        for group in groups:
            group_width = len(group)
            group_sample = jnp.zeros((group_width,), dtype=dtype)
            inner_state = self.inner_op.init_state(group_sample) if self.inner_op.is_stateful else None

            keys.append(jnp.full((self.capacity, self.n_keys), jnp.nan, dtype=jnp.float64))
            occupied.append(jnp.zeros((self.capacity,), dtype=bool))
            table_slots.append(jnp.full((hash_capacity,), -1, dtype=jnp.int32))
            counts.append(jnp.asarray(0, dtype=jnp.int32))
            slot_state = _tree_broadcast_slots(inner_state, self.capacity) if inner_state is not None else None
            cached_keys.append(jnp.full((self.n_keys,), jnp.nan, dtype=jnp.float64))
            cached_slots.append(jnp.asarray(-1, dtype=jnp.int32))
            cache_valid.append(jnp.asarray(False))
            cached_inner_states.append(_tree_take_slot(slot_state, jnp.asarray(0, dtype=jnp.int32)) if slot_state is not None else None)
            inner_states.append(slot_state)

        return GroupByState(
            keys=tuple(keys),
            occupied=tuple(occupied),
            table_slots=tuple(table_slots),
            counts=tuple(counts),
            cached_keys=tuple(cached_keys),
            cached_slots=tuple(cached_slots),
            cache_valid=tuple(cache_valid),
            cached_inner_state=tuple(cached_inner_states),
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
        return _ensure_dataclass_pytree(template)

    def _tick_group_groupwise(
        self,
        keys: jax.Array,
        occupied: jax.Array,
        table_slots: jax.Array,
        count: jax.Array,
        inner_state: Any,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        group_width = idx.shape[0]
        template = self._group_template(inner_state, idx, args)
        group_out0 = _group_output_like(template, group_width)

        def body(member_pos, carry):
            keys_c, occupied_c, table_slots_c, count_c, inner_state_c, group_out_c = carry
            key = key_matrix[member_pos]

            already_processed = jnp.asarray(False)
            for prev_pos in range(group_width):
                already_processed = jnp.where(
                    prev_pos < member_pos,
                    already_processed | _same_key_vector(key_matrix[prev_pos], key),
                    already_processed,
                )

            def process(process_carry):
                keys_p, occupied_p, table_slots_p, count_p, inner_state_p, group_out_p = process_carry

                slot, keys_found, occupied_found, table_slots_found, count_found = _lookup_or_insert_slot(
                    keys_p,
                    occupied_p,
                    table_slots_p,
                    count_p,
                    key,
                )

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

                return keys_found, occupied_found, table_slots_found, count_found, inner_state_next, group_out_next

            return jax.lax.cond(
                already_processed,
                lambda x: x,
                process,
                (keys_c, occupied_c, table_slots_c, count_c, inner_state_c, group_out_c),
            )

        return jax.lax.fori_loop(
            0,
            group_width,
            body,
            (keys, occupied, table_slots, count, inner_state, group_out0),
        )

    def _tick_group_elementwise(
        self,
        keys: jax.Array,
        occupied: jax.Array,
        table_slots: jax.Array,
        count: jax.Array,
        inner_state: Any,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        group_width = idx.shape[0]
        template = self._group_template(inner_state, idx, args)
        group_out0 = _group_output_like(template, group_width)

        def body(member_pos, carry):
            keys_c, occupied_c, table_slots_c, count_c, inner_state_c, group_out_c = carry
            key = key_matrix[member_pos]
            slot, keys_found, occupied_found, table_slots_found, count_found = _lookup_or_insert_slot(
                keys_c,
                occupied_c,
                table_slots_c,
                count_c,
                key,
            )

            selected_state = _tree_take_slot(inner_state_c, slot)
            member_args = tuple(_slice_member_arg(arg, idx, member_pos) for arg in args)
            updated_state, local_out = self.inner_op.tick(selected_state, *member_args)

            inner_state_next = _tree_set_slot(inner_state_c, slot, updated_state) if updated_state is not None else inner_state_c
            group_out_next = _set_member_output(group_out_c, member_pos, local_out)
            return keys_found, occupied_found, table_slots_found, count_found, inner_state_next, group_out_next

        return jax.lax.fori_loop(
            0,
            group_width,
            body,
            (keys, occupied, table_slots, count, inner_state, group_out0),
        )

    def _tick_group_cached_single_key(
        self,
        keys: jax.Array,
        occupied: jax.Array,
        table_slots: jax.Array,
        count: jax.Array,
        inner_state: Any,
        cached_key: jax.Array,
        cached_slot: jax.Array,
        cache_valid: jax.Array,
        cached_inner_state: Any,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        key = key_matrix[0]
        group_width = idx.shape[0]
        group_args = tuple(_slice_group_arg(arg, idx) for arg in args)
        cache_hit = cache_valid & _same_key_vector(cached_key, key)

        def hit(_):
            return keys, occupied, table_slots, count, inner_state, cached_slot, cached_inner_state

        def miss(_):
            flushed_state = _flush_cached_inner_state(inner_state, cached_inner_state, cached_slot, cache_valid)
            slot, keys_next, occupied_next, table_slots_next, count_next = _lookup_or_insert_slot(
                keys,
                occupied,
                table_slots,
                count,
                key,
            )
            selected_state = _tree_take_slot(flushed_state, slot) if flushed_state is not None else None
            return keys_next, occupied_next, table_slots_next, count_next, flushed_state, slot, selected_state

        keys_c, occupied_c, table_slots_c, count_c, inner_state_c, slot_c, selected_state = jax.lax.cond(
            cache_hit,
            hit,
            miss,
            operand=None,
        )

        updated_state, local_out = self.inner_op.tick(selected_state, *group_args)
        cached_state_next = updated_state if updated_state is not None else selected_state
        return (
            keys_c,
            occupied_c,
            table_slots_c,
            count_c,
            key,
            slot_c,
            jnp.asarray(True),
            cached_state_next,
            inner_state_c,
            _align_group_output(local_out, group_width),
        )

    def _tick_group_groupwise_uncached(
        self,
        keys: jax.Array,
        occupied: jax.Array,
        table_slots: jax.Array,
        count: jax.Array,
        inner_state: Any,
        cached_key: jax.Array,
        cached_slot: jax.Array,
        cache_valid: jax.Array,
        cached_inner_state: Any,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        flushed_state = _flush_cached_inner_state(inner_state, cached_inner_state, cached_slot, cache_valid)
        keys_next, occupied_next, table_slots_next, count_next, inner_state_next, group_out = self._tick_group_groupwise(
            keys,
            occupied,
            table_slots,
            count,
            flushed_state,
            idx,
            key_matrix,
            args,
        )
        return (
            keys_next,
            occupied_next,
            table_slots_next,
            count_next,
            cached_key,
            cached_slot,
            jnp.asarray(False),
            cached_inner_state,
            inner_state_next,
            group_out,
        )

    def _tick_group(
        self,
        keys: jax.Array,
        occupied: jax.Array,
        table_slots: jax.Array,
        count: jax.Array,
        cached_key: jax.Array,
        cached_slot: jax.Array,
        cache_valid: jax.Array,
        cached_inner_state: Any,
        inner_state: Any,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        # Formula-compiled groupby RHS graphs initialize state with a leading
        # group axis, so cross-sectional ops must see every member in the key.
        # Direct grouped-apply ops with scalar keyed state advance per member;
        # repeated keys in one cross-section should therefore tick sequentially.
        if inner_state is not None and not _tree_has_group_axis(_tree_take_slot(inner_state, jnp.asarray(0, dtype=jnp.int32)), idx.shape[0]):
            keys_next, occupied_next, table_slots_next, count_next, inner_state_next, group_out = self._tick_group_elementwise(
                keys,
                occupied,
                table_slots,
                count,
                inner_state,
                idx,
                key_matrix,
                args,
            )
            return (
                keys_next,
                occupied_next,
                table_slots_next,
                count_next,
                cached_key,
                cached_slot,
                cache_valid,
                cached_inner_state,
                inner_state_next,
                group_out,
            )

        if inner_state is None:
            keys_next, occupied_next, table_slots_next, count_next, inner_state_next, group_out = self._tick_group_groupwise(
                keys,
                occupied,
                table_slots,
                count,
                inner_state,
                idx,
                key_matrix,
                args,
            )
            return (
                keys_next,
                occupied_next,
                table_slots_next,
                count_next,
                cached_key,
                cached_slot,
                cache_valid,
                cached_inner_state,
                inner_state_next,
                group_out,
            )

        return jax.lax.cond(
            _all_same_group_key(key_matrix),
            lambda _: self._tick_group_cached_single_key(
                keys,
                occupied,
                table_slots,
                count,
                inner_state,
                cached_key,
                cached_slot,
                cache_valid,
                cached_inner_state,
                idx,
                key_matrix,
                args,
            ),
            lambda _: self._tick_group_groupwise_uncached(
                keys,
                occupied,
                table_slots,
                count,
                inner_state,
                cached_key,
                cached_slot,
                cache_valid,
                cached_inner_state,
                idx,
                key_matrix,
                args,
            ),
            operand=None,
        )

    def _batch_key_matrix_for_group(self, idx: jax.Array, key_cols: tuple[jax.Array, ...]) -> jax.Array:
        if self.n_keys == 0:
            n_steps = key_cols[0].shape[0]
            return jnp.zeros((n_steps, idx.shape[0], 0), dtype=jnp.float64)

        return jnp.stack(
            tuple(jnp.take(jnp.asarray(key_col, dtype=jnp.float64), idx, axis=1) for key_col in key_cols),
            axis=2,
        )

    def _scan_group_rows(
        self,
        keys: jax.Array,
        occupied: jax.Array,
        table_slots: jax.Array,
        count: jax.Array,
        cached_key: jax.Array,
        cached_slot: jax.Array,
        cache_valid: jax.Array,
        cached_inner_state: Any,
        inner_state: Any,
        idx: jax.Array,
        key_seq: jax.Array,
        args_seq: tuple[jax.Array, ...],
    ):
        n_steps = key_seq.shape[0]
        group_width = idx.shape[0]
        sample_args = tuple(_slice_group_arg(arg[0], idx) for arg in args_seq)
        sample_state = _tree_take_slot(inner_state, jnp.asarray(0, dtype=jnp.int32)) if inner_state is not None else None
        _, template = self.inner_op.tick(sample_state, *sample_args)
        out0 = _empty_batch_group_output_like(template, n_steps, group_width)

        def step(carry, row_values):
            (
                keys_c,
                occupied_c,
                table_slots_c,
                count_c,
                cached_key_c,
                cached_slot_c,
                cache_valid_c,
                cached_inner_state_c,
                inner_state_c,
            ) = carry
            key_matrix, row_args = row_values
            (
                keys_n,
                occupied_n,
                table_slots_n,
                count_n,
                cached_key_n,
                cached_slot_n,
                cache_valid_n,
                cached_inner_state_n,
                inner_state_n,
                group_out,
            ) = self._tick_group(
                keys_c,
                occupied_c,
                table_slots_c,
                count_c,
                cached_key_c,
                cached_slot_c,
                cache_valid_c,
                cached_inner_state_c,
                inner_state_c,
                idx,
                key_matrix,
                row_args,
            )
            return (
                keys_n,
                occupied_n,
                table_slots_n,
                count_n,
                cached_key_n,
                cached_slot_n,
                cache_valid_n,
                cached_inner_state_n,
                inner_state_n,
            ), group_out

        return jax.lax.scan(
            step,
            (
                keys,
                occupied,
                table_slots,
                count,
                cached_key,
                cached_slot,
                cache_valid,
                cached_inner_state,
                inner_state,
            ),
            (key_seq, tuple(args_seq)),
            unroll=32,
        )

    def _scan_group_runs(
        self,
        keys: jax.Array,
        occupied: jax.Array,
        table_slots: jax.Array,
        count: jax.Array,
        cached_key: jax.Array,
        cached_slot: jax.Array,
        cache_valid: jax.Array,
        cached_inner_state: Any,
        inner_state: Any,
        idx: jax.Array,
        key_seq: jax.Array,
        args_seq: tuple[jax.Array, ...],
        require_uniform: bool = False,
    ):
        n_steps = key_seq.shape[0]
        group_width = idx.shape[0]
        sample_args = tuple(_slice_group_arg(arg[0], idx) for arg in args_seq)
        sample_state = _tree_take_slot(inner_state, jnp.asarray(0, dtype=jnp.int32))
        _, template = self.inner_op.tick(sample_state, *sample_args)
        out0 = _empty_batch_group_output_like(template, n_steps, group_width)

        def find_run_end(start, key):
            def cond(i):
                return (i < n_steps) & _all_same_group_key(key_seq[i]) & _same_key_vector(key_seq[i, 0], key)

            def body(i):
                return i + 1

            return jax.lax.while_loop(cond, body, start)

        def process_uniform_run(carry):
            (
                pos_c,
                keys_c,
                occupied_c,
                table_slots_c,
                count_c,
                cached_key_c,
                cached_slot_c,
                cache_valid_c,
                cached_inner_state_c,
                inner_state_c,
                out_c,
            ) = carry
            key = key_seq[pos_c, 0]
            run_end = find_run_end(pos_c, key)
            cache_hit = cache_valid_c & _same_key_vector(cached_key_c, key)

            def hit(_):
                return keys_c, occupied_c, table_slots_c, count_c, inner_state_c, cached_slot_c, cached_inner_state_c

            def miss(_):
                flushed_state = _flush_cached_inner_state(inner_state_c, cached_inner_state_c, cached_slot_c, cache_valid_c)
                slot, keys_next, occupied_next, table_slots_next, count_next = _lookup_or_insert_slot(
                    keys_c,
                    occupied_c,
                    table_slots_c,
                    count_c,
                    key,
                )
                selected_state = _tree_take_slot(flushed_state, slot)
                return keys_next, occupied_next, table_slots_next, count_next, flushed_state, slot, selected_state

            keys_s, occupied_s, table_slots_s, count_s, inner_state_s, slot_s, selected_state = jax.lax.cond(
                cache_hit,
                hit,
                miss,
                operand=None,
            )

            def row_body(t, row_carry):
                state_c, out_rows_c = row_carry
                group_args = tuple(_slice_group_arg(arg[t], idx) for arg in args_seq)
                next_state, local_out = self.inner_op.tick(state_c, *group_args)
                group_out = _align_group_output(local_out, group_width)
                return next_state, _set_time_output(out_rows_c, t, group_out)

            cached_state_next, out_next = jax.lax.fori_loop(
                pos_c,
                run_end,
                row_body,
                (selected_state, out_c),
            )
            return (
                run_end,
                keys_s,
                occupied_s,
                table_slots_s,
                count_s,
                key,
                slot_s,
                jnp.asarray(True),
                cached_state_next,
                inner_state_s,
                out_next,
            )

        def process_mixed_row(carry):
            (
                pos_c,
                keys_c,
                occupied_c,
                table_slots_c,
                count_c,
                cached_key_c,
                cached_slot_c,
                cache_valid_c,
                cached_inner_state_c,
                inner_state_c,
                out_c,
            ) = carry
            row_args = tuple(arg[pos_c] for arg in args_seq)
            (
                keys_n,
                occupied_n,
                table_slots_n,
                count_n,
                cached_key_n,
                cached_slot_n,
                cache_valid_n,
                cached_inner_state_n,
                inner_state_n,
                group_out,
            ) = self._tick_group(
                keys_c,
                occupied_c,
                table_slots_c,
                count_c,
                cached_key_c,
                cached_slot_c,
                cache_valid_c,
                cached_inner_state_c,
                inner_state_c,
                idx,
                key_seq[pos_c],
                row_args,
            )
            return (
                pos_c + 1,
                keys_n,
                occupied_n,
                table_slots_n,
                count_n,
                cached_key_n,
                cached_slot_n,
                cache_valid_n,
                cached_inner_state_n,
                inner_state_n,
                _set_time_output(out_c, pos_c, group_out),
            )

        def cond(carry):
            return carry[0] < n_steps

        def body(carry):
            if require_uniform:
                return process_uniform_run(carry)
            return jax.lax.cond(
                _all_same_group_key(key_seq[carry[0]]),
                process_uniform_run,
                process_mixed_row,
                carry,
            )

        return jax.lax.while_loop(
            cond,
            body,
            (
                jnp.asarray(0, dtype=jnp.int32),
                keys,
                occupied,
                table_slots,
                count,
                cached_key,
                cached_slot,
                cache_valid,
                cached_inner_state,
                inner_state,
                out0,
            ),
        )

    def scan_batch(self, state: GroupByState, *child_sequences: jax.Array):
        key_cols = tuple(jnp.asarray(child_sequences[i]) for i in range(self.n_keys))
        args_seq = tuple(jnp.asarray(v) for v in child_sequences[self.n_keys:])
        n_steps = args_seq[0].shape[0] if args_seq else key_cols[0].shape[0]
        width = args_seq[0].shape[1] if args_seq and args_seq[0].ndim > 1 else key_cols[0].shape[1]

        groups = self.universe_groups
        if groups is None:
            group_indices = (jnp.arange(width, dtype=jnp.int64),)
        else:
            group_indices = tuple(jnp.asarray(group, dtype=jnp.int64) for group in groups)

        out = None
        new_keys = []
        new_occupied = []
        new_table_slots = []
        new_counts = []
        new_cached_keys = []
        new_cached_slots = []
        new_cache_valid = []
        new_cached_inner_states = []
        new_inner_states = []

        for group_i, idx in enumerate(group_indices):
            key_seq = self._batch_key_matrix_for_group(idx, key_cols)
            group_args_seq = args_seq
            group_state = state.inner_state[group_i]
            use_run_scan = group_state is not None and _tree_has_group_axis(
                _tree_take_slot(group_state, jnp.asarray(0, dtype=jnp.int32)),
                idx.shape[0],
            )

            if use_run_scan:
                all_uniform = jnp.all(jax.vmap(_all_same_group_key)(key_seq))

                def run_scan(require_uniform):
                    (
                        _,
                        keys_r,
                        occupied_r,
                        table_slots_r,
                        count_r,
                        cached_key_r,
                        cached_slot_r,
                        cache_valid_r,
                        cached_inner_state_r,
                        inner_state_r,
                        group_out_r,
                    ) = self._scan_group_runs(
                        state.keys[group_i],
                        state.occupied[group_i],
                        state.table_slots[group_i],
                        state.counts[group_i],
                        state.cached_keys[group_i],
                        state.cached_slots[group_i],
                        state.cache_valid[group_i],
                        state.cached_inner_state[group_i],
                        group_state,
                        idx,
                        key_seq,
                        group_args_seq,
                        require_uniform=require_uniform,
                    )
                    return (
                        keys_r,
                        occupied_r,
                        table_slots_r,
                        count_r,
                        cached_key_r,
                        cached_slot_r,
                        cache_valid_r,
                        cached_inner_state_r,
                        inner_state_r,
                    ), group_out_r

                (
                    keys_i,
                    occupied_i,
                    table_slots_i,
                    count_i,
                    cached_key_i,
                    cached_slot_i,
                    cache_valid_i,
                    cached_inner_state_i,
                    inner_state_i,
                ), group_out = jax.lax.cond(
                    all_uniform,
                    lambda _: run_scan(True),
                    lambda _: run_scan(False),
                    operand=None,
                )
            else:
                (
                    keys_i,
                    occupied_i,
                    table_slots_i,
                    count_i,
                    cached_key_i,
                    cached_slot_i,
                    cache_valid_i,
                    cached_inner_state_i,
                    inner_state_i,
                ), group_out = self._scan_group_rows(
                    state.keys[group_i],
                    state.occupied[group_i],
                    state.table_slots[group_i],
                    state.counts[group_i],
                    state.cached_keys[group_i],
                    state.cached_slots[group_i],
                    state.cache_valid[group_i],
                    state.cached_inner_state[group_i],
                    group_state,
                    idx,
                    key_seq,
                    group_args_seq,
                )

            if out is None:
                out = _empty_batch_output_like(group_out, n_steps, width)
            out = _scatter_batch_group_output(out, idx, group_out)

            new_keys.append(keys_i)
            new_occupied.append(occupied_i)
            new_table_slots.append(table_slots_i)
            new_counts.append(count_i)
            new_cached_keys.append(cached_key_i)
            new_cached_slots.append(cached_slot_i)
            new_cache_valid.append(cache_valid_i)
            new_cached_inner_states.append(cached_inner_state_i)
            new_inner_states.append(inner_state_i)

        return (
            GroupByState(
                keys=tuple(new_keys),
                occupied=tuple(new_occupied),
                table_slots=tuple(new_table_slots),
                counts=tuple(new_counts),
                cached_keys=tuple(new_cached_keys),
                cached_slots=tuple(new_cached_slots),
                cache_valid=tuple(new_cache_valid),
                cached_inner_state=tuple(new_cached_inner_states),
                inner_state=tuple(new_inner_states),
            ),
            out,
        )


    def tick(self, state: GroupByState, *child_values: jax.Array):
        key_cols = tuple(jnp.asarray(child_values[i]) for i in range(self.n_keys))
        args = tuple(jnp.asarray(v) for v in child_values[self.n_keys:])

        width = args[0].shape[0] if args and args[0].ndim > 0 else key_cols[0].shape[0]

        groups = self.universe_groups
        if groups is None:
            group_indices = (jnp.arange(width, dtype=jnp.int64),)
        else:
            group_indices = tuple(jnp.asarray(group, dtype=jnp.int64) for group in groups)

        out = None

        new_keys = []
        new_occupied = []
        new_table_slots = []
        new_counts = []
        new_cached_keys = []
        new_cached_slots = []
        new_cache_valid = []
        new_cached_inner_states = []
        new_inner_states = []

        for group_i, idx in enumerate(group_indices):
            key_matrix = self._key_matrix_for_group(idx, key_cols)

            (
                keys_i,
                occupied_i,
                table_slots_i,
                count_i,
                cached_key_i,
                cached_slot_i,
                cache_valid_i,
                cached_inner_state_i,
                inner_state_i,
                group_out,
            ) = self._tick_group(
                state.keys[group_i],
                state.occupied[group_i],
                state.table_slots[group_i],
                state.counts[group_i],
                state.cached_keys[group_i],
                state.cached_slots[group_i],
                state.cache_valid[group_i],
                state.cached_inner_state[group_i],
                state.inner_state[group_i],
                idx,
                key_matrix,
                args,
            )

            if out is None:
                out = _empty_output_like(group_out, width)
            out = _scatter_group_output(out, idx, group_out)

            new_keys.append(keys_i)
            new_occupied.append(occupied_i)
            new_table_slots.append(table_slots_i)
            new_counts.append(count_i)
            new_cached_keys.append(cached_key_i)
            new_cached_slots.append(cached_slot_i)
            new_cache_valid.append(cache_valid_i)
            new_cached_inner_states.append(cached_inner_state_i)
            new_inner_states.append(inner_state_i)

        return (
            GroupByState(
                keys=tuple(new_keys),
                occupied=tuple(new_occupied),
                table_slots=tuple(new_table_slots),
                counts=tuple(new_counts),
                cached_keys=tuple(new_cached_keys),
                cached_slots=tuple(new_cached_slots),
                cache_valid=tuple(new_cache_valid),
                cached_inner_state=tuple(new_cached_inner_states),
                inner_state=tuple(new_inner_states),
            ),
            out,
        )

