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
class GroupTableState:
    keys: jax.Array
    occupied: jax.Array
    table_slots: jax.Array
    count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GroupCacheState:
    key: jax.Array
    slot: jax.Array
    valid: jax.Array
    inner_state: Any


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GroupRuntimeState:
    table: GroupTableState
    cache: GroupCacheState
    inner_state: Any


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


def _runtime_from_group_state(state: GroupByState, group_i: int) -> GroupRuntimeState:
    return GroupRuntimeState(
        table=GroupTableState(
            state.keys[group_i],
            state.occupied[group_i],
            state.table_slots[group_i],
            state.counts[group_i],
        ),
        cache=GroupCacheState(
            state.cached_keys[group_i],
            state.cached_slots[group_i],
            state.cache_valid[group_i],
            state.cached_inner_state[group_i],
        ),
        inner_state=state.inner_state[group_i],
    )


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


def _tree_take_slot_member(tree: Any, slot: jax.Array, member_pos: jax.Array):
    tree = _ensure_dataclass_pytree(tree)

    def take_leaf(leaf):
        leaf = jnp.asarray(leaf)
        slot_leaf = jnp.take(leaf, slot, axis=0)
        if slot_leaf.ndim == 0:
            return slot_leaf
        return jnp.take(slot_leaf, member_pos, axis=0)

    return jax.tree_util.tree_map(take_leaf, tree)


def _tree_set_slot_member(tree: Any, slot: jax.Array, member_pos: jax.Array, value: Any):
    tree = _ensure_dataclass_pytree(tree)
    value = _ensure_dataclass_pytree(value)

    def set_leaf(dst, src):
        dst = jnp.asarray(dst)
        if dst.ndim <= 1:
            return dst.at[slot].set(src)
        return dst.at[slot, member_pos].set(src)

    return jax.tree_util.tree_map(set_leaf, tree, value)


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


def _raise_groupby_lookup_error(error_code: jax.Array) -> None:
    messages = {
        1: "jax_flat groupby capacity exceeded; increase GroupByOp.capacity",
        2: (
            "jax_flat groupby hash table exhausted before finding an empty bucket; "
            "increase GroupByOp.hash_capacity"
        ),
    }
    raise RuntimeError(messages[int(error_code)])


def _check_lookup_insertable(
    has_match: jax.Array,
    has_insert_bucket: jax.Array,
    count: jax.Array,
    capacity: int,
) -> None:
    missing_key = ~has_match
    slots_full = missing_key & (count >= jnp.asarray(capacity, dtype=jnp.int32))
    hash_exhausted = missing_key & ~has_insert_bucket
    error_code = jnp.where(slots_full, jnp.asarray(1, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
    error_code = jnp.where(hash_exhausted & ~slots_full, jnp.asarray(2, dtype=jnp.int32), error_code)

    def raise_error(code):
        jax.debug.callback(_raise_groupby_lookup_error, code)
        return code

    jax.lax.cond(error_code != 0, raise_error, lambda code: code, error_code)


def _lookup_or_insert_slot(
    table: GroupTableState,
    key: jax.Array,
) -> tuple[jax.Array, GroupTableState]:
    keys = table.keys
    occupied = table.occupied
    table_slots = table.table_slots
    count = table.count
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
    has_insert_bucket = table_slots[insert_bucket] < 0
    _check_lookup_insertable(has_match, has_insert_bucket, count, capacity)

    next_free = jnp.minimum(count, jnp.asarray(capacity - 1, dtype=jnp.int32))
    slot = jnp.where(has_match, found_slot, next_free)
    keys_next = jax.lax.cond(has_match, lambda x: x, lambda x: x.at[slot].set(key), keys)
    occupied_next = jax.lax.cond(has_match, lambda x: x, lambda x: x.at[slot].set(True), occupied)
    table_next = jax.lax.cond(has_match, lambda x: x, lambda x: x.at[insert_bucket].set(slot), table_slots)
    count_next = jnp.where(has_match, count, jnp.minimum(count + 1, jnp.asarray(capacity, dtype=jnp.int32)))
    return slot, GroupTableState(keys_next, occupied_next, table_next, count_next)


def _empty_output_like(template: Any, width: int):
    template = _ensure_dataclass_pytree(template)

    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        suffix = leaf.shape[1:] if leaf.ndim > 0 else ()
        return jnp.full((width,) + suffix, jnp.nan, dtype=leaf.dtype)

    return jax.tree_util.tree_map(alloc, template)


def _is_group_aligned_arg(value: jax.Array, source_width: int | None) -> bool:
    value = jnp.asarray(value)
    return value.ndim == 0 or source_width is None or value.shape[0] == source_width


def _slice_group_arg(value: jax.Array, idx: jax.Array, source_width: int | None = None) -> jax.Array:
    value = jnp.asarray(value)
    if value.ndim == 0:
        return value
    if not _is_group_aligned_arg(value, source_width):
        return value
    return jnp.take(value, idx, axis=0)


def _slice_sequence_group_arg(value: jax.Array, idx: jax.Array, source_width: int | None = None) -> jax.Array:
    value = jnp.asarray(value)
    if value.ndim <= 1:
        return value
    if source_width is not None and value.shape[1] != source_width:
        return value
    return jnp.take(value, idx, axis=1)


def _mask_group_arg(
    value: jax.Array,
    idx: jax.Array,
    mask: jax.Array,
    source_width: int | None = None,
) -> jax.Array:
    group_value = _slice_group_arg(value, idx, source_width)
    if jnp.asarray(group_value).ndim == 0 or not _is_group_aligned_arg(value, source_width):
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


def _group_suffix_shape(leaf: jax.Array, group_width: int) -> tuple[int, ...]:
    leaf = jnp.asarray(leaf)
    if leaf.ndim == 0 or leaf.shape[0] == 1 or leaf.shape[0] == group_width:
        return leaf.shape[1:]
    return leaf.shape



def _same_group_key_matrix(key_matrix: jax.Array) -> jax.Array:
    """Pairwise key equality for one group row, treating NaN keys as equal.

    Groupwise execution consumes one representative per distinct key and masks
    all group members with that key. Precomputing the pairwise matrix once keeps
    the per-representative loop generic for ndarray/object outputs while avoiding
    repeated key scans in the hot path.
    """
    return jax.vmap(lambda key: jax.vmap(lambda row_key: _same_key_vector(row_key, key))(key_matrix))(key_matrix)


def _first_occurrence_mask(same_keys: jax.Array) -> jax.Array:
    positions = jnp.arange(same_keys.shape[0], dtype=jnp.int32)
    earlier = positions[None, :] < positions[:, None]
    return ~jnp.any(same_keys & earlier, axis=1)

def _group_output_like(template: Any, group_width: int):
    template = _ensure_dataclass_pytree(template)

    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        suffix = _group_suffix_shape(leaf, group_width)
        return jnp.full((group_width,) + suffix, jnp.nan, dtype=leaf.dtype)

    return jax.tree_util.tree_map(alloc, template)


def _align_group_output(value: Any, group_width: int):
    value = _ensure_dataclass_pytree(value)

    def align_leaf(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.ndim == 0:
            return jnp.broadcast_to(leaf, (group_width,))
        if leaf.shape[0] == group_width:
            return leaf
        if leaf.shape[0] == 1:
            return jnp.broadcast_to(leaf[0], (group_width,) + leaf.shape[1:])
        return jnp.broadcast_to(leaf, (group_width,) + leaf.shape)

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
        suffix = _group_suffix_shape(leaf, group_width)
        return jnp.full((n_steps, group_width) + suffix, jnp.nan, dtype=leaf.dtype)

    return jax.tree_util.tree_map(alloc, template)


def _set_time_output(out: Any, t: jax.Array, value: Any):
    out = _ensure_dataclass_pytree(out)
    value = _ensure_dataclass_pytree(value)
    return jax.tree_util.tree_map(lambda dst, src: dst.at[t].set(src), out, value)


def _align_batch_group_output(value: Any, group_width: int):
    value = _ensure_dataclass_pytree(value)

    def align_leaf(leaf):
        leaf = jnp.asarray(leaf)
        if leaf.ndim == 1:
            return jnp.broadcast_to(leaf[:, None], (leaf.shape[0], group_width))
        if leaf.shape[1] == group_width:
            return leaf
        if leaf.shape[1] == 1:
            return jnp.broadcast_to(leaf[:, 0], (leaf.shape[0], group_width) + leaf.shape[2:])
        return jnp.broadcast_to(leaf[:, None], (leaf.shape[0], group_width) + leaf.shape[1:])

    return jax.tree_util.tree_map(align_leaf, value)


def _scatter_batch_group_output(out: Any, idx: jax.Array, group_values: Any):
    out = _ensure_dataclass_pytree(out)
    group_values = _ensure_dataclass_pytree(group_values)
    return jax.tree_util.tree_map(lambda dst, src: dst.at[:, idx].set(src), out, group_values)


def _slice_member_arg(
    value: jax.Array,
    idx: jax.Array,
    member_pos: jax.Array,
    source_width: int | None = None,
) -> jax.Array:
    group_value = _slice_group_arg(value, idx, source_width)
    if jnp.asarray(group_value).ndim == 0 or not _is_group_aligned_arg(value, source_width):
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

    def _source_width(self, fallback_width: int) -> int:
        if self.universe_groups is None:
            return fallback_width
        return max(max(group) for group in self.universe_groups) + 1

    def _group_sample(self, sample: jax.Array, group_width: int, source_width: int) -> jax.Array:
        sample = jnp.asarray(sample)
        if sample.ndim > 0 and sample.shape[0] != source_width:
            return jnp.zeros_like(sample)
        return jnp.zeros((group_width,) + sample.shape[1:], dtype=sample.dtype)

    def _row_arg_source_width(self, arg: jax.Array, idx: jax.Array) -> int:
        arg = jnp.asarray(arg)
        fallback_width = arg.shape[0] if arg.ndim > 0 else idx.shape[0]
        return self._source_width(fallback_width)

    def _sequence_arg_source_width(self, arg: jax.Array, idx: jax.Array) -> int:
        arg = jnp.asarray(arg)
        fallback_width = arg.shape[1] if arg.ndim > 1 else idx.shape[0]
        return self._source_width(fallback_width)

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

        sample_width = int(jnp.asarray(sample).shape[0]) if jnp.asarray(sample).ndim > 0 else 1
        source_width = self._source_width(sample_width)
        groups = self.universe_groups
        if groups is None:
            groups = (tuple(range(source_width)),)

        for group in groups:
            group_width = len(group)
            group_sample = self._group_sample(sample, group_width, source_width)
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
        sample_args = tuple(_slice_group_arg(arg, idx, self._row_arg_source_width(arg, idx)) for arg in args)
        _, template = self.inner_op.tick(sample_state, *sample_args)
        return _ensure_dataclass_pytree(template)

    def _tick_group_groupwise(
        self,
        table: GroupTableState,
        inner_state: Any,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        group_width = idx.shape[0]
        template = self._group_template(inner_state, idx, args)
        group_out0 = _group_output_like(template, group_width)
        same_keys = _same_group_key_matrix(key_matrix)
        first_occurrence = _first_occurrence_mask(same_keys)

        def body(member_pos, carry):
            table_c, inner_state_c, group_out_c = carry
            key = key_matrix[member_pos]
            already_processed = ~first_occurrence[member_pos]

            def process(process_carry):
                table_p, inner_state_p, group_out_p = process_carry

                slot, table_found = _lookup_or_insert_slot(table_p, key)

                mask = same_keys[member_pos]
                group_args = tuple(_mask_group_arg(arg, idx, mask, self._row_arg_source_width(arg, idx)) for arg in args)

                selected_state = _tree_take_slot(inner_state_p, slot) if inner_state_p is not None else None
                updated_state, local_out = self.inner_op.tick(selected_state, *group_args)

                inner_state_next = (
                    _tree_set_slot(inner_state_p, slot, updated_state)
                    if inner_state_p is not None and updated_state is not None
                    else inner_state_p
                )

                aligned_out = _align_group_output(local_out, group_width)
                group_out_next = _mask_group_output(group_out_p, aligned_out, mask)

                return table_found, inner_state_next, group_out_next

            return jax.lax.cond(
                already_processed,
                lambda x: x,
                process,
                (table_c, inner_state_c, group_out_c),
            )

        return jax.lax.fori_loop(
            0,
            group_width,
            body,
            (table, inner_state, group_out0),
        )

    def _tick_group_elementwise(
        self,
        table: GroupTableState,
        inner_state: Any,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        group_width = idx.shape[0]
        template = self._group_template(inner_state, idx, args)
        group_out0 = _group_output_like(template, group_width)

        def body(member_pos, carry):
            table_c, inner_state_c, group_out_c = carry
            key = key_matrix[member_pos]
            slot, table_found = _lookup_or_insert_slot(table_c, key)

            selected_state = _tree_take_slot(inner_state_c, slot)
            member_args = tuple(_slice_member_arg(arg, idx, member_pos, self._row_arg_source_width(arg, idx)) for arg in args)
            updated_state, local_out = self.inner_op.tick(selected_state, *member_args)

            inner_state_next = _tree_set_slot(inner_state_c, slot, updated_state) if updated_state is not None else inner_state_c
            group_out_next = _set_member_output(group_out_c, member_pos, local_out)
            return table_found, inner_state_next, group_out_next

        return jax.lax.fori_loop(
            0,
            group_width,
            body,
            (table, inner_state, group_out0),
        )

    def _inner_graph_is_memberwise(self) -> bool:
        nodes = getattr(self.inner_op, "nodes", None)
        if nodes is None:
            return False
        allowed_cpp_names = {
            "add", "sub", "mul", "div", "mod", "pow", "floordiv",
            "eq", "ne", "lt", "gt", "and", "or", "xor", "fillna", "where",
            "abs", "ln", "ceil", "floor", "round", "exp", "sign", "arctan",
            "isnan", "purify", "fraction",
        }
        allowed_stateful = {"CumsumOp", "EwmOp", "FFillOp"}
        for node in nodes:
            op = node.op
            name = type(op).__name__
            if name in {"InputOp", "LiteralOp"}:
                continue
            if name in allowed_stateful:
                if op.output_width not in (None, 1):
                    return False
                continue
            cpp_name = getattr(op, "cpp_name", None)
            if cpp_name in allowed_cpp_names and op.output_width in (None, 1):
                continue
            return False
        return True

    def _tick_group_memberwise(
        self,
        runtime: GroupRuntimeState,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        group_width = idx.shape[0]
        sample_state = (
            _tree_take_slot_member(runtime.inner_state, jnp.asarray(0, dtype=jnp.int32), jnp.asarray(0, dtype=jnp.int32))
            if runtime.inner_state is not None
            else None
        )
        sample_args = tuple(_slice_member_arg(arg, idx, jnp.asarray(0, dtype=jnp.int32), self._row_arg_source_width(arg, idx)) for arg in args)
        _, template = self.inner_op.tick(sample_state, *sample_args)
        group_out0 = _group_output_like(template, group_width)

        def body(member_pos, carry):
            table_c, inner_state_c, group_out_c = carry
            key = key_matrix[member_pos]
            slot, table_next = _lookup_or_insert_slot(table_c, key)
            member_args = tuple(_slice_member_arg(arg, idx, member_pos, self._row_arg_source_width(arg, idx)) for arg in args)
            selected_state = (
                _tree_take_slot_member(inner_state_c, slot, member_pos)
                if inner_state_c is not None
                else None
            )
            updated_state, local_out = self.inner_op.tick(selected_state, *member_args)
            inner_state_next = (
                _tree_set_slot_member(inner_state_c, slot, member_pos, updated_state)
                if inner_state_c is not None and updated_state is not None
                else inner_state_c
            )
            group_out_next = _set_member_output(group_out_c, member_pos, local_out)
            return table_next, inner_state_next, group_out_next

        table_next, inner_state_next, group_out = jax.lax.fori_loop(
            0,
            group_width,
            body,
            (runtime.table, runtime.inner_state, group_out0),
        )
        return GroupRuntimeState(table_next, runtime.cache, inner_state_next), group_out

    def _tick_group_cached_single_key(
        self,
        runtime: GroupRuntimeState,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        key = key_matrix[0]
        group_width = idx.shape[0]
        group_args = tuple(_slice_group_arg(arg, idx, self._row_arg_source_width(arg, idx)) for arg in args)
        cache_hit = runtime.cache.valid & _same_key_vector(runtime.cache.key, key)

        def hit(_):
            return runtime.table, runtime.inner_state, runtime.cache.slot, runtime.cache.inner_state

        def miss(_):
            flushed_state = _flush_cached_inner_state(
                runtime.inner_state,
                runtime.cache.inner_state,
                runtime.cache.slot,
                runtime.cache.valid,
            )
            slot, table_next = _lookup_or_insert_slot(runtime.table, key)
            selected_state = _tree_take_slot(flushed_state, slot) if flushed_state is not None else None
            return table_next, flushed_state, slot, selected_state

        table_c, inner_state_c, slot_c, selected_state = jax.lax.cond(
            cache_hit,
            hit,
            miss,
            operand=None,
        )

        updated_state, local_out = self.inner_op.tick(selected_state, *group_args)
        cached_state_next = updated_state if updated_state is not None else selected_state
        return (
            GroupRuntimeState(
                table=table_c,
                cache=GroupCacheState(key, slot_c, jnp.asarray(True), cached_state_next),
                inner_state=inner_state_c,
            ),
            _align_group_output(local_out, group_width),
        )

    def _tick_group_groupwise_uncached(
        self,
        runtime: GroupRuntimeState,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        flushed_state = _flush_cached_inner_state(
            runtime.inner_state,
            runtime.cache.inner_state,
            runtime.cache.slot,
            runtime.cache.valid,
        )
        table_next, inner_state_next, group_out = self._tick_group_groupwise(
            runtime.table,
            flushed_state,
            idx,
            key_matrix,
            args,
        )
        return (
            GroupRuntimeState(
                table=table_next,
                cache=GroupCacheState(
                    runtime.cache.key,
                    runtime.cache.slot,
                    jnp.asarray(False),
                    runtime.cache.inner_state,
                ),
                inner_state=inner_state_next,
            ),
            group_out,
        )

    def _tick_group(
        self,
        runtime: GroupRuntimeState,
        idx: jax.Array,
        key_matrix: jax.Array,
        args: tuple[jax.Array, ...],
    ):
        # The memberwise path is only profitable when each member owns keyed
        # inner state.  Stateless RHS graphs are faster through the generic
        # groupwise path because it evaluates one vectorized inner tick per
        # distinct key instead of one scalar tick per member.
        if runtime.inner_state is not None and self._inner_graph_is_memberwise():
            return self._tick_group_memberwise(runtime, idx, key_matrix, args)

        # Formula-compiled groupby RHS graphs initialize state with a leading
        # group axis, so cross-sectional ops must see every member in the key.
        # Direct grouped-apply ops with scalar keyed state advance per member;
        # repeated keys in one cross-section should therefore tick sequentially.
        if runtime.inner_state is not None and not _tree_has_group_axis(_tree_take_slot(runtime.inner_state, jnp.asarray(0, dtype=jnp.int32)), idx.shape[0]):
            table_next, inner_state_next, group_out = self._tick_group_elementwise(
                runtime.table,
                runtime.inner_state,
                idx,
                key_matrix,
                args,
            )
            return (
                GroupRuntimeState(table_next, runtime.cache, inner_state_next),
                group_out,
            )

        if runtime.inner_state is None:
            table_next, inner_state_next, group_out = self._tick_group_groupwise(
                runtime.table,
                runtime.inner_state,
                idx,
                key_matrix,
                args,
            )
            return (
                GroupRuntimeState(table_next, runtime.cache, inner_state_next),
                group_out,
            )

        return jax.lax.cond(
            _all_same_group_key(key_matrix),
            lambda _: self._tick_group_cached_single_key(
                runtime,
                idx,
                key_matrix,
                args,
            ),
            lambda _: self._tick_group_groupwise_uncached(
                runtime,
                idx,
                key_matrix,
                args,
            ),
            operand=None,
        )

    def _batch_key_matrix_for_group(
        self,
        idx: jax.Array,
        key_cols: tuple[jax.Array, ...],
        n_steps: int,
    ) -> jax.Array:
        if self.n_keys == 0:
            return jnp.zeros((n_steps, idx.shape[0], 0), dtype=jnp.float64)

        return jnp.stack(
            tuple(jnp.take(jnp.asarray(key_col, dtype=jnp.float64), idx, axis=1) for key_col in key_cols),
            axis=2,
        )

    def _scan_group_rows(
        self,
        runtime: GroupRuntimeState,
        idx: jax.Array,
        key_seq: jax.Array,
        args_seq: tuple[jax.Array, ...],
    ):
        n_steps = key_seq.shape[0]
        group_width = idx.shape[0]
        sample_args = tuple(_slice_group_arg(arg[0], idx, self._sequence_arg_source_width(arg, idx)) for arg in args_seq)
        sample_state = _tree_take_slot(runtime.inner_state, jnp.asarray(0, dtype=jnp.int32)) if runtime.inner_state is not None else None
        _, template = self.inner_op.tick(sample_state, *sample_args)
        out0 = _empty_batch_group_output_like(template, n_steps, group_width)

        def step(runtime_c, row_values):
            key_matrix, row_args = row_values
            runtime_n, group_out = self._tick_group(runtime_c, idx, key_matrix, row_args)
            return runtime_n, group_out

        return jax.lax.scan(
            step,
            runtime,
            (key_seq, tuple(args_seq)),
            unroll=32,
        )

    def _scan_group_runs(
        self,
        runtime: GroupRuntimeState,
        idx: jax.Array,
        key_seq: jax.Array,
        args_seq: tuple[jax.Array, ...],
        require_uniform: bool = False,
    ):
        n_steps = key_seq.shape[0]
        group_width = idx.shape[0]
        sample_args = tuple(_slice_group_arg(arg[0], idx, self._sequence_arg_source_width(arg, idx)) for arg in args_seq)
        sample_state = _tree_take_slot(runtime.inner_state, jnp.asarray(0, dtype=jnp.int32))
        _, template = self.inner_op.tick(sample_state, *sample_args)
        out0 = _empty_batch_group_output_like(template, n_steps, group_width)

        def find_run_end(start, key):
            def cond(i):
                return (i < n_steps) & _all_same_group_key(key_seq[i]) & _same_key_vector(key_seq[i, 0], key)

            def body(i):
                return i + 1

            return jax.lax.while_loop(cond, body, start)

        def process_uniform_run(carry):
            pos_c, runtime_c, out_c = carry
            key = key_seq[pos_c, 0]
            run_end = find_run_end(pos_c, key)
            cache_hit = runtime_c.cache.valid & _same_key_vector(runtime_c.cache.key, key)

            def hit(_):
                return runtime_c.table, runtime_c.inner_state, runtime_c.cache.slot, runtime_c.cache.inner_state

            def miss(_):
                flushed_state = _flush_cached_inner_state(
                    runtime_c.inner_state,
                    runtime_c.cache.inner_state,
                    runtime_c.cache.slot,
                    runtime_c.cache.valid,
                )
                slot, table_next = _lookup_or_insert_slot(runtime_c.table, key)
                selected_state = _tree_take_slot(flushed_state, slot)
                return table_next, flushed_state, slot, selected_state

            table_s, inner_state_s, slot_s, selected_state = jax.lax.cond(
                cache_hit,
                hit,
                miss,
                operand=None,
            )

            def row_body(t, row_carry):
                state_c, out_rows_c = row_carry
                group_args = tuple(_slice_group_arg(arg[t], idx, self._sequence_arg_source_width(arg, idx)) for arg in args_seq)
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
                GroupRuntimeState(
                    table=table_s,
                    cache=GroupCacheState(key, slot_s, jnp.asarray(True), cached_state_next),
                    inner_state=inner_state_s,
                ),
                out_next,
            )

        def process_mixed_row(carry):
            pos_c, runtime_c, out_c = carry
            row_args = tuple(arg[pos_c] for arg in args_seq)
            runtime_n, group_out = self._tick_group(runtime_c, idx, key_seq[pos_c], row_args)
            return (
                pos_c + 1,
                runtime_n,
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
                runtime,
                out0,
            ),
        )

    def _scan_static_universe_group(
        self,
        state: GroupByState,
        group_i: int,
        idx: jax.Array,
        args_seq: tuple[jax.Array, ...],
    ):
        runtime = _runtime_from_group_state(state, group_i)
        selected_state = (
            _tree_take_slot(runtime.inner_state, jnp.asarray(0, dtype=jnp.int32))
            if runtime.inner_state is not None
            else None
        )
        group_args_seq = tuple(
            _slice_sequence_group_arg(arg, idx, self._sequence_arg_source_width(arg, idx))
            for arg in args_seq
        )
        updated_state, group_out = self.inner_op.scan_batch(selected_state, *group_args_seq)
        group_out = _align_batch_group_output(group_out, idx.shape[0])
        inner_state_next = (
            _tree_set_slot(runtime.inner_state, jnp.asarray(0, dtype=jnp.int32), updated_state)
            if runtime.inner_state is not None and updated_state is not None
            else runtime.inner_state
        )
        return GroupRuntimeState(runtime.table, runtime.cache, inner_state_next), group_out

    def scan_batch(self, state: GroupByState, *child_sequences: jax.Array):
        key_cols = tuple(jnp.asarray(child_sequences[i]) for i in range(self.n_keys))
        args_seq = tuple(jnp.asarray(v) for v in child_sequences[self.n_keys:])
        n_steps = args_seq[0].shape[0] if args_seq else key_cols[0].shape[0]
        fallback_width = args_seq[0].shape[1] if args_seq and args_seq[0].ndim > 1 else key_cols[0].shape[1]
        width = self._source_width(fallback_width)

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
            if self.n_keys == 0 and _runtime_from_group_state(state, group_i).inner_state is not None:
                runtime_i, group_out = self._scan_static_universe_group(state, group_i, idx, args_seq)
                if out is None:
                    out = _empty_batch_output_like(group_out, n_steps, width)
                out = _scatter_batch_group_output(out, idx, group_out)

                new_keys.append(runtime_i.table.keys)
                new_occupied.append(runtime_i.table.occupied)
                new_table_slots.append(runtime_i.table.table_slots)
                new_counts.append(runtime_i.table.count)
                new_cached_keys.append(runtime_i.cache.key)
                new_cached_slots.append(runtime_i.cache.slot)
                new_cache_valid.append(runtime_i.cache.valid)
                new_cached_inner_states.append(runtime_i.cache.inner_state)
                new_inner_states.append(runtime_i.inner_state)
                continue

            runtime = _runtime_from_group_state(state, group_i)
            key_seq = self._batch_key_matrix_for_group(idx, key_cols, n_steps)
            group_args_seq = args_seq
            use_run_scan = runtime.inner_state is not None and _tree_has_group_axis(
                _tree_take_slot(runtime.inner_state, jnp.asarray(0, dtype=jnp.int32)),
                idx.shape[0],
            )

            if use_run_scan:
                all_uniform = jnp.all(jax.vmap(_all_same_group_key)(key_seq))

                def run_scan(require_uniform):
                    _, runtime_r, group_out_r = self._scan_group_runs(
                        runtime,
                        idx,
                        key_seq,
                        group_args_seq,
                        require_uniform=require_uniform,
                    )
                    return runtime_r, group_out_r

                runtime_i, group_out = jax.lax.cond(
                    all_uniform,
                    lambda _: run_scan(True),
                    lambda _: run_scan(False),
                    operand=None,
                )
            else:
                runtime_i, group_out = self._scan_group_rows(
                    runtime,
                    idx,
                    key_seq,
                    group_args_seq,
                )

            if out is None:
                out = _empty_batch_output_like(group_out, n_steps, width)
            out = _scatter_batch_group_output(out, idx, group_out)

            new_keys.append(runtime_i.table.keys)
            new_occupied.append(runtime_i.table.occupied)
            new_table_slots.append(runtime_i.table.table_slots)
            new_counts.append(runtime_i.table.count)
            new_cached_keys.append(runtime_i.cache.key)
            new_cached_slots.append(runtime_i.cache.slot)
            new_cache_valid.append(runtime_i.cache.valid)
            new_cached_inner_states.append(runtime_i.cache.inner_state)
            new_inner_states.append(runtime_i.inner_state)

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

        fallback_width = args[0].shape[0] if args and args[0].ndim > 0 else key_cols[0].shape[0]
        width = self._source_width(fallback_width)

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
            runtime_i, group_out = self._tick_group(
                _runtime_from_group_state(state, group_i),
                idx,
                key_matrix,
                args,
            )

            if out is None:
                out = _empty_output_like(group_out, width)
            out = _scatter_group_output(out, idx, group_out)

            new_keys.append(runtime_i.table.keys)
            new_occupied.append(runtime_i.table.occupied)
            new_table_slots.append(runtime_i.table.table_slots)
            new_counts.append(runtime_i.table.count)
            new_cached_keys.append(runtime_i.cache.key)
            new_cached_slots.append(runtime_i.cache.slot)
            new_cache_valid.append(runtime_i.cache.valid)
            new_cached_inner_states.append(runtime_i.cache.inner_state)
            new_inner_states.append(runtime_i.inner_state)

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
