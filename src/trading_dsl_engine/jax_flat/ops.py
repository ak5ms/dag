from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


class Op:
    output_kind: str = "vector"
    is_stateful: bool = False

    def init_state(self, sample: jax.Array):
        return None

    def tick(self, state: Any, *child_values: jax.Array):
        del state, child_values
        raise NotImplementedError


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EwmState:
    value: jax.Array
    initialized: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CumsumState:
    value: jax.Array
    initialized: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class GroupbyState:
    by_key: dict[tuple[float | str, ...], Any]


@dataclass(frozen=True)
class InputOp(Op):
    input_index: int


@dataclass(frozen=True)
class LiteralOp(Op):
    value: float
    output_kind: str = "scalar"


@dataclass(frozen=True)
class NaryOp(Op):
    fn: Callable[..., jax.Array]
    output_kind: str = "vector"

    def tick(self, state: Any, *child_values: jax.Array):
        del state
        return None, self.fn(*child_values)


@dataclass(frozen=True)
class EwmOp(Op):
    span: float
    output_kind: str = "vector"
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        return EwmState(value=jnp.zeros_like(sample), initialized=jnp.zeros_like(sample, dtype=bool))

    def tick(self, state: EwmState, *child_values: jax.Array):
        x = child_values[0]
        value, initialized = state.value, state.initialized
        alpha = 2.0 / (self.span + 1.0)
        valid = jnp.isfinite(x)
        init_or_valid = initialized | valid
        blended = alpha * x + (1.0 - alpha) * value
        out = jnp.where(initialized, blended, x)
        out = jnp.where(init_or_valid, jnp.where(valid, out, value), jnp.nan)
        return EwmState(value=out, initialized=init_or_valid), out


@dataclass(frozen=True)
class CumsumOp(Op):
    output_kind: str = "vector"
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        return CumsumState(value=jnp.zeros_like(sample), initialized=jnp.zeros_like(sample, dtype=bool))

    def tick(self, state: CumsumState, *child_values: jax.Array):
        x = child_values[0]
        value, initialized = state.value, state.initialized
        valid = jnp.isfinite(x)
        init_or_valid = initialized | valid
        prev = jnp.where(initialized, value, 0.0)
        accum = prev + jnp.where(valid, x, 0.0)
        out = jnp.where(init_or_valid, jnp.where(valid, accum, value), jnp.nan)
        return CumsumState(value=out, initialized=init_or_valid), out


def _normalize_key(raw_key: tuple[float, ...]) -> tuple[float | str, ...]:
    return tuple("__nan__" if jnp.isnan(v) else v for v in raw_key)


def _concat_rows(rows: list[Any]):
    first = rows[0]
    if isinstance(first, jax.Array):
        return jnp.concatenate([jnp.asarray(r) for r in rows], axis=0)
    if hasattr(first, "__dataclass_fields__"):
        return jax.tree_util.tree_map(lambda *vals: jnp.concatenate([jnp.asarray(v) for v in vals], axis=0), *rows)
    return rows


@dataclass(frozen=True)
class GroupbyOp(Op):
    inner_op: Op
    n_keys: int
    universe_groups: tuple[tuple[int, ...], ...] = ()
    output_kind: str = "vector"
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        return GroupbyState(by_key={})

    def _group_by_col(self, width: int):
        if not self.universe_groups:
            return None
        group_by_col = np.full((width,), -1, dtype=np.int32)
        for gid, cols in enumerate(self.universe_groups):
            for c in cols:
                if 0 <= c < width:
                    group_by_col[c] = gid
        return group_by_col

    def tick(self, state: GroupbyState, *child_values: jax.Array):
        key_cols = tuple(np.asarray(child_values[i]) for i in range(self.n_keys))
        args_np = tuple(np.asarray(v) for v in child_values[self.n_keys :])
        n = int(args_np[0].shape[0])
        group_by_col = self._group_by_col(n)

        # Build stable Python hash keys once, then process in key-buckets.
        keys: list[tuple[float | str, ...]] = []
        for i in range(n):
            raw_key = tuple(float(col[i]) for col in key_cols)
            if group_by_col is not None:
                raw_key = (float(group_by_col[i]),) + raw_key
            keys.append(_normalize_key(raw_key))

        by_key = dict(state.by_key)
        out_buffer = None

        unique_order: list[tuple[float | str, ...]] = []
        slot_map: dict[tuple[float | str, ...], list[int]] = {}
        for i, key in enumerate(keys):
            if key not in slot_map:
                slot_map[key] = []
                unique_order.append(key)
            slot_map[key].append(i)

        for key in unique_order:
            slots = np.asarray(slot_map[key], dtype=np.int32)
            prev_state = by_key.get(key)
            group_args = tuple(jnp.asarray(a[slots]) if getattr(a, "ndim", 0) > 0 else jnp.asarray(a) for a in args_np)
            next_state = prev_state
            group_rows: list[Any] = []
            # Keep generic semantics identical to per-row ticking but reduce global overhead.
            for r in range(group_args[0].shape[0]):
                row_args = tuple(ga[r : r + 1] if getattr(ga, "ndim", 0) > 0 else ga for ga in group_args)
                if next_state is None and self.inner_op.is_stateful:
                    next_state = self.inner_op.init_state(row_args[0])
                next_state, row_out = self.inner_op.tick(next_state, *row_args)
                group_rows.append(row_out)
            by_key[key] = next_state
            group_out = _concat_rows(group_rows)

            if out_buffer is None:
                if isinstance(group_out, jax.Array):
                    out_buffer = jnp.full((n,) + group_out.shape[1:], jnp.nan, dtype=group_out.dtype)
                elif hasattr(group_out, "__dataclass_fields__"):
                    out_buffer = jax.tree_util.tree_map(
                        lambda leaf: jnp.full((n,) + leaf.shape[1:], jnp.nan, dtype=leaf.dtype),
                        group_out,
                    )
                else:
                    out_buffer = [None] * n

            if isinstance(out_buffer, jax.Array):
                out_buffer = out_buffer.at[slots].set(group_out)
            elif hasattr(group_out, "__dataclass_fields__"):
                out_buffer = jax.tree_util.tree_map(lambda dst, src: dst.at[slots].set(src), out_buffer, group_out)
            else:
                for pos, slot in enumerate(slots.tolist()):
                    out_buffer[slot] = group_out[pos]

        return GroupbyState(by_key=by_key), out_buffer


def _nan_cmp(a, b, pred):
    return jnp.where(jnp.isnan(a) | jnp.isnan(b), jnp.nan, jnp.where(pred, 1.0, 0.0))


def _xstd(x):
    valid = jnp.isfinite(x)
    safe = jnp.where(valid, x, 0.0)
    count = jnp.maximum(jnp.sum(valid).astype(jnp.float64), 1.0)
    mean = jnp.sum(safe) / count
    centered = jnp.where(valid, x - mean, 0.0)
    var = jnp.sum(centered * centered) / count
    std = jnp.sqrt(jnp.maximum(var, 0.0))
    z = centered / jnp.where(std > 0.0, std, jnp.nan)
    return jnp.where(valid, z, jnp.nan)


def _xs_rank(x):
    valid = jnp.isfinite(x)
    n_valid = jnp.sum(valid).astype(jnp.int32)
    compact = jnp.where(valid, x, jnp.inf)
    sorted_compact = jnp.sort(compact)
    le_counts = jnp.minimum(jnp.searchsorted(sorted_compact, x, side="right"), n_valid)
    ranks = le_counts.astype(jnp.float64) / jnp.maximum(n_valid, 1).astype(jnp.float64)
    return jnp.where(valid, ranks, jnp.nan)


def _xs_sort(x):
    return jnp.sort(x)


OP_FACTORIES: dict[tuple[str, int], Callable[[], Op]] = {
    ("abs", 1): lambda: NaryOp(jnp.abs),
    ("ln", 1): lambda: NaryOp(jnp.log),
    ("exp", 1): lambda: NaryOp(jnp.exp),
    ("xs_rank", 1): lambda: NaryOp(_xs_rank),
    ("xs_sort", 1): lambda: NaryOp(_xs_sort),
    ("xstd", 1): lambda: NaryOp(_xstd),
    ("cumsum", 1): lambda: CumsumOp(),
    ("add", 2): lambda: NaryOp(lambda l, r: l + r),
    ("sub", 2): lambda: NaryOp(lambda l, r: l - r),
    ("mul", 2): lambda: NaryOp(lambda l, r: l * r),
    ("div", 2): lambda: NaryOp(lambda l, r: jnp.where(r == 0.0, jnp.nan, l / r)),
    ("gt", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l > r)),
    ("fillna", 2): lambda: NaryOp(lambda l, r: jnp.where(jnp.isnan(l), r, l)),
    ("where", 3): lambda: NaryOp(lambda c, t, f: jnp.where(c != 0.0, t, f)),
}
