from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from trading_dsl_engine.base.compiler import FormulaCompileError
from trading_dsl_engine.base.parser import Expr, Number


def _scalar_value(x):
    return jnp.ravel(x)[0]


def _nan_cmp(a, b, pred):
    return jnp.where(jnp.isnan(a) | jnp.isnan(b), jnp.nan, jnp.where(pred, 1.0, 0.0))


def _project_output(x, output_kind: str):
    if output_kind == "scalar":
        return _scalar_value(x)
    return x[:, 0] if x.ndim == 2 and x.shape[1] == 1 else x


def _children_init(children, n_instruments: int):
    return tuple(child.init_state(n_instruments) for child in children)


def _children_tick(children, states, frame2d):
    new_states = []
    values = []
    for child, state in zip(children, states):
        new_state, value = child.tick(state, frame2d)
        new_states.append(new_state)
        values.append(value)
    return tuple(new_states), tuple(values)


def _combine_kind(child_kinds: tuple[str, ...]) -> str:
    if "matrix" in child_kinds:
        return "matrix"
    if "vector" in child_kinds:
        return "vector"
    return "scalar"


def _tree_broadcast_slots(tree, n_slots: int):
    return jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (n_slots,) + x.shape), tree)


def _tree_take_slots(tree, slots):
    return jax.tree_util.tree_map(lambda x: x[slots], tree)


def _tree_set_slots(tree, slots, values):
    return jax.tree_util.tree_map(lambda x, y: x.at[slots].set(y), tree, values)


def _tree_has_leaves(tree) -> bool:
    return bool(jax.tree_util.tree_leaves(tree))


def _vmap_child_tick(child, selected_state, local_frames):
    if _tree_has_leaves(selected_state):
        return jax.vmap(lambda slot_state, frame: child.tick(slot_state, frame))(selected_state, local_frames)
    return jax.vmap(lambda frame: child.tick(selected_state, frame))(local_frames)




class TupleKeyOp(eqx.Module):
    children: tuple[Any, ...] = eqx.field(static=True)
    output_kind: str = eqx.field(static=True, default="vector")

    def init_state(self, n_instruments: int):
        return _children_init(self.children, n_instruments)

    def tick(self, state, frame2d):
        new_state, values = _children_tick(self.children, state, frame2d)
        n_cols = frame2d.shape[1]
        acc = jnp.zeros((n_cols,), dtype=jnp.float64)
        valid = jnp.ones((n_cols,), dtype=bool)
        for value in values:
            vector = jnp.full((n_cols,), value[0, 0], dtype=jnp.float64) if value.shape[0] == 1 else value[:, 0]
            finite = jnp.isfinite(vector)
            valid = valid & finite
            encoded = jnp.where(finite, vector, 0.0).astype(jnp.int64).astype(jnp.float64)
            acc = jnp.mod(acc * 1009.0 + encoded + 1024.0, 2147483647.0)
        return new_state, jnp.where(valid, acc, jnp.nan)[:, None]


class InputOp(eqx.Module):
    index: int = eqx.field(static=True)
    output_kind: str = eqx.field(static=True, default="vector")

    def init_state(self, n_instruments: int):
        return ()

    def tick(self, state, frame2d):
        return state, frame2d[self.index][:, None]


class LiteralOp(eqx.Module):
    value: float = eqx.field(static=True)
    output_kind: str = eqx.field(static=True, default="scalar")

    def init_state(self, n_instruments: int):
        return ()

    def tick(self, state, frame2d):
        return state, jnp.asarray([[self.value]], dtype=jnp.float64)


class LocalValueOp(eqx.Module):
    type_kind: str = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return ()

    def tick(self, state, frame2d):
        return state, frame2d[-1:, :].T


class UnaryOp(eqx.Module):
    child: Any = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return self.child.init_state(n_instruments)

    def apply(self, x):
        raise NotImplementedError

    def tick(self, state, frame2d):
        new_state, x = self.child.tick(state, frame2d)
        return new_state, self.apply(x)


class BinaryOp(eqx.Module):
    left: Any = eqx.field(static=True)
    right: Any = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return self.left.init_state(n_instruments), self.right.init_state(n_instruments)

    def apply(self, left, right):
        raise NotImplementedError

    def tick(self, state, frame2d):
        left_state, right_state = state
        new_left_state, left = self.left.tick(left_state, frame2d)
        new_right_state, right = self.right.tick(right_state, frame2d)
        return (new_left_state, new_right_state), self.apply(left, right)



class WhereOp(eqx.Module):
    condition: Any = eqx.field(static=True)
    true_value: Any = eqx.field(static=True)
    false_value: Any = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return _children_init((self.condition, self.true_value, self.false_value), n_instruments)

    def tick(self, state, frame2d):
        new_state, (condition, true_value, false_value) = _children_tick(
            (self.condition, self.true_value, self.false_value),
            state,
            frame2d,
        )
        return new_state, jnp.where(condition != 0.0, true_value, false_value)


class MeanOp(UnaryOp):
    def apply(self, x):
        return jnp.asarray([[jnp.nanmean(x)]], dtype=jnp.float64)


class OuterOp(UnaryOp):
    def apply(self, x):
        vector = x[:, 0]
        return vector[:, None] * vector[None, :]


class ColOp(eqx.Module):
    child: Any = eqx.field(static=True)
    index: int = eqx.field(static=True)
    output_kind: str = eqx.field(static=True, default="vector")

    def init_state(self, n_instruments: int):
        return self.child.init_state(n_instruments)

    def tick(self, state, frame2d):
        new_state, matrix = self.child.tick(state, frame2d)
        return new_state, matrix[:, self.index : self.index + 1]


class BsplineOp(eqx.Module):
    child: Any = eqx.field(static=True)
    n_basis: int = eqx.field(static=True)
    output_kind: str = eqx.field(static=True, default="matrix")

    def init_state(self, n_instruments: int):
        return self.child.init_state(n_instruments)

    def tick(self, state, frame2d):
        new_state, x = self.child.tick(state, frame2d)
        clipped = jnp.clip(x[:, 0], 0.0, 1.0)
        centers = jnp.arange(self.n_basis, dtype=jnp.float64) / self.n_basis
        sigma = 1.0 / self.n_basis
        dist = jnp.abs(clipped[:, None] - centers[None, :])
        circ_dist = jnp.minimum(dist, 1.0 - dist)
        values = jnp.exp(-0.5 * (circ_dist / sigma) ** 2)
        values = values / jnp.sum(values, axis=1, keepdims=True)
        return new_state, jnp.where(jnp.isnan(x), jnp.nan, values)


class XsRankOp(UnaryOp):
    def apply(self, x):
        vector = x[:, 0]
        valid = jnp.isfinite(vector)
        n_valid = jnp.sum(valid).astype(jnp.int32)
        compact = jnp.where(valid, vector, jnp.inf)
        sorted_compact = jnp.sort(compact)
        le_counts = jnp.minimum(jnp.searchsorted(sorted_compact, vector, side="right"), n_valid)
        ranks = le_counts.astype(jnp.float64) / jnp.maximum(n_valid, 1).astype(jnp.float64)
        return jnp.where(valid, ranks, jnp.nan)[:, None]


class EwmOp(eqx.Module):
    child: Any = eqx.field(static=True)
    span: Any = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return (
            _children_init((self.child, self.span), n_instruments),
            jnp.full((n_instruments, 1), jnp.nan),
            jnp.zeros((n_instruments, 1), dtype=bool),
        )

    def tick(self, state, frame2d):
        child_states, previous, initialized = state
        new_child_states, (x, span) = _children_tick((self.child, self.span), child_states, frame2d)
        alpha = 2.0 / (_scalar_value(span) + 1.0)
        valid = jnp.isfinite(x)
        out = jnp.where(initialized & valid, alpha * x + (1.0 - alpha) * previous, jnp.where(valid, x, previous))
        out = jnp.where(valid | initialized, out, jnp.nan)
        return (new_child_states, out, initialized | valid), out


class CumsumOp(eqx.Module):
    child: Any = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return (
            self.child.init_state(n_instruments),
            jnp.full((n_instruments, 1), jnp.nan),
            jnp.zeros((n_instruments, 1), dtype=bool),
        )

    def tick(self, state, frame2d):
        child_state, previous, initialized = state
        new_child_state, x = self.child.tick(child_state, frame2d)
        valid = jnp.isfinite(x)
        base = jnp.where(initialized, previous, 0.0)
        out = jnp.where(valid, base + x, previous)
        out = jnp.where(valid | initialized, out, jnp.nan)
        return (new_child_state, out, initialized | valid), out


class ShiftOp(eqx.Module):
    child: Any = eqx.field(static=True)
    lag: Any = eqx.field(static=True)
    max_size_source: Any = eqx.field(static=True)
    max_size: int = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return (
            _children_init((self.child, self.lag, self.max_size_source), n_instruments),
            jnp.full((self.max_size + 1, n_instruments, 1), jnp.nan),
            jnp.array(0, dtype=jnp.int64),
            jnp.array(0, dtype=jnp.int64),
        )

    def tick(self, state, frame2d):
        child_states, buffer, pos, count = state
        new_child_states, (x, lag, _) = _children_tick((self.child, self.lag, self.max_size_source), child_states, frame2d)
        cap = buffer.shape[0]
        lag_i = jnp.clip(jnp.asarray(_scalar_value(lag), dtype=jnp.int64), 0, cap - 1)
        read_pos = jnp.mod(pos - lag_i, cap)
        shifted = jnp.where(count >= lag_i, buffer[read_pos], jnp.nan)
        new_buffer = buffer.at[pos].set(x)
        return (
            new_child_states,
            new_buffer,
            jnp.mod(pos + 1, cap),
            jnp.minimum(count + 1, cap),
        ), shifted


class RollingQuantileOp(eqx.Module):
    child: Any = eqx.field(static=True)
    window_source: Any = eqx.field(static=True)
    quantile: Any = eqx.field(static=True)
    window: int = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return (
            _children_init((self.child, self.window_source, self.quantile), n_instruments),
            jnp.full((self.window, n_instruments, 1), jnp.nan),
            jnp.array(0, dtype=jnp.int64),
            jnp.array(0, dtype=jnp.int64),
        )

    def tick(self, state, frame2d):
        child_states, buffer, pos, count = state
        new_child_states, (x, _, quantile) = _children_tick((self.child, self.window_source, self.quantile), child_states, frame2d)
        new_buffer = buffer.at[pos].set(x)
        out = jnp.nanquantile(new_buffer, _scalar_value(quantile), axis=0, method="linear")
        out = jnp.where(count >= 0, out, jnp.nan)
        return (
            new_child_states,
            new_buffer,
            jnp.mod(pos + 1, buffer.shape[0]),
            jnp.minimum(count + 1, buffer.shape[0]),
        ), out


class FfillOp(eqx.Module):
    child: Any = eqx.field(static=True)
    limit: Any = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return (
            _children_init((self.child, self.limit), n_instruments),
            jnp.full((n_instruments, 1), jnp.nan),
            jnp.zeros((n_instruments, 1), dtype=jnp.int64),
            jnp.zeros((n_instruments, 1), dtype=bool),
        )

    def tick(self, state, frame2d):
        child_states, last, streak, seen = state
        new_child_states, (x, limit) = _children_tick((self.child, self.limit), child_states, frame2d)
        limit_i = jnp.maximum(jnp.asarray(_scalar_value(limit), dtype=jnp.int64), 0)
        valid = jnp.isfinite(x)
        next_last = jnp.where(valid, x, last)
        next_seen = seen | valid
        can_fill = (~valid) & next_seen & (streak < limit_i)
        out = jnp.where(valid, x, jnp.where(can_fill, last, jnp.nan))
        next_streak = jnp.where(valid, 0, jnp.where(can_fill, streak + 1, streak))
        return (new_child_states, next_last, next_streak, next_seen), out


class GroupByOp(eqx.Module):
    key: Any = eqx.field(static=True)
    child: Any = eqx.field(static=True)
    n_inputs: int = eqx.field(static=True)
    lhs: Any | None = eqx.field(static=True, default=None)
    capacity: int = eqx.field(static=True, default=4096)
    output_kind: str = eqx.field(static=True, default="vector")

    def init_state(self, n_instruments: int):
        return (
            _children_init((self.key, self.lhs), n_instruments) if self.lhs is not None else (self.key.init_state(n_instruments), ()),
            _tree_broadcast_slots(self.child.init_state(1), n_instruments * self.capacity),
            jnp.full((n_instruments, self.capacity), jnp.nan),
            jnp.zeros((n_instruments, self.capacity), dtype=bool),
        )

    def tick(self, state, frame2d):
        outer_state, child_state, keys, occupied = state
        if self.lhs is None:
            key_state, _ = outer_state
            new_key_state, key_values = self.key.tick(key_state, frame2d)
            source = frame2d[: self.n_inputs]
            new_outer_state = (new_key_state, ())
        else:
            new_outer_state, (key_values, lhs_values) = _children_tick((self.key, self.lhs), outer_state, frame2d)
            n_inputs = frame2d.shape[0]
            n_instruments = frame2d.shape[1]
            rhs_width = lhs_values.shape[1]
            source = jnp.full((n_instruments, n_inputs + 1, rhs_width), jnp.nan)
            source = source.at[:, :n_inputs, 0].set(frame2d.T)
            lhs_local = lhs_values if lhs_values.shape[0] > 1 else jnp.broadcast_to(lhs_values[0], (n_instruments, rhs_width))
            source = source.at[:, n_inputs, :].set(lhs_local)
        key_vector = key_values[:, 0]
        safe_key = jnp.where(jnp.isnan(key_vector), jnp.inf, key_vector)
        safe_keys = jnp.where(jnp.isnan(keys), jnp.inf, keys)
        matches = occupied & (safe_keys == safe_key[:, None])
        has_match = jnp.any(matches, axis=1)
        first_free = jnp.argmax(~occupied, axis=1)
        slot = jnp.where(has_match, jnp.argmax(matches, axis=1), first_free)
        row_idx = jnp.arange(frame2d.shape[1])
        flat_slot = row_idx * self.capacity + slot
        new_keys = keys.at[row_idx, slot].set(key_vector)
        new_occupied = occupied.at[row_idx, slot].set(True)

        selected_state = _tree_take_slots(child_state, flat_slot)
        local_frames = jnp.swapaxes(source, 0, 1) if self.lhs is None else source
        new_selected_state, grouped_out = _vmap_child_tick(self.child, selected_state, local_frames)
        new_child_state = _tree_set_slots(child_state, flat_slot, new_selected_state)
        return (new_outer_state, new_child_state, new_keys, new_occupied), grouped_out[:, 0, :]



class RidgeOp(eqx.Module):
    features: tuple[Any, ...] = eqx.field(static=True)
    y: Any = eqx.field(static=True)
    weights: Any = eqx.field(static=True)
    half_life: Any = eqx.field(static=True)
    ridge_lambda: Any = eqx.field(static=True)
    feature_widths: tuple[int, ...] = eqx.field(static=True)
    output_kind: str = eqx.field(static=True, default="object")

    def init_state(self, n_instruments: int):
        k = sum(self.feature_widths)
        children = self.features + (self.y, self.weights, self.half_life, self.ridge_lambda)
        return (
            _children_init(children, n_instruments),
            jnp.zeros((k, k), dtype=jnp.float64),
            jnp.zeros((k,), dtype=jnp.float64),
            jnp.zeros((k, k), dtype=bool),
            jnp.zeros((k,), dtype=bool),
            jnp.zeros((k, k), dtype=jnp.int64),
            jnp.zeros((k,), dtype=jnp.int64),
            jnp.zeros((k,), dtype=jnp.float64),
            jnp.full((n_instruments, 1), jnp.nan),
            jnp.array(0, dtype=jnp.int64),
        )

    def tick(self, state, frame2d):
        child_states, xx, xy, has_xx, has_xy, last_xx, last_xy, beta, _, t = state
        children = self.features + (self.y, self.weights, self.half_life, self.ridge_lambda)
        new_child_states, values = _children_tick(children, child_states, frame2d)
        feature_values = values[: len(self.features)]
        y2d, w2d, hl2d, lam2d = values[-4:]
        xmat = jnp.concatenate(feature_values, axis=1)
        y = y2d[:, 0]
        hl = _scalar_value(hl2d)
        lam = jnp.maximum(jnp.where(jnp.isnan(_scalar_value(lam2d)), 0.0, _scalar_value(lam2d)), 0.0)
        row_valid = jnp.isfinite(y) & jnp.all(jnp.isfinite(xmat), axis=1)
        preds = jnp.where(row_valid, xmat @ beta, jnp.nan)[:, None]
        xx_new, xy_new, xx_valid, xy_valid = _ridge_moments(xmat, y, w2d)
        rho = jnp.where((hl <= 0.0) | jnp.isnan(hl), 0.0, jnp.exp(jnp.log(0.5) / hl))
        alpha = jnp.clip(1.0 - rho, 0.0, 1.0)
        dt_xx = t - last_xx
        dt_xy = t - last_xy
        a_xx = alpha**dt_xx
        a_xy = alpha**dt_xy
        updated_xx = jnp.where(has_xx, xx * (1.0 - a_xx) + xx_new * a_xx, xx_new)
        updated_xy = jnp.where(has_xy, xy * (1.0 - a_xy) + xy_new * a_xy, xy_new)
        xx = jnp.where(xx_valid, updated_xx, xx)
        xy = jnp.where(xy_valid, updated_xy, xy)
        has_xx = has_xx | xx_valid
        has_xy = has_xy | xy_valid
        last_xx = jnp.where(xx_valid, t, last_xx)
        last_xy = jnp.where(xy_valid, t, last_xy)
        xx = 0.5 * (xx + xx.T)
        last_xx = jnp.maximum(last_xx, last_xx.T)
        has_xx = has_xx | has_xx.T
        system = xx + lam * jnp.diag(jnp.diag(xx))
        beta_new = jnp.linalg.solve(system, xy)
        beta = jnp.where(jnp.all(jnp.isfinite(beta_new)), beta_new, beta)
        return (new_child_states, xx, xy, has_xx, has_xy, last_xx, last_xy, beta, preds, t + 1), (
            beta,
            preds,
        )


class GetBetaOp(UnaryOp):
    output_kind: str = eqx.field(static=True, default="vector")

    def apply(self, ridge_state):
        beta, _ = ridge_state
        return beta[:, None]


class GetPredsOp(UnaryOp):
    output_kind: str = eqx.field(static=True, default="vector")

    def apply(self, ridge_state):
        _, preds = ridge_state
        return preds


def _ridge_moments(xmat, y, weights):
    valid_x = jnp.isfinite(xmat)
    valid_y = jnp.isfinite(y)
    x0 = jnp.where(valid_x, xmat, 0.0)
    y0 = jnp.where(valid_y, y, 0.0)
    n = xmat.shape[0]
    if weights.shape[0] == 1 and weights.shape[1] == 1:
        w = jnp.full((n,), weights[0, 0])
        return _ridge_vector_weight_moments(x0, y0, w, valid_x, valid_y, jnp.isfinite(w))
    if weights.shape[1] == 1:
        w = weights[:, 0]
        return _ridge_vector_weight_moments(x0, y0, jnp.where(jnp.isfinite(w), w, 0.0), valid_x, valid_y, jnp.isfinite(w))
    valid_w = jnp.isfinite(weights)
    w0 = jnp.where(valid_w, weights, 0.0)
    xx_new = x0.T @ w0 @ x0
    xy_new = x0.T @ (w0 @ y0)
    xx_valid = (valid_x.astype(jnp.int64).T @ (valid_w.astype(jnp.int64) @ valid_x.astype(jnp.int64))) > 0
    xy_valid = (valid_x.astype(jnp.int64).T @ (valid_w.astype(jnp.int64) @ valid_y.astype(jnp.int64))) > 0
    return xx_new, xy_new, xx_valid, xy_valid


def _ridge_vector_weight_moments(x0, y0, w, valid_x, valid_y, valid_w):
    xw = x0 * w[:, None]
    xx_new = x0.T @ xw
    xx_counts = valid_x.astype(jnp.int64).T @ (valid_x & valid_w[:, None]).astype(jnp.int64)
    xy_new = x0.T @ (w * y0)
    xy_counts = valid_x.astype(jnp.int64).T @ (valid_y & valid_w).astype(jnp.int64)
    return xx_new, xy_new, xx_counts > 0, xy_counts > 0




class UniverseDynamicGroupByOp(eqx.Module):
    key: Any = eqx.field(static=True)
    child: Any = eqx.field(static=True)
    groups: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    capacity: int = eqx.field(static=True, default=4096)
    output_kind: str = eqx.field(static=True, default="vector")

    def init_state(self, n_instruments: int):
        return tuple(
            (
                self.key.init_state(len(group)),
                _tree_broadcast_slots(self.child.init_state(len(group)), self.capacity),
                jnp.full((self.capacity,), jnp.nan),
                jnp.zeros((self.capacity,), dtype=bool),
            )
            for group in self.groups
        )

    def _tick_group(self, state, frame2d, group: tuple[int, ...]):
        key_state, child_state, keys, occupied = state
        idx = jnp.asarray(group, dtype=jnp.int64)
        group_frame = frame2d[:, idx]
        new_key_state, key_values = self.key.tick(key_state, group_frame)
        key_vector = key_values[:, 0]
        group_width = idx.shape[0]
        output_width = group_width if self.child.output_kind == "matrix" else 1
        group_result = jnp.full((group_width, output_width), jnp.nan)
        new_child_state = child_state
        new_keys = keys
        new_occupied = occupied

        for member_pos in range(len(group)):
            key = key_vector[member_pos]
            already_processed = jnp.any(key_vector[:member_pos] == key)
            matches = new_occupied & (new_keys == key)
            has_match = jnp.any(matches)
            first_free = jnp.argmax(~new_occupied)
            slot = jnp.where(has_match, jnp.argmax(matches), first_free)
            mask = key_vector == key
            local_frame = jnp.where(mask[None, :], group_frame, jnp.nan)

            def process(args):
                child_state_arg, keys_arg, occupied_arg, result_arg = args
                selected_state = _tree_take_slots(child_state_arg, slot)
                updated_state, child_out = self.child.tick(selected_state, local_frame)
                child_state_arg = _tree_set_slots(child_state_arg, slot, updated_state)
                keys_arg = keys_arg.at[slot].set(key)
                occupied_arg = occupied_arg.at[slot].set(True)
                value = (
                    jnp.broadcast_to(_scalar_value(child_out), (group_width, 1))
                    if child_out.shape[0] == 1
                    else child_out[:, :output_width]
                )
                result_arg = jnp.where(mask[:, None], value, result_arg)
                return child_state_arg, keys_arg, occupied_arg, result_arg

            new_child_state, new_keys, new_occupied, group_result = jax.lax.cond(
                already_processed,
                lambda args: args,
                process,
                (new_child_state, new_keys, new_occupied, group_result),
            )
        return (new_key_state, new_child_state, new_keys, new_occupied), idx, group_result

    def tick(self, state, frame2d):
        output_width = frame2d.shape[1] if self.child.output_kind == "matrix" else 1
        out = jnp.full((frame2d.shape[1], output_width), jnp.nan)
        new_states = []
        for group_state, group in zip(state, self.groups):
            new_state, idx, group_result = self._tick_group(group_state, frame2d, group)
            out = out.at[idx, :].set(group_result)
            new_states.append(new_state)
        return tuple(new_states), out


class UniverseGroupByOp(eqx.Module):
    child: Any = eqx.field(static=True)
    groups: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    output_kind: str = eqx.field(static=True, default="vector")

    def init_state(self, n_instruments: int):
        return tuple(self.child.init_state(len(group)) for group in self.groups)

    def tick(self, state, frame2d):
        out = jnp.full((frame2d.shape[1], 1), jnp.nan)
        new_states = []
        for group, child_state in zip(self.groups, state):
            idx = jnp.asarray(group, dtype=jnp.int64)
            sub_frame = frame2d[:, idx]
            new_state, group_out = self.child.tick(child_state, sub_frame)
            value = (
                jnp.broadcast_to(_scalar_value(group_out), (idx.shape[0], 1))
                if group_out.shape[0] == 1
                else group_out
            )
            out = out.at[idx, :].set(value)
            new_states.append(new_state)
        return tuple(new_states), out


_UNARY_OPS = {
    "abs": type("AbsOp", (UnaryOp,), {"apply": lambda self, x: jnp.abs(x)}),
    "ln": type("LnOp", (UnaryOp,), {"apply": lambda self, x: jnp.log(x)}),
    "ceil": type("CeilOp", (UnaryOp,), {"apply": lambda self, x: jnp.ceil(x)}),
    "floor": type("FloorOp", (UnaryOp,), {"apply": lambda self, x: jnp.floor(x)}),
    "exp": type("ExpOp", (UnaryOp,), {"apply": lambda self, x: jnp.exp(x)}),
    "sign": type("SignOp", (UnaryOp,), {"apply": lambda self, x: jnp.sign(x)}),
    "arctan": type("ArctanOp", (UnaryOp,), {"apply": lambda self, x: jnp.arctan(x)}),
    "isnan": type("IsNanOp", (UnaryOp,), {"apply": lambda self, x: jnp.where(jnp.isnan(x), 1.0, 0.0)}),
    "purify": type("PurifyOp", (UnaryOp,), {"apply": lambda self, x: jnp.where(jnp.isfinite(x), x, jnp.nan)}),
    "fraction": type("FractionOp", (UnaryOp,), {"apply": lambda self, x: x - jnp.floor(x)}),
}

_BINARY_OPS = {
    "add": type("AddOp", (BinaryOp,), {"apply": lambda self, left, right: left + right}),
    "sub": type("SubOp", (BinaryOp,), {"apply": lambda self, left, right: left - right}),
    "mul": type("MulOp", (BinaryOp,), {"apply": lambda self, left, right: left * right}),
    "mod": type("ModOp", (BinaryOp,), {"apply": lambda self, left, right: jnp.mod(left, right)}),
    "pow": type("PowOp", (BinaryOp,), {"apply": lambda self, left, right: left**right}),
    "div": type("DivOp", (BinaryOp,), {"apply": lambda self, left, right: jnp.where(right == 0.0, jnp.nan, left / right)}),
    "floordiv": type("FloorDivOp", (BinaryOp,), {"apply": lambda self, left, right: jnp.where(right == 0.0, jnp.nan, left // right)}),
    "eq": type("EqOp", (BinaryOp,), {"apply": lambda self, left, right: _nan_cmp(left, right, left == right)}),
    "ne": type("NeOp", (BinaryOp,), {"apply": lambda self, left, right: _nan_cmp(left, right, left != right)}),
    "lt": type("LtOp", (BinaryOp,), {"apply": lambda self, left, right: _nan_cmp(left, right, left < right)}),
    "gt": type("GtOp", (BinaryOp,), {"apply": lambda self, left, right: _nan_cmp(left, right, left > right)}),
    "and": type("AndOp", (BinaryOp,), {"apply": lambda self, left, right: _nan_cmp(left, right, (left != 0.0) & (right != 0.0))}),
    "and_": type("AndOp", (BinaryOp,), {"apply": lambda self, left, right: _nan_cmp(left, right, (left != 0.0) & (right != 0.0))}),
    "or": type("OrOp", (BinaryOp,), {"apply": lambda self, left, right: _nan_cmp(left, right, (left != 0.0) | (right != 0.0))}),
    "or_": type("OrOp", (BinaryOp,), {"apply": lambda self, left, right: _nan_cmp(left, right, (left != 0.0) | (right != 0.0))}),
    "xor": type("XorOp", (BinaryOp,), {"apply": lambda self, left, right: _nan_cmp(left, right, (left != 0.0) ^ (right != 0.0))}),
    "fillna": type("FillNaOp", (BinaryOp,), {"apply": lambda self, left, right: jnp.where(jnp.isnan(left), right, left)}),
}


def _literal_arg(expr: Expr, op_name: str, position: int) -> float:
    if not isinstance(expr, Number):
        raise FormulaCompileError(f"JAX backend requires literal argument {position} for {op_name}")
    return expr.value



def _feature_width(op) -> int:
    if getattr(op, "output_kind", "vector") in ("scalar", "vector"):
        return 1
    if isinstance(op, BsplineOp):
        return op.n_basis
    if isinstance(op, ColOp):
        return 1
    return 1

def _combine_output_kind(children: tuple[Any, ...]) -> str:
    return _combine_kind(tuple(child.output_kind for child in children))


def _build_unary(op_cls, args, children):
    return op_cls(children[0], children[0].output_kind)


def _build_binary(op_cls, args, children):
    return op_cls(children[0], children[1], _combine_output_kind(children))


def _build_where(args, children):
    return WhereOp(children[0], children[1], children[2], _combine_output_kind(children))


def _build_col(args, children):
    return ColOp(children[0], int(_literal_arg(args[1], "col", 2)))


def _build_bspline(args, children):
    return BsplineOp(children[0], int(_literal_arg(args[1], "bspline", 2)))


def _build_shift(args, children):
    max_size_arg = args[2] if len(args) > 2 else args[1]
    max_size = int(_literal_arg(max_size_arg, "shift", 3 if len(args) > 2 else 2))
    max_size_source = children[2] if len(children) > 2 else children[1]
    return ShiftOp(children[0], children[1], max_size_source, max(1, max_size), children[0].output_kind)


def _build_rolling_quantile(args, children):
    window = int(_literal_arg(args[1], "rolling_quantile", 2))
    return RollingQuantileOp(children[0], children[1], children[2], max(1, window), children[0].output_kind)


def _build_ffill(args, children):
    return FfillOp(children[0], children[1], children[0].output_kind)


def _build_ridge(args, children):
    has_weights = len(children) >= 5
    features = children[:-4] if has_weights else children[:-3]
    y = children[-4] if has_weights else children[-3]
    weights = children[-3] if has_weights else LiteralOp(1.0)
    half_life = children[-2]
    ridge_lambda = children[-1]
    return RidgeOp(features, y, weights, half_life, ridge_lambda, tuple(_feature_width(feature) for feature in features))


_CALL_BUILDERS = {
    **{(name, 1): (lambda args, children, op_cls=op_cls: _build_unary(op_cls, args, children)) for name, op_cls in _UNARY_OPS.items()},
    **{(name, 2): (lambda args, children, op_cls=op_cls: _build_binary(op_cls, args, children)) for name, op_cls in _BINARY_OPS.items()},
    ("where", 3): _build_where,
    ("mean", 1): lambda args, children: MeanOp(children[0], "scalar"),
    ("outer", 1): lambda args, children: OuterOp(children[0], "matrix"),
    ("col", 2): _build_col,
    ("bspline", 2): _build_bspline,
    ("xs_rank", 1): lambda args, children: XsRankOp(children[0], "vector"),
    ("ewm", 2): lambda args, children: EwmOp(children[0], children[1], children[0].output_kind),
    ("cumsum", 1): lambda args, children: CumsumOp(children[0], children[0].output_kind),
    ("shift", 2): _build_shift,
    ("shift", 3): _build_shift,
    ("rolling_quantile", 3): _build_rolling_quantile,
    ("ffill", 2): _build_ffill,
    ("Ridge", 4): _build_ridge,
    ("Ridge", 5): _build_ridge,
    ("Ridge", 6): _build_ridge,
    ("Ridge", 7): _build_ridge,
    ("Ridge", 8): _build_ridge,
    ("get_beta", 1): lambda args, children: GetBetaOp(children[0], "vector"),
    ("get_preds", 1): lambda args, children: GetPredsOp(children[0], "vector"),
}


def _make_call_op(fn: str, args: tuple[Expr, ...], children: tuple[Any, ...]):
    builder = _CALL_BUILDERS.get((fn, len(children)))
    if builder is None:
        raise FormulaCompileError(f"JAX backend does not support op '{fn}' yet")
    return builder(args, children)
