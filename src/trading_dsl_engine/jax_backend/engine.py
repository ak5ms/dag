from __future__ import annotations

from dataclasses import dataclass
import tempfile
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from trading_dsl_engine.compiler import CompileStats, FormulaCompileError
from trading_dsl_engine.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.parser import Call, Expr, Identifier, Number, Universe, parse_formula


@dataclass(frozen=True)
class JaxCompiledArtifact:
    compiled: "JaxProgram"
    input_names: tuple[str, ...]
    output_kind: str
    stats: CompileStats


class JaxProgram(eqx.Module):
    """Static Equinox module containing a compiled operator tree."""

    root: Any = eqx.field(static=True)
    n_inputs: int = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return self.root.init_state(n_instruments)

    def tick(self, state, frame2d):
        new_state, out = self.root.tick(state, frame2d)
        return new_state, _project_output(out, self.output_kind)

    def run_batch(self, inputs):
        n_instruments = inputs[0].shape[1]
        state0 = self.init_state(n_instruments)

        def step(state, rows):
            frame = jnp.stack(rows, axis=0)
            new_state, out = self.tick(state, frame)
            return new_state, out

        _, outputs = jax.lax.scan(step, state0, inputs)
        return outputs


@eqx.filter_jit
def _jit_tick(program: JaxProgram, state, frame2d):
    return program.tick(state, frame2d)


@eqx.filter_jit
def _jit_batch(program: JaxProgram, inputs):
    return program.run_batch(inputs)


class JaxEngineHandle:
    def __init__(self, compiled: JaxProgram, input_names: tuple[str, ...], output_kind: str):
        self.compiled = compiled
        self.input_names = input_names
        self.output_kind = output_kind
        self.output_code = {"scalar": 0, "vector": 1, "matrix": 2, "object": 3}[output_kind]
        self._state = None
        self._n_instruments = None

    def on_data(self, frame2d):
        frame = jnp.asarray(frame2d, dtype=jnp.float64)
        if self._state is None or self._n_instruments != frame.shape[1]:
            self._state = self.compiled.init_state(frame.shape[1])
            self._n_instruments = frame.shape[1]
        self._state, self._last = _jit_tick(self.compiled, self._state, frame)
        return self._last

    def emit(self):
        return np.asarray(self._last)


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
        return state, frame2d


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


class AbsOp(UnaryOp):
    def apply(self, x):
        return jnp.abs(x)


class LnOp(UnaryOp):
    def apply(self, x):
        return jnp.log(x)


class CeilOp(UnaryOp):
    def apply(self, x):
        return jnp.ceil(x)


class FloorOp(UnaryOp):
    def apply(self, x):
        return jnp.floor(x)


class ExpOp(UnaryOp):
    def apply(self, x):
        return jnp.exp(x)


class SignOp(UnaryOp):
    def apply(self, x):
        return jnp.sign(x)


class ArctanOp(UnaryOp):
    def apply(self, x):
        return jnp.arctan(x)


class IsNanOp(UnaryOp):
    def apply(self, x):
        return jnp.where(jnp.isnan(x), 1.0, 0.0)


class PurifyOp(UnaryOp):
    def apply(self, x):
        return jnp.where(jnp.isfinite(x), x, jnp.nan)


class FractionOp(UnaryOp):
    def apply(self, x):
        return x - jnp.floor(x)


class AddOp(BinaryOp):
    def apply(self, left, right):
        return left + right


class SubOp(BinaryOp):
    def apply(self, left, right):
        return left - right


class MulOp(BinaryOp):
    def apply(self, left, right):
        return left * right


class ModOp(BinaryOp):
    def apply(self, left, right):
        return jnp.mod(left, right)


class PowOp(BinaryOp):
    def apply(self, left, right):
        return left**right


class DivOp(BinaryOp):
    def apply(self, left, right):
        return jnp.where(right == 0.0, jnp.nan, left / right)


class FloorDivOp(BinaryOp):
    def apply(self, left, right):
        return jnp.where(right == 0.0, jnp.nan, left // right)


class EqOp(BinaryOp):
    def apply(self, left, right):
        return _nan_cmp(left, right, left == right)


class NeOp(BinaryOp):
    def apply(self, left, right):
        return _nan_cmp(left, right, left != right)


class LtOp(BinaryOp):
    def apply(self, left, right):
        return _nan_cmp(left, right, left < right)


class GtOp(BinaryOp):
    def apply(self, left, right):
        return _nan_cmp(left, right, left > right)


class AndOp(BinaryOp):
    def apply(self, left, right):
        return _nan_cmp(left, right, (left != 0.0) & (right != 0.0))


class OrOp(BinaryOp):
    def apply(self, left, right):
        return _nan_cmp(left, right, (left != 0.0) | (right != 0.0))


class XorOp(BinaryOp):
    def apply(self, left, right):
        return _nan_cmp(left, right, (left != 0.0) ^ (right != 0.0))


class FillNaOp(BinaryOp):
    def apply(self, left, right):
        return jnp.where(jnp.isnan(left), right, left)


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
        n_valid = jnp.sum(valid)
        le_counts = jnp.sum(
            (vector[None, :] <= vector[:, None]) & valid[None, :] & valid[:, None],
            axis=1,
        )
        ranks = le_counts / jnp.maximum(n_valid, 1)
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
        shifted = jnp.where(count > lag_i, buffer[read_pos], jnp.nan)
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


class GroupByOp(eqx.Module):
    key: Any = eqx.field(static=True)
    child: Any = eqx.field(static=True)
    n_inputs: int = eqx.field(static=True)
    capacity: int = eqx.field(static=True, default=4096)
    output_kind: str = eqx.field(static=True, default="vector")

    def init_state(self, n_instruments: int):
        return (
            self.key.init_state(n_instruments),
            self.child.init_state(n_instruments * self.capacity),
            jnp.full((n_instruments, self.capacity), jnp.nan),
            jnp.zeros((n_instruments, self.capacity), dtype=bool),
        )

    def tick(self, state, frame2d):
        key_state, child_state, keys, occupied = state
        new_key_state, key_values = self.key.tick(key_state, frame2d)
        key_vector = key_values[:, 0]
        if frame2d.shape[0] == self.n_inputs:
            source = frame2d
        else:
            source = frame2d[: self.n_inputs]
        matches = occupied & (keys == key_vector[:, None])
        has_match = jnp.any(matches, axis=1)
        first_free = jnp.argmax(~occupied, axis=1)
        slot = jnp.where(has_match, jnp.argmax(matches, axis=1), first_free)
        row_idx = jnp.arange(frame2d.shape[1])
        new_keys = keys.at[row_idx, slot].set(key_vector)
        new_occupied = occupied.at[row_idx, slot].set(True)

        grouped_frame = jnp.full((self.n_inputs, frame2d.shape[1] * self.capacity), jnp.nan)
        flat_slot = row_idx * self.capacity + slot
        grouped_frame = grouped_frame.at[:, flat_slot].set(source)
        new_child_state, grouped_out = self.child.tick(child_state, grouped_frame)
        out = grouped_out[flat_slot, :]
        return (new_key_state, new_child_state, new_keys, new_occupied), out


class ScopedGroupByOp(eqx.Module):
    key: Any = eqx.field(static=True)
    lhs: Any = eqx.field(static=True)
    child: Any = eqx.field(static=True)
    capacity: int = eqx.field(static=True, default=4096)
    output_kind: str = eqx.field(static=True, default="vector")

    def init_state(self, n_instruments: int):
        return (
            _children_init((self.key, self.lhs), n_instruments),
            self.child.init_state(n_instruments * self.capacity),
            jnp.full((n_instruments, self.capacity), jnp.nan),
            jnp.zeros((n_instruments, self.capacity), dtype=bool),
        )

    def tick(self, state, frame2d):
        outer_states, child_state, keys, occupied = state
        new_outer_states, (key_values, lhs_values) = _children_tick((self.key, self.lhs), outer_states, frame2d)
        key_vector = key_values[:, 0]
        matches = occupied & (keys == key_vector[:, None])
        has_match = jnp.any(matches, axis=1)
        first_free = jnp.argmax(~occupied, axis=1)
        slot = jnp.where(has_match, jnp.argmax(matches, axis=1), first_free)
        row_idx = jnp.arange(frame2d.shape[1])
        new_keys = keys.at[row_idx, slot].set(key_vector)
        new_occupied = occupied.at[row_idx, slot].set(True)

        flat_slot = row_idx * self.capacity + slot
        local_cols = frame2d.shape[1] * self.capacity
        if lhs_values.ndim == 1:
            local_frame = jnp.full((local_cols, 1), jnp.nan)
            local_frame = local_frame.at[flat_slot, 0].set(lhs_values)
        else:
            local_frame = jnp.full((local_cols, lhs_values.shape[1]), jnp.nan)
            local_frame = local_frame.at[flat_slot, :].set(lhs_values)
        new_child_state, grouped_out = self.child.tick(child_state, local_frame)
        out = grouped_out[flat_slot, :]
        return (new_outer_states, new_child_state, new_keys, new_occupied), out



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
        beta_new = jnp.linalg.pinv(system) @ xy
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
    "abs": AbsOp,
    "ln": LnOp,
    "ceil": CeilOp,
    "floor": FloorOp,
    "exp": ExpOp,
    "sign": SignOp,
    "arctan": ArctanOp,
    "isnan": IsNanOp,
    "purify": PurifyOp,
    "fraction": FractionOp,
}

_BINARY_OPS = {
    "add": AddOp,
    "sub": SubOp,
    "mul": MulOp,
    "mod": ModOp,
    "pow": PowOp,
    "div": DivOp,
    "floordiv": FloorDivOp,
    "eq": EqOp,
    "ne": NeOp,
    "lt": LtOp,
    "gt": GtOp,
    "and": AndOp,
    "and_": AndOp,
    "or": OrOp,
    "or_": OrOp,
    "xor": XorOp,
    "fillna": FillNaOp,
}


def _expr_key(node: Expr):
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", float(node.value))
    if isinstance(node, Universe):
        return ("univ", node.groups)
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(a) for a in node.args))
    raise FormulaCompileError(f"Unsupported expression node: {node}")


def _resolve_universe_groups(universe: Universe, column_names):
    name_to_idx = {name: i for i, name in enumerate(column_names or ())}
    groups = []
    seen = set()
    for group in universe.groups:
        resolved = []
        for item in group:
            if isinstance(item, int):
                idx = item
            else:
                if item not in name_to_idx:
                    raise FormulaCompileError(
                        f"Unknown universe column {item!r}. Pass column_names to compile_formula/build_engine."
                    )
                idx = name_to_idx[item]
            if idx in seen:
                raise FormulaCompileError(f"Universe column index {idx} appears in more than one group")
            seen.add(idx)
            resolved.append(int(idx))
        groups.append(tuple(resolved))
    return tuple(groups)


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

def _make_call_op(fn: str, args: tuple[Expr, ...], children: tuple[Any, ...]):
    if fn in _UNARY_OPS and len(children) == 1:
        return _UNARY_OPS[fn](children[0], children[0].output_kind)
    if fn in _BINARY_OPS and len(children) == 2:
        return _BINARY_OPS[fn](children[0], children[1], _combine_kind(tuple(child.output_kind for child in children)))
    if fn == "where" and len(children) == 3:
        return WhereOp(children[0], children[1], children[2], _combine_kind(tuple(child.output_kind for child in children)))
    if fn == "mean" and len(children) == 1:
        return MeanOp(children[0], "scalar")
    if fn == "outer" and len(children) == 1:
        return OuterOp(children[0], "matrix")
    if fn == "col" and len(children) == 2:
        return ColOp(children[0], int(_literal_arg(args[1], fn, 2)))
    if fn == "bspline" and len(children) == 2:
        return BsplineOp(children[0], int(_literal_arg(args[1], fn, 2)))
    if fn == "xs_rank" and len(children) == 1:
        return XsRankOp(children[0], "vector")
    if fn == "ewm" and len(children) == 2:
        return EwmOp(children[0], children[1], children[0].output_kind)
    if fn == "cumsum" and len(children) == 1:
        return CumsumOp(children[0], children[0].output_kind)
    if fn == "shift" and len(children) in (2, 3):
        max_size_arg = args[2] if len(args) > 2 else args[1]
        max_size = int(_literal_arg(max_size_arg, fn, 3 if len(args) > 2 else 2))
        max_size_source = children[2] if len(children) > 2 else children[1]
        return ShiftOp(children[0], children[1], max_size_source, max(1, max_size), children[0].output_kind)
    if fn == "rolling_quantile" and len(children) == 3:
        window = int(_literal_arg(args[1], fn, 2))
        return RollingQuantileOp(children[0], children[1], children[2], max(1, window), children[0].output_kind)
    if fn == "Ridge" and len(children) >= 4:
        has_weights = len(children) >= 5
        features = children[:-4] if has_weights else children[:-3]
        y = children[-4] if has_weights else children[-3]
        weights = children[-3] if has_weights else LiteralOp(1.0)
        half_life = children[-2]
        ridge_lambda = children[-1]
        return RidgeOp(features, y, weights, half_life, ridge_lambda, tuple(_feature_width(feature) for feature in features))
    if fn == "get_beta" and len(children) == 1:
        return GetBetaOp(children[0], "vector")
    if fn == "get_preds" and len(children) == 1:
        return GetPredsOp(children[0], "vector")
    raise FormulaCompileError(f"JAX backend does not support op '{fn}' yet")


def compile_formula(
    formula: str | Expr,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
) -> JaxCompiledArtifact:
    started_at = perf_counter()
    ast_expr = parse_formula(formula) if isinstance(formula, str) else formula
    dsl_registry = dsl_registry or DEFAULT_DSL_REGISTRY
    inputs: dict[str, int] = {}
    cache: dict[tuple, Any] = {}
    cache_hits = 0
    expanded_nodes = 0

    def build(expr: Expr, local_inputs: dict[str, Any] | None = None) -> Any:
        nonlocal cache_hits, expanded_nodes
        use_cache = local_inputs is None
        key = _expr_key(expr)
        if use_cache and key in cache:
            cache_hits += 1
            return cache[key]
        expanded_nodes += 1
        if isinstance(expr, Identifier):
            if local_inputs is not None:
                if expr.name not in local_inputs:
                    raise FormulaCompileError("groupby local op expressions may only reference the 'self_' lhs placeholder")
                op = local_inputs[expr.name]
            else:
                inputs.setdefault(expr.name, len(inputs))
                op = InputOp(inputs[expr.name])
        elif isinstance(expr, Number):
            op = LiteralOp(float(expr.value))
        elif isinstance(expr, Call):
            macro = dsl_registry.get(expr.fn)
            if macro is not None:
                op = build(macro(*expr.args), local_inputs)
            elif expr.fn == "groupby" and len(expr.args) == 2 and isinstance(expr.args[0], Universe):
                child = build(expr.args[1], local_inputs)
                op = UniverseGroupByOp(child, _resolve_universe_groups(expr.args[0], column_names))
            elif expr.fn == "groupby" and len(expr.args) == 2:
                key_child = build(expr.args[0], local_inputs)
                op_child = build(expr.args[1], local_inputs)
                op = GroupByOp(key_child, op_child, len(inputs), output_kind=op_child.output_kind)
            elif expr.fn == "groupby" and len(expr.args) == 3:
                key_child = build(expr.args[0], local_inputs)
                lhs_child = build(expr.args[1], local_inputs)
                local_value = LocalValueOp(lhs_child.output_kind, lhs_child.output_kind)
                rhs_child = build(expr.args[2], {"self_": local_value})
                op = ScopedGroupByOp(key_child, lhs_child, rhs_child, output_kind=rhs_child.output_kind)
            else:
                children = tuple(build(arg, local_inputs) for arg in expr.args)
                op = _make_call_op(expr.fn, expr.args, children)
        else:
            raise FormulaCompileError(f"Unsupported expression node: {expr}")
        if use_cache:
            cache[key] = op
        return op

    root = build(ast_expr)
    return JaxCompiledArtifact(
        compiled=JaxProgram(root, len(inputs), root.output_kind),
        input_names=tuple(inputs.keys()),
        output_kind=root.output_kind,
        stats=CompileStats(
            expanded_nodes=expanded_nodes,
            cache_hits=cache_hits,
            compile_seconds=perf_counter() - started_at,
        ),
    )


def build_jax_engine(
    formula: str | Expr,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
):
    artifact = compile_formula(formula, dsl_registry=dsl_registry, column_names=column_names)
    return JaxEngineHandle(artifact.compiled, artifact.input_names, artifact.output_kind)


build_engine = build_jax_engine


def _as_aligned_inputs(engine: JaxEngineHandle, data: dict[str, np.ndarray]):
    arrays = []
    for name in engine.input_names:
        arr = np.asarray(data[name], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input for '{name}', got shape {arr.shape}")
        arrays.append(jnp.asarray(arr))
    return tuple(arrays)


def run_batch_from_mapping(
    engine: JaxEngineHandle,
    data: dict[str, np.ndarray],
    out=None,
    out_path: str | None = f"{tempfile.gettempdir()}/trading_dsl_engine_jax_out.memmap",
    chunk_size: int = 8192,
):
    inputs = _as_aligned_inputs(engine, data)
    result = np.asarray(_jit_batch(engine.compiled, inputs))
    if out is not None:
        out[...] = result
        return out
    if out_path is not None:
        mapped = np.memmap(out_path, mode="w+", dtype=np.float64, shape=result.shape)
        mapped[...] = result
        return mapped
    return result


def update_from_mapping(engine: JaxEngineHandle, data: dict[str, np.ndarray]):
    frame = np.empty(
        (len(engine.input_names), np.asarray(data[engine.input_names[0]]).shape[0]),
        dtype=np.float64,
    )
    for i, name in enumerate(engine.input_names):
        frame[i, :] = np.asarray(data[name], dtype=np.float64)
    engine.on_data(frame)
    return engine.emit()
