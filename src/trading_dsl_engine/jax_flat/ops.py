from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
from jax.experimental import io_callback
import jax.numpy as jnp
import numpy as np
import jax.scipy.special as jsp_special

from trading_dsl_engine.jax_ffi.nnqp import nnqp

jax.config.update("jax_enable_x64", True)


class Op:
    output_kind: str = "vector"
    output_width: int | None = 1
    is_stateful: bool = False

    def init_state(self, sample: jax.Array):
        return None

    def tick(self, state: Any, *child_values: jax.Array):
        del state, child_values
        raise NotImplementedError

    def scan_batch(self, state: Any, *child_sequences: jax.Array):
        def step(carry, values):
            return self.tick(carry, *values)

        return jax.lax.scan(step, state, xs=child_sequences, unroll=32)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EwmState:
    value: jax.Array
    weight: jax.Array
    initialized: jax.Array
    count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RollingMeanState:
    buffer: jax.Array
    pos: jax.Array
    count: jax.Array
    total: jax.Array
    valid_count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RollingState:
    buffer: jax.Array
    pos: jax.Array
    count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CumsumState:
    value: jax.Array
    initialized: jax.Array



@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class ShiftState:
    buffer: jax.Array
    pos: jax.Array
    count: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FFillState:
    last: jax.Array
    streak: jax.Array
    seen: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RidgeState:
    xx: jax.Array
    xy: jax.Array
    has_xx: jax.Array
    has_xy: jax.Array
    last_xx: jax.Array
    last_xy: jax.Array
    beta: jax.Array
    preds: jax.Array
    t: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InstrumentBasisMeanState:
    num: jax.Array
    den: jax.Array
    has_value: jax.Array
    beta: jax.Array
    preds: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class RidgeValue:
    beta: jax.Array
    preds: jax.Array


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class InstrumentBasisMeanValue:
    beta: jax.Array
    preds: jax.Array


@dataclass(frozen=True)
class InputOp(Op):
    input_index: int
    output_kind: str = "vector"
    output_width: int | None = 1


@dataclass(frozen=True)
class LiteralOp(Op):
    value: float
    output_kind: str = "scalar"
    output_width: int | None = 1


@dataclass(frozen=True)
class CacheOp(Op):
    storage: str = "ram"
    output_kind: str = "vector"
    output_width: int | None = 1
    cpp_name: str | None = "cache"
    cache_write_target: Any = None

    def tick(self, state: Any, *child_values: jax.Array):
        del state
        return None, child_values[0]

    def scan_batch(self, state: Any, *child_sequences: jax.Array):
        return self.scan_batch_with_start(state, jnp.asarray(0, dtype=jnp.int64), *child_sequences)

    def scan_batch_with_start(self, state: Any, batch_start: jax.Array, *child_sequences: jax.Array):
        del state
        value = child_sequences[0]
        if self.cache_write_target is not None:
            io_callback(
                lambda start, chunk, target=self.cache_write_target: target.write(start, chunk),
                None,
                batch_start,
                value,
                ordered=False,
            )
        return None, value


@dataclass(frozen=True)
class NaryOp(Op):
    fn: Callable[..., jax.Array]
    output_kind: str = "vector"
    output_width: int | None = 1
    cpp_name: str | None = None
    cpp_param: float = 0.0
    cpp_int_param: int = 0
    cpp_str_param: str = ""
    diagnostic_name: str | None = None

    def tick(self, state: Any, *child_values: jax.Array):
        del state
        return None, self.fn(*child_values)

    def scan_batch(self, state: Any, *child_sequences: jax.Array):
        del state
        return None, jax.vmap(self.fn)(*child_sequences)


@dataclass(frozen=True)
class EwmOp(Op):
    span: float | None = None
    min_periods: float | None = None
    ignore_na: bool = False
    adjust: bool = True
    output_kind: str = "vector"
    output_width: int | None = 1
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        return EwmState(
            value=jnp.zeros_like(sample),
            weight=jnp.zeros_like(sample),
            initialized=jnp.zeros_like(sample, dtype=bool),
            count=jnp.zeros_like(sample, dtype=jnp.int64),
        )

    def tick(self, state: EwmState, *child_values: jax.Array):
        x = child_values[0]
        span = self.span if self.span is not None else _scalar_value(child_values[1])
        value, weight, initialized = state.value, state.weight, state.initialized
        alpha = 2.0 / (span + 1.0)
        old_wt_factor = 1.0 - alpha
        valid = jnp.isfinite(x)
        if self.adjust:
            decay = valid | (not self.ignore_na)
            decayed_weight = jnp.where(initialized & decay, weight * old_wt_factor, weight)
            new_wt = 1.0
            weighted = (decayed_weight * value + new_wt * x) / (decayed_weight + new_wt)
            next_weight_if_valid = decayed_weight + new_wt
        else:
            decay = valid | (not self.ignore_na)
            decayed_weight = jnp.where(initialized & decay, weight * old_wt_factor, weight)
            normalized = (decayed_weight * value + alpha * x) / (decayed_weight + alpha)
            alpha_half = jnp.isclose(alpha, 0.5)
            half_alpha_weighted = decayed_weight * value + (1.0 - decayed_weight) * x
            weighted = jnp.where(alpha_half, half_alpha_weighted, normalized)
            next_weight_if_valid = jnp.ones_like(decayed_weight)
        next_value = jnp.where(valid, jnp.where(initialized, weighted, x), value)
        next_weight = jnp.where(valid, next_weight_if_valid, decayed_weight)
        next_initialized = initialized | valid
        next_count = state.count + valid.astype(jnp.int64)
        min_periods = self.min_periods if self.min_periods is not None else None
        if min_periods is None and len(child_values) > 2:
            min_periods = _scalar_value(child_values[2])
        enough = True if min_periods is None else next_count >= jnp.rint(min_periods).astype(jnp.int64)
        out = jnp.where(next_initialized & enough, next_value, jnp.nan)
        return EwmState(value=next_value, weight=next_weight, initialized=next_initialized, count=next_count), out


@dataclass(frozen=True)
class RollingMeanOp(Op):
    lookback: int
    min_periods: int
    output_kind: str = "vector"
    output_width: int | None = 1
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        sample = jnp.asarray(sample)
        return RollingMeanState(
            buffer=jnp.full((self.lookback,) + sample.shape, jnp.nan, dtype=jnp.float64),
            pos=jnp.asarray(0, dtype=jnp.int32),
            count=jnp.asarray(0, dtype=jnp.int32),
            total=jnp.zeros_like(sample, dtype=jnp.float64),
            valid_count=jnp.zeros_like(sample, dtype=jnp.int64),
        )

    def tick(self, state: RollingMeanState, *child_values: jax.Array):
        x = jnp.asarray(child_values[0])
        old = state.buffer[state.pos]
        old_valid = jnp.isfinite(old)
        valid = jnp.isfinite(x)
        total = state.total - jnp.where(old_valid, old, 0.0) + jnp.where(valid, x, 0.0)
        valid_count = state.valid_count - old_valid.astype(jnp.int64) + valid.astype(jnp.int64)
        count = jnp.minimum(state.count + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(self.lookback, dtype=jnp.int32))
        out = total / jnp.where(valid_count > 0, valid_count, jnp.nan)
        out = jnp.where((count >= self.min_periods) & (valid_count >= self.min_periods), out, jnp.nan)
        next_state = RollingMeanState(
            buffer=state.buffer.at[state.pos].set(x),
            pos=jnp.mod(state.pos + 1, self.lookback),
            count=count,
            total=total,
            valid_count=valid_count,
        )
        return next_state, out

    def scan_batch(self, state: RollingMeanState, *child_sequences: jax.Array):
        def step(carry, x):
            state_c = RollingMeanState(*carry)
            next_state, out = self.tick(state_c, x)
            return (next_state.buffer, next_state.pos, next_state.count, next_state.total, next_state.valid_count), out

        carry, out = jax.lax.scan(
            step,
            (state.buffer, state.pos, state.count, state.total, state.valid_count),
            child_sequences[0],
            unroll=32,
        )
        return RollingMeanState(*carry), out


@dataclass(frozen=True)
class RollingOp(Op):
    lookback: int
    min_periods: int
    fn: Callable[[jax.Array], jax.Array]
    output_kind: str = "vector"
    output_width: int | None = 1
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        sample = jnp.asarray(sample)
        return RollingState(
            buffer=jnp.full((self.lookback,) + sample.shape, jnp.nan, dtype=jnp.float64),
            pos=jnp.asarray(0, dtype=jnp.int32),
            count=jnp.asarray(0, dtype=jnp.int32),
        )

    def tick(self, state: RollingState, *child_values: jax.Array):
        x = jnp.asarray(child_values[0])
        buffer = state.buffer.at[state.pos].set(x)
        count = jnp.minimum(state.count + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(self.lookback, dtype=jnp.int32))
        chronological = buffer[jnp.mod(state.pos + 1 + jnp.arange(self.lookback, dtype=jnp.int32), self.lookback)]
        valid_count = jnp.sum(jnp.isfinite(chronological), axis=0)
        enough = (count >= self.min_periods) & (valid_count >= self.min_periods)
        out = jnp.where(enough, self.fn(chronological), jnp.nan)
        return RollingState(buffer=buffer, pos=jnp.mod(state.pos + 1, self.lookback), count=count), out

    def scan_batch(self, state: RollingState, *child_sequences: jax.Array):
        def step(carry, x):
            state_c = RollingState(*carry)
            next_state, out = self.tick(state_c, x)
            return (next_state.buffer, next_state.pos, next_state.count), out

        carry, out = jax.lax.scan(step, (state.buffer, state.pos, state.count), child_sequences[0], unroll=32)
        return RollingState(*carry), out


@dataclass(frozen=True)
class ShiftOp(Op):
    max_size: int
    default_lag: float = 1.0
    output_kind: str = "vector"
    output_width: int | None = 1
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        shape = (self.max_size + 1,) + jnp.asarray(sample).shape

        return ShiftState(
            buffer=jnp.full(shape, jnp.nan, dtype=jnp.float64),
            pos=jnp.asarray(0, dtype=jnp.int32),
            count=jnp.asarray(0, dtype=jnp.int32),
        )

    def tick(self, state: ShiftState, *child_values: jax.Array):
        x = child_values[0]
        lag = child_values[1] if len(child_values) > 1 else jnp.asarray(self.default_lag)
        cap = state.buffer.shape[0]
        lag_values = _lag_vector(lag, x.shape[0])
        finite_lag = jnp.isfinite(lag_values)
        lag_i = jnp.clip(jnp.rint(jnp.where(finite_lag, lag_values, 0.0)).astype(jnp.int32), 0, cap - 1)
        read_pos = jnp.mod(state.pos - lag_i, cap)
        shifted = state.buffer[read_pos, jnp.arange(x.shape[0], dtype=jnp.int32)]
        shifted = jnp.where(lag_i == 0, x, shifted)
        shifted = jnp.where(finite_lag & (state.count >= lag_i), shifted, jnp.nan)
        next_buffer = state.buffer.at[state.pos].set(x)
        return (
            ShiftState(
                buffer=next_buffer,
                pos=jnp.mod(state.pos + 1, cap),
                count=jnp.minimum(state.count + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(cap, dtype=jnp.int32)),
            ),
            shifted,
        )

    def scan_batch(self, state: ShiftState, *child_sequences: jax.Array):
        x = child_sequences[0]
        lag = child_sequences[1] if len(child_sequences) > 1 else jnp.asarray(self.default_lag)
        cap = state.buffer.shape[0]
        rows, cols = x.shape[:2]
        history = jnp.concatenate((_chronological_buffer(state), x), axis=0)
        lag_values = _lag_matrix(lag, rows, cols)
        finite_lag = jnp.isfinite(lag_values)
        lag_i = jnp.clip(jnp.rint(jnp.where(finite_lag, lag_values, 0.0)).astype(jnp.int32), 0, cap - 1)
        time_idx = jnp.arange(rows, dtype=jnp.int32)[:, None]
        col_idx = jnp.arange(cols, dtype=jnp.int32)[None, :]
        shifted = history[cap + time_idx - lag_i, col_idx]
        shifted = jnp.where(lag_i == 0, x, shifted)
        available = state.count + time_idx >= lag_i
        shifted = jnp.where(finite_lag & available, shifted, jnp.nan)
        return _shift_next_state(state, x), shifted


@dataclass(frozen=True)
class BufferShiftOp(Op):
    max_size: int
    max_lag: int
    output_kind: str = "matrix"
    output_width: int | None = None
    is_stateful: bool = True

    def __post_init__(self):
        object.__setattr__(self, "output_width", self.max_lag)

    def init_state(self, sample: jax.Array):
        shape = (self.max_size + 1,) + jnp.asarray(sample).shape
        return ShiftState(
            buffer=jnp.full(shape, jnp.nan, dtype=jnp.float64),
            pos=jnp.asarray(0, dtype=jnp.int32),
            count=jnp.asarray(0, dtype=jnp.int32),
        )

    def tick(self, state: ShiftState, *child_values: jax.Array):
        x, upper_lag, min_lag = child_values[:3]
        cap = state.buffer.shape[0]
        n = x.shape[0]
        lag_cols = jnp.arange(1, self.max_lag + 1, dtype=jnp.int32)
        read_pos = jnp.mod(state.pos - lag_cols, cap)
        history = jnp.moveaxis(jnp.swapaxes(state.buffer[read_pos], 0, 1), 1, -1)

        min_values = _lag_vector(min_lag, n)
        upper_values = _lag_vector(upper_lag, n)
        finite_bounds = jnp.isfinite(min_values) & jnp.isfinite(upper_values)
        min_bound = jnp.rint(jnp.where(jnp.isfinite(min_values), min_values, 0.0))
        upper_bound = jnp.rint(jnp.where(jnp.isfinite(upper_values), upper_values, -1.0))
        lag_cols_f = lag_cols.astype(jnp.float64)
        available = state.count >= lag_cols
        in_window = (lag_cols_f[None, :] >= min_bound[:, None]) & (lag_cols_f[None, :] <= upper_bound[:, None])
        valid = finite_bounds[:, None] & available[None, :] & in_window
        valid = self._tick_lag_mask(valid, x.ndim)
        out = jnp.where(valid, history, jnp.nan)

        next_buffer = state.buffer.at[state.pos].set(x)
        return (
            ShiftState(
                buffer=next_buffer,
                pos=jnp.mod(state.pos + 1, cap),
                count=jnp.minimum(state.count + jnp.asarray(1, dtype=jnp.int32), jnp.asarray(cap, dtype=jnp.int32)),
            ),
            out,
        )

    def scan_batch(self, state: ShiftState, *child_sequences: jax.Array):
        x, upper_lag, min_lag = child_sequences[:3]
        cap = state.buffer.shape[0]
        rows, cols = x.shape[:2]
        lag_cols = jnp.arange(1, self.max_lag + 1, dtype=jnp.int32)
        history = jnp.concatenate((_chronological_buffer(state)[-self.max_lag :], x), axis=0)
        if self.max_lag <= 32:
            out = jnp.stack(
                tuple(history[self.max_lag - lag : self.max_lag - lag + rows] for lag in range(1, self.max_lag + 1)),
                axis=-1,
            )
        else:
            time_idx = jnp.arange(rows, dtype=jnp.int32)[:, None, None]
            col_idx = jnp.arange(cols, dtype=jnp.int32)[None, :, None]
            lag_idx = lag_cols[None, None, :]
            out = jnp.moveaxis(history[self.max_lag + time_idx - lag_idx, col_idx], 2, -1)

        min_values = _lag_matrix(min_lag, rows, cols)
        upper_values = _lag_matrix(upper_lag, rows, cols)
        min_finite = jnp.isfinite(min_values)
        upper_finite = jnp.isfinite(upper_values)
        min_bound = jnp.rint(jnp.where(min_finite, min_values, 0.0))
        upper_bound = jnp.rint(jnp.where(upper_finite, upper_values, -1.0))
        lag_cols_f = lag_cols.astype(jnp.float64)
        available = state.count + jnp.arange(rows, dtype=jnp.int32)[:, None] >= lag_cols[None, :]
        in_window = (lag_cols_f[None, None, :] >= min_bound[:, :, None]) & (lag_cols_f[None, None, :] <= upper_bound[:, :, None])
        valid = (min_finite & upper_finite)[:, :, None] & available[:, None, :] & in_window
        valid = self._batch_lag_mask(valid, x.ndim)
        return _shift_next_state(state, x), jnp.where(valid, out, jnp.nan)

    @staticmethod
    def _tick_lag_mask(mask: jax.Array, value_ndim: int):
        if value_ndim <= 1:
            return mask
        return jnp.reshape(mask, (mask.shape[0],) + (1,) * (value_ndim - 1) + (mask.shape[1],))

    @staticmethod
    def _batch_lag_mask(mask: jax.Array, value_ndim: int):
        if value_ndim <= 2:
            return mask
        return jnp.reshape(mask, mask.shape[:2] + (1,) * (value_ndim - 2) + (mask.shape[2],))


def _chronological_buffer(state: ShiftState):
    cap = state.buffer.shape[0]
    return state.buffer[jnp.mod(state.pos + jnp.arange(cap, dtype=jnp.int32), cap)]


def _shift_next_state(state: ShiftState, x_seq: jax.Array):
    cap = state.buffer.shape[0]
    if x_seq.shape[0] >= cap:
        buffer = x_seq[-cap:]
    else:
        history = jnp.concatenate((_chronological_buffer(state), x_seq), axis=0)
        buffer = history[-cap:]
    return ShiftState(
        buffer=buffer,
        pos=jnp.asarray(0, dtype=jnp.int32),
        count=jnp.minimum(state.count + jnp.asarray(x_seq.shape[0], dtype=jnp.int32), jnp.asarray(cap, dtype=jnp.int32)),
    )


def _lag_matrix(lag, rows: int, cols: int):
    lag = jnp.asarray(lag)
    if lag.ndim == 0:
        return jnp.broadcast_to(lag, (rows, cols))
    if lag.ndim == 1:
        if lag.shape[0] == rows:
            return jnp.broadcast_to(lag[:, None], (rows, cols))
        return jnp.broadcast_to(lag[None, :], (rows, cols))
    if lag.shape[1] == 1:
        return jnp.broadcast_to(lag, (rows, cols))
    return lag


@dataclass(frozen=True)
class CumsumOp(Op):
    output_kind: str = "vector"
    output_width: int | None = 1
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        return CumsumState(value=jnp.zeros_like(sample), initialized=jnp.zeros_like(sample, dtype=bool))

    def tick(self, state: CumsumState, *child_values: jax.Array):
        x = child_values[0]
        valid = jnp.isfinite(x)
        initialized = state.initialized | valid
        value = state.value + jnp.where(valid, x, 0.0)
        out = jnp.where(valid, value, jnp.nan)
        return CumsumState(value=value, initialized=initialized), out

    def scan_batch(self, state: CumsumState, *child_sequences: jax.Array):
        x = _broadcast_sequence_to_state(child_sequences[0], state.value)
        valid = jnp.isfinite(x)
        cumulative = state.value + jnp.cumsum(jnp.where(valid, x, 0.0), axis=0)
        out = jnp.where(valid, cumulative, jnp.nan)
        next_value = cumulative[-1]
        next_initialized = state.initialized | jnp.any(valid, axis=0)
        return CumsumState(value=next_value, initialized=next_initialized), out



@dataclass(frozen=True)
class FFillOp(Op):
    limit: int | None = None
    dynamic_limit: bool = False
    output_kind: str = "vector"
    output_width: int | None = 1
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        sample = jnp.asarray(sample)
        return FFillState(
            last=jnp.full_like(sample, jnp.nan, dtype=jnp.float64),
            streak=jnp.zeros_like(sample, dtype=jnp.int64),
            seen=jnp.zeros_like(sample, dtype=bool),
        )

    def tick(self, state: FFillState, *child_values: jax.Array):
        x = child_values[0]
        if self.dynamic_limit:
            return self._dynamic_step(state, x, child_values[1])
        if self.limit is None:
            return self._unlimited_step(state, x)
        return self._limited_step(state, x, self.limit)

    def scan_batch(self, state: FFillState, *child_sequences: jax.Array):
        x = _broadcast_sequence_to_state(child_sequences[0], state.last)
        if not self.dynamic_limit:
            return self._static_scan(state, x, self.limit)

        limit = child_sequences[1]

        def step(carry, values):
            return self._dynamic_step(carry, *values)

        return jax.lax.scan(step, state, (x, limit), unroll=32)

    @staticmethod
    def _unlimited_step(state: FFillState, x: jax.Array):
        x = jnp.asarray(x)
        valid = jnp.isfinite(x)
        seen = state.seen | valid
        last = jnp.where(valid, x, state.last)
        return FFillState(last=last, streak=state.streak, seen=seen), jnp.where(seen, last, jnp.nan)

    @staticmethod
    def _limited_step(state: FFillState, x: jax.Array, limit: int):
        x = jnp.asarray(x)
        valid = jnp.isfinite(x)
        seen = state.seen | valid
        can_fill = (~valid) & seen & (state.streak < limit)
        last = jnp.where(valid, x, state.last)
        streak = jnp.where(valid, 0, jnp.where(can_fill, state.streak + 1, state.streak))
        out = jnp.where(valid, x, jnp.where(can_fill, state.last, jnp.nan))
        return FFillState(last=last, streak=streak, seen=seen), out

    @classmethod
    def _dynamic_step(cls, state: FFillState, x: jax.Array, limit: jax.Array):
        limit_value = _scalar_value(limit)
        limit_i = jnp.rint(limit_value).astype(jnp.int64)
        active = jnp.isfinite(limit_value) & (limit_i >= 0)
        next_state, out = cls._limited_step(state, x, limit_i)
        return (
            FFillState(
                last=jnp.where(active, next_state.last, state.last),
                streak=jnp.where(active, next_state.streak, state.streak),
                seen=jnp.where(active, next_state.seen, state.seen),
            ),
            jnp.where(active, out, jnp.nan),
        )

    @staticmethod
    def _static_scan(state: FFillState, x: jax.Array, limit: int | None):
        valid_x = jnp.isfinite(x)
        values = jnp.concatenate((state.last[None], x), axis=0)
        valid = jnp.concatenate((state.seen[None], valid_x), axis=0)
        time_shape = (values.shape[0],) + (1,) * (values.ndim - 1)
        time = jnp.arange(values.shape[0], dtype=jnp.int64).reshape(time_shape)
        last_idx = jnp.maximum.accumulate(jnp.where(valid, time, 0), axis=0)
        seen = jnp.maximum.accumulate(valid, axis=0)
        filled = jnp.take_along_axis(values, last_idx, axis=0)

        if limit is None:
            out = jnp.where(seen[1:], filled[1:], jnp.nan)
            return FFillState(last=filled[-1], streak=state.streak, seen=seen[-1]), out

        distance = jnp.where(last_idx == 0, state.streak[None] + time, time - last_idx)
        allowed = seen & (distance <= limit)
        out = jnp.where(allowed[1:], filled[1:], jnp.nan)
        next_streak = jnp.where(valid[-1], 0, jnp.minimum(distance[-1], jnp.asarray(limit, dtype=jnp.int64)))
        return FFillState(last=filled[-1], streak=next_streak, seen=seen[-1]), out


@dataclass(frozen=True)
class InstrumentBasisMeanOp(Op):
    feature_width: int
    has_weights: bool
    output_kind: str = "object"
    output_width: int | None = None
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        n = jnp.asarray(sample).shape[0]
        shape = (n, self.feature_width)
        return InstrumentBasisMeanState(
            num=jnp.zeros(shape, dtype=jnp.float64),
            den=jnp.zeros(shape, dtype=jnp.float64),
            has_value=jnp.zeros(shape, dtype=bool),
            beta=jnp.zeros(shape, dtype=jnp.float64),
            preds=jnp.full((n,), jnp.nan, dtype=jnp.float64),
        )

    def tick(self, state: InstrumentBasisMeanState, *child_values: jax.Array):
        if self.has_weights:
            features, y, weights, hl = child_values[:4]
        else:
            features, y, hl = child_values[:3]
            weights = jnp.asarray(1.0, dtype=jnp.float64)
        xmat = RidgeOp._as_feature_matrix(features)
        y = jnp.asarray(y)
        y_vec = y[:, 0] if y.ndim == 2 else y
        weights = jnp.asarray(weights)
        if weights.ndim == 0:
            w = jnp.full((xmat.shape[0],), weights)
        elif weights.ndim == 2 and weights.shape[1] == 1:
            w = weights[:, 0]
        else:
            w = weights
        valid_row = jnp.isfinite(y_vec) & jnp.isfinite(w)
        valid = jnp.isfinite(xmat) & valid_row[:, None]
        x0 = jnp.where(valid, xmat, 0.0)
        yw = jnp.where(valid_row, y_vec * w, 0.0)
        w0 = jnp.where(jnp.isfinite(w), w, 0.0)
        num_new = x0 * yw[:, None]
        den_new = x0 * w0[:, None]
        hl_value = _scalar_value(hl)
        rho = jnp.where((hl_value <= 0.0) | jnp.isnan(hl_value), 0.0, jnp.exp(jnp.log(0.5) / hl_value))
        alpha = jnp.clip(1.0 - rho, 0.0, 1.0)
        num_update = jnp.where(state.has_value, state.num * (1.0 - alpha) + num_new * alpha, num_new)
        den_update = jnp.where(state.has_value, state.den * (1.0 - alpha) + den_new * alpha, den_new)
        num = jnp.where(valid, num_update, state.num)
        den = jnp.where(valid, den_update, state.den)
        has_value = state.has_value | valid
        beta_candidate = num / jnp.where(den != 0.0, den, jnp.nan)
        beta = jnp.where(jnp.isfinite(beta_candidate), beta_candidate, state.beta)
        preds = jnp.where(valid_row & jnp.all(jnp.isfinite(xmat), axis=1), jnp.sum(xmat * state.beta, axis=1), jnp.nan)
        return InstrumentBasisMeanState(num=num, den=den, has_value=has_value, beta=beta, preds=preds), InstrumentBasisMeanValue(beta=beta, preds=preds)

    def scan_batch(self, state: InstrumentBasisMeanState, *child_sequences: jax.Array):
        def step(carry, values):
            state_c = InstrumentBasisMeanState(*carry)
            next_state, out = self.tick(state_c, *values)
            return (next_state.num, next_state.den, next_state.has_value, next_state.beta, next_state.preds), out

        carry, out = jax.lax.scan(
            step,
            (state.num, state.den, state.has_value, state.beta, state.preds),
            child_sequences,
            unroll=32,
        )
        return InstrumentBasisMeanState(*carry), out


@dataclass(frozen=True)
class RbfBasisOp(Op):
    n_basis: int
    output_kind: str = "matrix"
    output_width: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "output_width", self.n_basis)

    @staticmethod
    def _as_vector(value, rows: int | None = None):
        value = jnp.asarray(value)
        if value.ndim == 0:
            if rows is None:
                return value[None]
            return jnp.broadcast_to(value, (rows,))
        if value.ndim == 1:
            return value
        return value[:, 0]

    @staticmethod
    def _session_vectors(*values):
        arrays = tuple(jnp.asarray(value) for value in values)
        rows = next((array.shape[0] for array in arrays if array.ndim > 0), 1)
        return tuple(RbfBasisOp._as_vector(array, rows) for array in arrays)

    @staticmethod
    def _normalized_basis(x, n_basis: int):
        x = jnp.asarray(x)
        clipped = jnp.clip(x, 0.0, 1.0)
        centers = jnp.linspace(0.0, 1.0, n_basis, dtype=jnp.float64)
        sigma = 1.0 / max(n_basis - 1, 1)
        dist = clipped[:, None] - centers[None, :]
        values = jnp.exp(-0.5 * (dist / sigma) ** 2)
        total = jnp.sum(values, axis=-1, keepdims=True)
        values = jnp.where(total <= 1e-18, 1.0 / n_basis, values / total)
        return jnp.where(jnp.isnan(x)[:, None], jnp.nan, values)

    @staticmethod
    def _session_phase(ev_ts, session_start, session_end):
        ev_ts, session_start, session_end = RbfBasisOp._session_vectors(ev_ts, session_start, session_end)
        session_len = session_end - session_start
        finite = jnp.isfinite(ev_ts) & jnp.isfinite(session_start) & jnp.isfinite(session_end)
        valid_session = finite & (session_len > 0.0)
        phase = (ev_ts - session_start) / jnp.where(session_len > 0.0, session_len, jnp.nan)
        in_session = valid_session & (ev_ts >= session_start) & (ev_ts < session_end)
        return phase, in_session, valid_session

    @staticmethod
    def _session_rbf_basis(ev_ts, session_start, session_end, n_basis: int):
        phase, in_session, _ = RbfBasisOp._session_phase(ev_ts, session_start, session_end)
        out = RbfBasisOp._normalized_basis(phase, n_basis)
        return jnp.where(in_session[:, None], out, jnp.nan)

    def tick(self, state: Any, *child_values: jax.Array):
        del state
        ev_ts, session_start, session_end = child_values[:3]
        return None, self._session_rbf_basis(ev_ts, session_start, session_end, self.n_basis)

    def scan_batch(self, state: Any, *child_sequences: jax.Array):
        del state
        return None, jax.vmap(lambda ev_ts, start, end: self._session_rbf_basis(ev_ts, start, end, self.n_basis))(
            *child_sequences[:3]
        )


@dataclass(frozen=True)
class FutureRbfBasisSumOp(Op):
    n_basis: int
    n_steps: int
    output_kind: str = "matrix"
    output_width: int | None = None

    def __post_init__(self):
        object.__setattr__(self, "output_width", self.n_basis)

    @staticmethod
    def _basis_suffix_table(n_basis: int, n_steps: int):
        grid = jnp.arange(n_steps, dtype=jnp.float64) / n_steps
        values = RbfBasisOp._normalized_basis(grid, n_basis)
        suffix = jnp.flip(jnp.cumsum(jnp.flip(values, axis=0), axis=0), axis=0)
        return jnp.concatenate((suffix, jnp.zeros((1, n_basis), dtype=values.dtype)), axis=0)

    @staticmethod
    def _future_rbf_basis_sum(ev_ts, session_start, session_end, n_basis: int, n_steps: int):
        ev_ts, session_start, session_end = RbfBasisOp._session_vectors(ev_ts, session_start, session_end)
        session_len = session_end - session_start
        finite = jnp.isfinite(ev_ts) & jnp.isfinite(session_start) & jnp.isfinite(session_end)
        valid_session = finite & (session_len > 0.0)
        phase = (ev_ts - session_start) / jnp.where(session_len > 0.0, session_len, jnp.nan)
        clipped = jnp.clip(phase, 0.0, 1.0)
        inside_idx = jnp.floor(clipped * n_steps).astype(jnp.int32) + 1
        idx = jnp.where(ev_ts < session_start, 0, jnp.where(ev_ts >= session_end, n_steps, inside_idx))
        idx = jnp.clip(idx, 0, n_steps)
        out = FutureRbfBasisSumOp._basis_suffix_table(n_basis, n_steps)[idx]
        return jnp.where(valid_session[:, None], out, jnp.nan)

    def tick(self, state: Any, *child_values: jax.Array):
        del state
        ev_ts, session_start, session_end = child_values[:3]
        return None, self._future_rbf_basis_sum(ev_ts, session_start, session_end, self.n_basis, self.n_steps)

    def scan_batch(self, state: Any, *child_sequences: jax.Array):
        del state
        return None, jax.vmap(
            lambda ev_ts, start, end: self._future_rbf_basis_sum(ev_ts, start, end, self.n_basis, self.n_steps)
        )(*child_sequences[:3])


@dataclass(frozen=True)
class RidgeOp(Op):
    feature_widths: tuple[int, ...]
    has_weights: bool
    nonneg: bool = False
    output_kind: str = "object"
    output_width: int | None = None
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        n = jnp.asarray(sample).shape[0]
        k = sum(self.feature_widths)
        return RidgeState(
            xx=jnp.zeros((k, k), dtype=jnp.float64),
            xy=jnp.zeros((k,), dtype=jnp.float64),
            has_xx=jnp.zeros((k, k), dtype=bool),
            has_xy=jnp.zeros((k,), dtype=bool),
            last_xx=jnp.zeros((k, k), dtype=jnp.int64),
            last_xy=jnp.zeros((k,), dtype=jnp.int64),
            beta=jnp.zeros((k,), dtype=jnp.float64),
            preds=jnp.full((n,), jnp.nan, dtype=jnp.float64),
            t=jnp.asarray(0, dtype=jnp.int64),
        )

    def tick(self, state: RidgeState, *child_values: jax.Array):

        if self.has_weights:
            feature_values = child_values[: len(self.feature_widths)]
            y, weights, hl, lam = child_values[-4:]
        else:
            feature_values = child_values[: len(self.feature_widths)]
            y, hl, lam = child_values[-3:]
            weights = jnp.asarray(1.0, dtype=jnp.float64)

        features = tuple(self._as_feature_matrix(value) for value in feature_values)
        xmat = jnp.concatenate(features, axis=1)
        y = jnp.asarray(y)
        y_vec = y[:, 0] if y.ndim == 2 else y
        row_valid = jnp.isfinite(y_vec) & jnp.all(jnp.isfinite(xmat), axis=1)
        if not self.is_stateful:
            # TODO: implement cpp equivalent for stateless
            return self._stateless_tick(xmat, y_vec, weights, lam, row_valid)

        prior_preds = jnp.where(row_valid, xmat @ state.beta, jnp.nan)
        xx_new, xy_new, xx_valid, xy_valid = self._moments(xmat, y_vec, weights)
        hl_value = _scalar_value(hl)
        lam_value = jnp.maximum(jnp.where(jnp.isnan(_scalar_value(lam)), 0.0, _scalar_value(lam)), 0.0)
        instant = (hl_value <= 0.0) | jnp.isnan(hl_value)
        rho = jnp.where(instant, 0.0, jnp.exp(jnp.log(0.5) / hl_value))
        alpha = jnp.clip(1.0 - rho, 0.0, 1.0)

        a_xx = alpha  # observation time: missing samples do not age the statistic
        a_xy = alpha
        updated_xx = jnp.where(state.has_xx, state.xx * (1.0 - a_xx) + xx_new * a_xx, xx_new)
        updated_xy = jnp.where(state.has_xy, state.xy * (1.0 - a_xy) + xy_new * a_xy, xy_new)
        xx = jnp.where(xx_valid, updated_xx, state.xx)
        xy = jnp.where(xy_valid, updated_xy, state.xy)
        xx = jnp.where(instant, jnp.where(xx_valid, xx_new, 0.0), xx)
        xy = jnp.where(instant, jnp.where(xy_valid, xy_new, 0.0), xy)
        has_xx = jnp.where(instant, xx_valid, state.has_xx | xx_valid)
        has_xy = jnp.where(instant, xy_valid, state.has_xy | xy_valid)
        last_xx = jnp.where(xx_valid, state.t, state.last_xx)
        last_xy = jnp.where(xy_valid, state.t, state.last_xy)

        xx = 0.5 * (xx + xx.T)
        last_xx = jnp.maximum(last_xx, last_xx.T)
        has_xx = has_xx | has_xx.T
        system = xx + lam_value * jnp.diag(jnp.diag(xx))
        beta_fallback = jnp.where(instant, jnp.zeros_like(state.beta), state.beta)
        beta = self._solve_system(system, xy, beta_fallback)
        preds = jnp.where(instant, jnp.where(row_valid, xmat @ beta, jnp.nan), prior_preds)
        next_state = RidgeState(
            xx=xx,
            xy=xy,
            has_xx=has_xx,
            has_xy=has_xy,
            last_xx=last_xx,
            last_xy=last_xy,
            beta=beta,
            preds=preds,
            t=state.t + 1,
        )
        return next_state, RidgeValue(beta=beta, preds=preds)

    def _stateless_tick(self, xmat, y_vec, weights, lam, row_valid):
        xx_new, xy_new, _, _ = self._moments(xmat, y_vec, weights)
        lam_value = jnp.maximum(jnp.where(jnp.isnan(_scalar_value(lam)), 0.0, _scalar_value(lam)), 0.0)

        xx = 0.5 * (xx_new + xx_new.T)
        system = xx + lam_value * jnp.diag(jnp.diag(xx))
        beta = self._solve_system(system, xy_new, jnp.zeros_like(xy_new))
        preds = jnp.where(row_valid, xmat @ beta, jnp.nan)
        return None, RidgeValue(beta=beta, preds=preds)

    def scan_batch(self, state: RidgeState, *child_sequences: jax.Array):

        if self.has_weights:
            feature_sequences = child_sequences[: len(self.feature_widths)]
            y_seq, weights_seq, hl_seq, lam_seq = child_sequences[-4:]
        else:
            feature_sequences = child_sequences[: len(self.feature_widths)]
            y_seq, hl_seq, lam_seq = child_sequences[-3:]
            weights_seq = jnp.broadcast_to(jnp.asarray(1.0, dtype=jnp.float64), (y_seq.shape[0],))

        def row_xmat(*feature_rows):
            features = tuple(self._as_feature_matrix(value) for value in feature_rows)
            return jnp.concatenate(features, axis=1)

        xmat_seq = jax.vmap(row_xmat)(*feature_sequences)
        y_arr = jnp.asarray(y_seq)
        y_vec_seq = y_arr[:, :, 0] if y_arr.ndim == 3 else y_arr
        row_valid_seq = jax.vmap(lambda y_vec, xmat: jnp.isfinite(y_vec) & jnp.all(jnp.isfinite(xmat), axis=1))(
            y_vec_seq, xmat_seq
        )
        if not self.is_stateful:
            # TODO: implement cpp equivalent for stateless
            return jax.vmap(self._stateless_tick)(xmat_seq, y_vec_seq, weights_seq, lam_seq, row_valid_seq)

        def moment_step(carry, values):
            xx_c, xy_c, has_xx_c, has_xy_c, last_xx_c, last_xy_c, t_c = carry
            xmat, y_vec, weights, hl, lam = values
            del lam
            xx_new, xy_new, xx_valid, xy_valid = self._moments(xmat, y_vec, weights)
            hl_value = _scalar_value(hl)
            instant = (hl_value <= 0.0) | jnp.isnan(hl_value)
            rho = jnp.where(instant, 0.0, jnp.exp(jnp.log(0.5) / hl_value))
            alpha = jnp.clip(1.0 - rho, 0.0, 1.0)

            a_xx = alpha  # same observation-time clock as tick()
            a_xy = alpha
            updated_xx = jnp.where(has_xx_c, xx_c * (1.0 - a_xx) + xx_new * a_xx, xx_new)
            updated_xy = jnp.where(has_xy_c, xy_c * (1.0 - a_xy) + xy_new * a_xy, xy_new)
            xx = jnp.where(xx_valid, updated_xx, xx_c)
            xy = jnp.where(xy_valid, updated_xy, xy_c)
            xx = jnp.where(instant, jnp.where(xx_valid, xx_new, 0.0), xx)
            xy = jnp.where(instant, jnp.where(xy_valid, xy_new, 0.0), xy)
            has_xx = jnp.where(instant, xx_valid, has_xx_c | xx_valid)
            has_xy = jnp.where(instant, xy_valid, has_xy_c | xy_valid)
            last_xx = jnp.where(xx_valid, t_c, last_xx_c)
            last_xy = jnp.where(xy_valid, t_c, last_xy_c)

            xx = 0.5 * (xx + xx.T)
            last_xx = jnp.maximum(last_xx, last_xx.T)
            has_xx = has_xx | has_xx.T
            next_carry = (xx, xy, has_xx, has_xy, last_xx, last_xy, t_c + 1)
            return next_carry, (xx, xy)

        init = (state.xx, state.xy, state.has_xx, state.has_xy, state.last_xx, state.last_xy, state.t)
        carry, (xx_seq, xy_seq) = jax.lax.scan(
            moment_step,
            init,
            (xmat_seq, y_vec_seq, weights_seq, hl_seq, lam_seq),
            unroll=32,
        )

        lam_values = jax.vmap(
            lambda lam: jnp.maximum(jnp.where(jnp.isnan(_scalar_value(lam)), 0.0, _scalar_value(lam)), 0.0)
        )(lam_seq)
        diag_seq = jax.vmap(lambda xx: jnp.diag(jnp.diag(xx)))(xx_seq)
        systems = xx_seq + lam_values[:, None, None] * diag_seq
        beta_candidates = jax.vmap(lambda system, xy: self._solve_system(system, xy, jnp.zeros_like(xy)))(systems, xy_seq)
        finite_beta = jnp.all(jnp.isfinite(beta_candidates), axis=1)

        def beta_step(beta_prev, values):
            beta_candidate, finite, system, xy, xmat, y_vec, hl = values
            row_valid = jnp.isfinite(y_vec) & jnp.all(jnp.isfinite(xmat), axis=1)
            instant = (_scalar_value(hl) <= 0.0) | jnp.isnan(_scalar_value(hl))
            beta_fallback = jnp.where(instant, jnp.zeros_like(beta_prev), beta_prev)
            beta = self._finish_solve(beta_candidate, finite, system, xy, beta_fallback)
            emit_beta = jnp.where(instant, beta, beta_prev)
            preds = jnp.where(row_valid, xmat @ emit_beta, jnp.nan)
            return beta, (beta, preds)

        beta, (beta_seq, preds_seq) = jax.lax.scan(
            beta_step,
            state.beta,
            (beta_candidates, finite_beta, systems, xy_seq, xmat_seq, y_vec_seq, hl_seq),
            unroll=32,
        )
        instant_seq = jax.vmap(lambda hl: (_scalar_value(hl) <= 0.0) | jnp.isnan(_scalar_value(hl)))(hl_seq)
        _, instant_out = jax.vmap(self._stateless_tick)(xmat_seq, y_vec_seq, weights_seq, lam_seq, row_valid_seq)
        beta_seq = jnp.where(instant_seq[:, None], instant_out.beta, beta_seq)
        preds_seq = jnp.where(instant_seq[:, None], instant_out.preds, preds_seq)
        beta = jnp.where(instant_seq[-1], beta_seq[-1], beta)
        xx, xy, has_xx, has_xy, last_xx, last_xy, t = carry
        next_state = RidgeState(
            xx=xx,
            xy=xy,
            has_xx=has_xx,
            has_xy=has_xy,
            last_xx=last_xx,
            last_xy=last_xy,
            beta=beta,
            preds=preds_seq[-1],
            t=t,
        )
        return next_state, RidgeValue(beta=beta_seq, preds=preds_seq)

    def _solve_system(self, system, rhs, fallback):
        if self.nonneg:
            return self._solve_nonnegative(system, rhs, fallback)
        return self._solve_with_pinv_fallback(system, rhs, fallback)

    @staticmethod
    def _solve_with_pinv_fallback(system, rhs, fallback):
        beta_candidate = jnp.linalg.solve(system, rhs)
        finite = jnp.all(jnp.isfinite(beta_candidate))
        return RidgeOp._finish_solve(beta_candidate, finite, system, rhs, fallback)

    @staticmethod
    def _finish_solve(beta_candidate, finite, system, rhs, fallback):
        beta = jax.lax.cond(
            finite,
            lambda _: beta_candidate,
            lambda _: jnp.linalg.pinv(system) @ rhs,
            operand=None,
        )
        return jnp.where(jnp.all(jnp.isfinite(beta)), beta, fallback)

    @staticmethod
    def _solve_nonnegative(system, rhs, fallback):
        valid = jnp.all(jnp.isfinite(system)) & jnp.all(jnp.isfinite(rhs))
        beta_candidate = nnqp(system, rhs)
        beta = jnp.where(jnp.all(jnp.isfinite(beta_candidate)), beta_candidate, jnp.maximum(fallback, 0.0))
        return jnp.where(valid, beta, jnp.maximum(fallback, 0.0))

    @staticmethod
    def _as_feature_matrix(value):
        value = jnp.asarray(value)
        if value.ndim == 1:
            return value[:, None]
        return value

    @classmethod
    def _moments(cls, xmat, y, weights):
        valid_x = jnp.isfinite(xmat)
        valid_y = jnp.isfinite(y)
        x0 = jnp.where(valid_x, xmat, 0.0)
        y0 = jnp.where(valid_y, y, 0.0)
        weights = jnp.asarray(weights)
        if weights.ndim == 0:
            w = jnp.full((xmat.shape[0],), weights)
            return cls._vector_weight_moments(x0, y0, w, valid_x, valid_y, jnp.isfinite(w))
        if weights.ndim == 1:
            w = weights
            return cls._vector_weight_moments(x0, y0, jnp.where(jnp.isfinite(w), w, 0.0), valid_x, valid_y, jnp.isfinite(w))
        if weights.shape[0] == 1 and weights.shape[1] == 1:
            w = jnp.full((xmat.shape[0],), weights[0, 0])
            return cls._vector_weight_moments(x0, y0, w, valid_x, valid_y, jnp.isfinite(w))
        if weights.shape[1] == 1:
            w = weights[:, 0]
            return cls._vector_weight_moments(x0, y0, jnp.where(jnp.isfinite(w), w, 0.0), valid_x, valid_y, jnp.isfinite(w))
        valid_w = jnp.isfinite(weights)
        w0 = jnp.where(valid_w, weights, 0.0)
        xx_new = x0.T @ w0 @ x0
        xy_new = x0.T @ (w0 @ y0)
        xx_valid = (valid_x.astype(jnp.int64).T @ (valid_w.astype(jnp.int64) @ valid_x.astype(jnp.int64))) > 0
        xy_valid = (valid_x.astype(jnp.int64).T @ (valid_w.astype(jnp.int64) @ valid_y.astype(jnp.int64))) > 0
        return xx_new, xy_new, xx_valid, xy_valid

    @staticmethod
    def _vector_weight_moments(x0, y0, w, valid_x, valid_y, valid_w):
        xw = x0 * w[:, None]
        xx_new = x0.T @ xw
        xx_counts = valid_x.astype(jnp.int64).T @ (valid_x & valid_w[:, None]).astype(jnp.int64)
        xy_new = x0.T @ (w * y0)
        xy_counts = valid_x.astype(jnp.int64).T @ (valid_y & valid_w).astype(jnp.int64)
        return xx_new, xy_new, xx_counts > 0, xy_counts > 0


def _as_tick_matrix(value, rows: int | None = None):
    value = jnp.asarray(value)
    if value.ndim == 0:
        if rows is None:
            return value[None]
        return jnp.broadcast_to(value, (rows, 1))
    if value.ndim == 1:
        return value[:, None]
    return value


def _cat(*values):
    arrays = tuple(jnp.asarray(value) for value in values)
    rows = next((array.shape[0] for array in arrays if array.ndim >= 1), None)
    return jnp.concatenate(tuple(_as_tick_matrix(array, rows) for array in arrays), axis=-1)


def _einsum(subscripts, *child_values):
    return jnp.einsum(subscripts, *child_values)


def _broadcast_sequence_to_state(x, state_value):
    x = jnp.asarray(x)
    state_value = jnp.asarray(state_value)
    if x.ndim == 1 and state_value.ndim == 1:
        return jnp.broadcast_to(x[:, None], (x.shape[0], state_value.shape[0]))
    return x



def _scalar_value(x):
    return jnp.ravel(jnp.asarray(x))[0]


def _lag_vector(x, rows: int):
    x = jnp.asarray(x)
    if x.ndim == 0:
        return jnp.broadcast_to(x, (rows,))
    if x.ndim == 1:
        return x
    return x[:, 0]



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


def _norm_inv(x):
    return jsp_special.ndtri(x)


def _xs_norm(x):
    denom = jnp.nansum(jnp.abs(x))
    return jnp.where(denom > 0.0, x / denom, jnp.nan)


def _xs_rank(x):
    valid = jnp.isfinite(x)
    n_valid = jnp.sum(valid).astype(jnp.int32)
    compact = jnp.where(valid, x, jnp.nan)
    sorted_compact = jnp.sort(compact)
    le_counts = jnp.minimum(jnp.searchsorted(sorted_compact, x, side="right"), n_valid)
    ranks = le_counts.astype(jnp.float64) / (n_valid.astype(jnp.float64) + 1.0)
    return _norm_inv(jnp.where(valid, ranks, jnp.nan))


def _xs_sort(x):
    return jnp.sort(x)


def _bspline(x, n_basis: int):
    x = jnp.asarray(x)
    clipped = jnp.clip(x, 0.0, 1.0)
    centers = jnp.arange(n_basis, dtype=jnp.float64) / n_basis
    sigma = 1.0 / n_basis
    dist = jnp.abs(clipped[:, None] - centers[None, :])
    circ_dist = jnp.minimum(dist, 1.0 - dist)
    values = jnp.exp(-0.5 * (circ_dist / sigma) ** 2)
    total = jnp.sum(values, axis=-1, keepdims=True)
    values = jnp.where(total <= 1e-18, 1.0 / n_basis, values / total)
    return jnp.where(jnp.isnan(x)[:, None], jnp.nan, values)



def _col(matrix, index: int):
    return jnp.asarray(matrix)[:, index]


ANY_ARITY = -1


OP_FACTORIES: dict[tuple[str, int], Callable[..., Op]] = {
    ("abs", 1): lambda: NaryOp(jnp.abs, cpp_name="abs"),
    ("ln", 1): lambda: NaryOp(jnp.log, cpp_name="ln"),
    ("ceil", 1): lambda: NaryOp(jnp.ceil, cpp_name="ceil"),
    ("floor", 1): lambda: NaryOp(jnp.floor, cpp_name="floor"),
    ("round", 1): lambda: NaryOp(jnp.round, cpp_name="round"),
    ("exp", 1): lambda: NaryOp(jnp.exp, cpp_name="exp"),
    ("sign", 1): lambda: NaryOp(jnp.sign, cpp_name="sign"),
    ("arctan", 1): lambda: NaryOp(jnp.arctan, cpp_name="arctan"),
    ("isnan", 1): lambda: NaryOp(lambda x: jnp.where(jnp.isnan(x), 1.0, 0.0), cpp_name="isnan"),
    ("purify", 1): lambda: NaryOp(lambda x: jnp.where(jnp.isfinite(x), x, jnp.nan), cpp_name="purify"),
    ("fraction", 1): lambda: NaryOp(lambda x: x - jnp.floor(x), cpp_name="fraction"),
    ("norm_inv", 1): lambda: NaryOp(_norm_inv, cpp_name="norm_inv"),
    ("xs_norm", 1): lambda: NaryOp(_xs_norm, cpp_name="xs_norm"),
    ("xs_rank", 1): lambda: NaryOp(_xs_rank, cpp_name="xs_rank"),
    ("get_beta", 1): lambda: NaryOp(lambda x: x.beta, cpp_name="get_beta"),
    ("get_preds", 1): lambda: NaryOp(lambda x: x.preds, cpp_name="get_preds"),
    ("xs_sort", 1): lambda: NaryOp(_xs_sort, cpp_name="xs_sort"),
    ("xstd", 1): lambda: NaryOp(_xstd, cpp_name="xstd"),
    ("mean", 1): lambda: NaryOp(lambda x: jnp.nanmean(x), output_kind="scalar", cpp_name="mean"),
    ("outer", 1): lambda: NaryOp(lambda x: x[:, None] * x[None, :], output_kind="matrix", output_width=None, cpp_name="outer"),
    ("cumsum", 1): lambda: CumsumOp(),
    ("add", 2): lambda: NaryOp(lambda l, r: l + r, cpp_name="add"),
    ("sub", 2): lambda: NaryOp(lambda l, r: l - r, cpp_name="sub"),
    ("mul", 2): lambda: NaryOp(lambda l, r: l * r, cpp_name="mul"),
    ("mod", 2): lambda: NaryOp(lambda l, r: jnp.mod(l, r), cpp_name="mod"),
    ("pow", 2): lambda: NaryOp(lambda l, r: l**r, cpp_name="pow"),
    ("div", 2): lambda: NaryOp(lambda l, r: jnp.where(r == 0.0, jnp.nan, l / r), cpp_name="div"),
    ("floordiv", 2): lambda: NaryOp(lambda l, r: jnp.where(r == 0.0, jnp.nan, l // r), cpp_name="floordiv"),
    ("eq", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l == r), cpp_name="eq"),
    ("ne", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l != r), cpp_name="ne"),
    ("lt", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l < r), cpp_name="lt"),
    ("gt", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l > r), cpp_name="gt"),
    ("le", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l <= r), cpp_name="le"),
    ("ge", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l >= r), cpp_name="ge"),
    ("and", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) & (r != 0.0)), cpp_name="and"),
    ("and_", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) & (r != 0.0)), cpp_name="and"),
    ("or", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) | (r != 0.0)), cpp_name="or"),
    ("or_", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) | (r != 0.0)), cpp_name="or"),
    ("xor", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) ^ (r != 0.0)), cpp_name="xor"),
    ("fillna", 2): lambda: NaryOp(lambda l, r: jnp.where(jnp.isnan(l), r, l), cpp_name="fillna"),
    ("where", 3): lambda: NaryOp(lambda c, t, f: jnp.where(c != 0.0, t, f), cpp_name="where"),
    ("clip", 3): lambda: NaryOp(lambda x, lo, hi: jnp.clip(x, lo, hi), cpp_name="clip"),
    ("einsum", ANY_ARITY): lambda subscripts: NaryOp(lambda *child_values: _einsum(subscripts, *child_values), cpp_name="einsum", cpp_str_param=str(subscripts)),
}

from trading_dsl_engine.jax_flat.ops_groupby import *

__all__ = ["GroupByOp"]
