from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

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
    initialized: jax.Array


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
class RidgeValue:
    beta: jax.Array
    preds: jax.Array


@dataclass(frozen=True)
class InputOp(Op):
    input_index: int


@dataclass(frozen=True)
class LiteralOp(Op):
    value: float
    output_kind: str = "scalar"
    output_width: int | None = 1


@dataclass(frozen=True)
class NaryOp(Op):
    fn: Callable[..., jax.Array]
    output_kind: str = "vector"
    output_width: int | None = 1

    def tick(self, state: Any, *child_values: jax.Array):
        del state
        return None, self.fn(*child_values)

    def scan_batch(self, state: Any, *child_sequences: jax.Array):
        del state
        return None, jax.vmap(self.fn)(*child_sequences)


@dataclass(frozen=True)
class EwmOp(Op):
    span: float | None = None
    output_kind: str = "vector"
    output_width: int | None = 1
    is_stateful: bool = True

    def init_state(self, sample: jax.Array):
        return EwmState(value=jnp.zeros_like(sample), initialized=jnp.zeros_like(sample, dtype=bool))

    def tick(self, state: EwmState, *child_values: jax.Array):
        x = child_values[0]
        span = self.span if self.span is not None else _scalar_value(child_values[1])
        value, initialized = state.value, state.initialized
        alpha = 2.0 / (span + 1.0)
        valid = jnp.isfinite(x)
        init_or_valid = initialized | valid
        blended = alpha * x + (1.0 - alpha) * value
        next_value = jnp.where(valid, jnp.where(initialized, blended, x), value)
        out = jnp.where(init_or_valid, next_value, jnp.nan)
        return EwmState(value=next_value, initialized=init_or_valid), out


@dataclass(frozen=True)
class ShiftOp(Op):
    max_size: int
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
        x, lag = child_values[:2]
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
        x, lag = child_sequences[:2]
        cap = state.buffer.shape[0]
        rows, cols = x.shape
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
        history = jnp.swapaxes(state.buffer[read_pos], 0, 1)

        min_values = _lag_vector(min_lag, n)
        upper_values = _lag_vector(upper_lag, n)
        finite_bounds = jnp.isfinite(min_values) & jnp.isfinite(upper_values)
        min_bound = jnp.rint(jnp.where(jnp.isfinite(min_values), min_values, 0.0))
        upper_bound = jnp.rint(jnp.where(jnp.isfinite(upper_values), upper_values, -1.0))
        lag_cols_f = lag_cols.astype(jnp.float64)
        available = state.count >= lag_cols
        in_window = (lag_cols_f[None, :] >= min_bound[:, None]) & (lag_cols_f[None, :] <= upper_bound[:, None])
        valid = finite_bounds[:, None] & available[None, :] & in_window
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
        rows, cols = x.shape
        lag_cols = jnp.arange(1, self.max_lag + 1, dtype=jnp.int32)
        history = jnp.concatenate((_chronological_buffer(state)[-self.max_lag :], x), axis=0)
        if self.max_lag <= 32:
            out = jnp.stack(
                tuple(history[self.max_lag - lag : self.max_lag - lag + rows] for lag in range(1, self.max_lag + 1)),
                axis=2,
            )
        else:
            time_idx = jnp.arange(rows, dtype=jnp.int32)[:, None, None]
            col_idx = jnp.arange(cols, dtype=jnp.int32)[None, :, None]
            lag_idx = lag_cols[None, None, :]
            out = history[self.max_lag + time_idx - lag_idx, col_idx]

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
        return _shift_next_state(state, x), jnp.where(valid, out, jnp.nan)


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
class RidgeOp(Op):
    feature_widths: tuple[int, ...]
    has_weights: bool
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

        features = tuple(_as_feature_matrix(value) for value in feature_values)
        xmat = jnp.concatenate(features, axis=1)
        y = jnp.asarray(y)
        y_vec = y[:, 0] if y.ndim == 2 else y
        row_valid = jnp.isfinite(y_vec) & jnp.all(jnp.isfinite(xmat), axis=1)
        preds = jnp.where(row_valid, xmat @ state.beta, jnp.nan)

        xx_new, xy_new, xx_valid, xy_valid = _ridge_moments(xmat, y_vec, weights)
        hl_value = _scalar_value(hl)
        lam_value = jnp.maximum(jnp.where(jnp.isnan(_scalar_value(lam)), 0.0, _scalar_value(lam)), 0.0)
        rho = jnp.where((hl_value <= 0.0) | jnp.isnan(hl_value), 0.0, jnp.exp(jnp.log(0.5) / hl_value))
        alpha = jnp.clip(1.0 - rho, 0.0, 1.0)

        a_xx = alpha ** (state.t - state.last_xx)
        a_xy = alpha ** (state.t - state.last_xy)
        updated_xx = jnp.where(state.has_xx, state.xx * (1.0 - a_xx) + xx_new * a_xx, xx_new)
        updated_xy = jnp.where(state.has_xy, state.xy * (1.0 - a_xy) + xy_new * a_xy, xy_new)
        xx = jnp.where(xx_valid, updated_xx, state.xx)
        xy = jnp.where(xy_valid, updated_xy, state.xy)
        has_xx = state.has_xx | xx_valid
        has_xy = state.has_xy | xy_valid
        last_xx = jnp.where(xx_valid, state.t, state.last_xx)
        last_xy = jnp.where(xy_valid, state.t, state.last_xy)

        xx = 0.5 * (xx + xx.T)
        last_xx = jnp.maximum(last_xx, last_xx.T)
        has_xx = has_xx | has_xx.T
        system = xx + lam_value * jnp.diag(jnp.diag(xx))
        beta_candidate = jnp.linalg.solve(system, xy)
        beta = jnp.where(jnp.all(jnp.isfinite(beta_candidate)), beta_candidate, state.beta)
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


def _as_feature_matrix(value):
    value = jnp.asarray(value)
    if value.ndim == 1:
        return value[:, None]
    return value


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


def _bspline(x, n_basis: int):
    x = jnp.asarray(x)
    clipped = jnp.clip(x, 0.0, 1.0)
    centers = jnp.arange(n_basis, dtype=jnp.float64) / n_basis
    sigma = 1.0 / n_basis
    dist = jnp.abs(clipped[:, None] - centers[None, :])
    circ_dist = jnp.minimum(dist, 1.0 - dist)
    values = jnp.exp(-0.5 * (circ_dist / sigma) ** 2)
    total = jnp.sum(values, axis=1, keepdims=True)
    values = jnp.where(total <= 1e-18, 1.0 / n_basis, values / total)
    return jnp.where(jnp.isnan(x)[:, None], jnp.nan, values)


def _ridge_moments(xmat, y, weights):
    valid_x = jnp.isfinite(xmat)
    valid_y = jnp.isfinite(y)
    x0 = jnp.where(valid_x, xmat, 0.0)
    y0 = jnp.where(valid_y, y, 0.0)
    weights = jnp.asarray(weights)
    if weights.ndim == 0:
        w = jnp.full((xmat.shape[0],), weights)
        return _ridge_vector_weight_moments(x0, y0, w, valid_x, valid_y, jnp.isfinite(w))
    if weights.ndim == 1:
        w = weights
        return _ridge_vector_weight_moments(x0, y0, jnp.where(jnp.isfinite(w), w, 0.0), valid_x, valid_y, jnp.isfinite(w))
    if weights.shape[0] == 1 and weights.shape[1] == 1:
        w = jnp.full((xmat.shape[0],), weights[0, 0])
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


def _get_beta(value: RidgeValue):
    return value.beta


def _get_preds(value: RidgeValue):
    return value.preds


def _col(matrix, index: int):
    return jnp.asarray(matrix)[:, index]


ANY_ARITY = -1


OP_FACTORIES: dict[tuple[str, int], Callable[..., Op]] = {
    ("abs", 1): lambda: NaryOp(jnp.abs),
    ("ln", 1): lambda: NaryOp(jnp.log),
    ("ceil", 1): lambda: NaryOp(jnp.ceil),
    ("floor", 1): lambda: NaryOp(jnp.floor),
    ("round", 1): lambda: NaryOp(jnp.round),
    ("exp", 1): lambda: NaryOp(jnp.exp),
    ("sign", 1): lambda: NaryOp(jnp.sign),
    ("arctan", 1): lambda: NaryOp(jnp.arctan),
    ("isnan", 1): lambda: NaryOp(lambda x: jnp.where(jnp.isnan(x), 1.0, 0.0)),
    ("purify", 1): lambda: NaryOp(lambda x: jnp.where(jnp.isfinite(x), x, jnp.nan)),
    ("fraction", 1): lambda: NaryOp(lambda x: x - jnp.floor(x)),
    ("xs_rank", 1): lambda: NaryOp(_xs_rank),
    ("xs_sort", 1): lambda: NaryOp(_xs_sort),
    ("xstd", 1): lambda: NaryOp(_xstd),
    ("mean", 1): lambda: NaryOp(lambda x: jnp.nanmean(x), output_kind="scalar"),
    ("outer", 1): lambda: NaryOp(lambda x: x[:, None] * x[None, :], output_kind="matrix", output_width=None),
    ("cumsum", 1): lambda: CumsumOp(),
    ("get_beta", 1): lambda: NaryOp(_get_beta),
    ("get_preds", 1): lambda: NaryOp(_get_preds),
    ("add", 2): lambda: NaryOp(lambda l, r: l + r),
    ("sub", 2): lambda: NaryOp(lambda l, r: l - r),
    ("mul", 2): lambda: NaryOp(lambda l, r: l * r),
    ("mod", 2): lambda: NaryOp(lambda l, r: jnp.mod(l, r)),
    ("pow", 2): lambda: NaryOp(lambda l, r: l**r),
    ("div", 2): lambda: NaryOp(lambda l, r: jnp.where(r == 0.0, jnp.nan, l / r)),
    ("floordiv", 2): lambda: NaryOp(lambda l, r: jnp.where(r == 0.0, jnp.nan, l // r)),
    ("eq", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l == r)),
    ("ne", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l != r)),
    ("lt", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l < r)),
    ("gt", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l > r)),
    ("and", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) & (r != 0.0))),
    ("and_", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) & (r != 0.0))),
    ("or", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) | (r != 0.0))),
    ("or_", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) | (r != 0.0))),
    ("xor", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) ^ (r != 0.0))),
    ("fillna", 2): lambda: NaryOp(lambda l, r: jnp.where(jnp.isnan(l), r, l)),
    ("where", 3): lambda: NaryOp(lambda c, t, f: jnp.where(c != 0.0, t, f)),
    ("einsum", ANY_ARITY): lambda subscripts: NaryOp(lambda *child_values: _einsum(subscripts, *child_values)),
}

from trading_dsl_engine.jax_flat.ops_groupby import *

__all__ = ["GroupByOp"]
