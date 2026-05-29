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
    span: float
    output_kind: str = "vector"
    output_width: int | None = 1
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


def _broadcast_sequence_to_state(x, state_value):
    x = jnp.asarray(x)
    state_value = jnp.asarray(state_value)
    if x.ndim == 1 and state_value.ndim == 1:
        return jnp.broadcast_to(x[:, None], (x.shape[0], state_value.shape[0]))
    return x


_UNIT_ALIASES = {
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
    "us": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ms": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "T": 60 * 1_000_000_000,
    "min": 60 * 1_000_000_000,
    "minute": 60 * 1_000_000_000,
    "minutes": 60 * 1_000_000_000,
    "H": 60 * 60 * 1_000_000_000,
    "h": 60 * 60 * 1_000_000_000,
    "hour": 60 * 60 * 1_000_000_000,
    "hours": 60 * 60 * 1_000_000_000,
    "D": 24 * 60 * 60 * 1_000_000_000,
    "d": 24 * 60 * 60 * 1_000_000_000,
    "day": 24 * 60 * 60 * 1_000_000_000,
    "days": 24 * 60 * 60 * 1_000_000_000,
}


def _unit_ns(unit: str) -> int:
    try:
        return _UNIT_ALIASES[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported datetime unit {unit!r}") from exc


def _duration_ns(value: str | int | float, default_unit: str = "ns") -> int:
    if isinstance(value, (int, float)):
        return int(value) * _unit_ns(default_unit)
    text = value.strip()
    if not text:
        raise ValueError("Duration offset cannot be empty")
    idx = 0
    while idx < len(text) and (text[idx].isdigit() or text[idx] in "+-."):
        idx += 1
    number = float(text[:idx]) if idx else 1.0
    unit = text[idx:] or default_unit
    return int(round(number * _unit_ns(unit)))


def _to_unit_offset(offset: str | int | float, unit: str) -> int:
    return int(_duration_ns(offset, unit) // _unit_ns(unit))


def _valid_datetime_result(x, value):
    return jnp.where(jnp.isfinite(jnp.asarray(x)), value.astype(jnp.float64), jnp.nan)


def _safe_datetime_int(x):
    x = jnp.asarray(x)
    return jnp.where(jnp.isfinite(x), x, 0).astype(jnp.int64)


def _days_from_timestamp(x, unit: str, offset: str | int | float = 0):
    units_per_day = _unit_ns("D") // _unit_ns(unit)
    return jnp.floor_divide(_safe_datetime_int(x) + _to_unit_offset(offset, unit), units_per_day)


def _civil_from_days(days):
    z = jnp.asarray(days, dtype=jnp.int64) + 719468
    era = jnp.floor_divide(jnp.where(z >= 0, z, z - 146096), 146097)
    doe = z - era * 146097
    yoe = jnp.floor_divide(doe - jnp.floor_divide(doe, 1460) + jnp.floor_divide(doe, 36524) - jnp.floor_divide(doe, 146096), 365)
    y = yoe + era * 400
    doy = doe - (365 * yoe + jnp.floor_divide(yoe, 4) - jnp.floor_divide(yoe, 100))
    mp = jnp.floor_divide(5 * doy + 2, 153)
    d = doy - jnp.floor_divide(153 * mp + 2, 5) + 1
    m = mp + jnp.where(mp < 10, 3, -9)
    y = y + (m <= 2)
    return y, m, d


def _is_leap_year(year):
    return ((year % 4) == 0) & (((year % 100) != 0) | ((year % 400) == 0))


def _dayofyear(x, unit: str = "ns", offset: str | int | float = 0):
    days = _days_from_timestamp(x, unit, offset)
    year, month, day = _civil_from_days(days)
    before = jnp.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=jnp.int64)
    leap_adjust = (month > 2) & _is_leap_year(year)
    return _valid_datetime_result(x, before[month] + day + leap_adjust.astype(jnp.int64))


def _date_part(x, unit: str, part: str, offset: str | int | float = 0):
    days = _days_from_timestamp(x, unit, offset)
    year, month, day = _civil_from_days(days)
    if part == "year":
        return _valid_datetime_result(x, year)
    if part == "month":
        return _valid_datetime_result(x, month)
    if part == "day":
        return _valid_datetime_result(x, day)
    if part == "dayofweek":
        return _valid_datetime_result(x, jnp.mod(days + 3, 7))
    raise ValueError(f"Unsupported date part {part!r}")


def _timeofday_units(x, unit: str = "ns", offset: str | int | float = 0):
    units_per_day = _unit_ns("D") // _unit_ns(unit)
    shifted = _safe_datetime_int(x) + _to_unit_offset(offset, unit)
    return jnp.mod(shifted, units_per_day)


def _timeofday(x, unit: str = "ns", offset: str | int | float = 0):
    value = _timeofday_units(x, unit, offset).astype(jnp.float64) * (_unit_ns(unit) / _unit_ns("s"))
    return jnp.where(jnp.isfinite(jnp.asarray(x)), value, jnp.nan)


def _time_part(x, unit: str, part: str, offset: str | int | float = 0):
    seconds = jnp.floor_divide(_timeofday_units(x, unit, offset) * _unit_ns(unit), _unit_ns("s"))
    if part == "hour":
        return _valid_datetime_result(x, jnp.floor_divide(seconds, 3600))
    if part == "minute":
        return _valid_datetime_result(x, jnp.mod(jnp.floor_divide(seconds, 60), 60))
    if part == "second":
        return _valid_datetime_result(x, jnp.mod(seconds, 60))
    raise ValueError(f"Unsupported time part {part!r}")


def _datetime_round(x, unit: str, freq: str, mode: str):
    stride = _duration_ns(freq, unit) // _unit_ns(unit)
    if stride <= 0:
        raise ValueError("datetime rounding frequency must be positive")
    values = jnp.asarray(x, dtype=jnp.float64)
    stride_value = jnp.asarray(stride, dtype=jnp.float64)
    scaled = values / stride_value
    if mode == "floor":
        rounded = jnp.floor(scaled)
    elif mode == "ceil":
        rounded = jnp.ceil(scaled)
    elif mode == "round":
        rounded = jnp.floor(scaled + 0.5)
    else:
        raise ValueError(f"Unsupported datetime rounding mode {mode!r}")
    return jnp.where(jnp.isfinite(jnp.asarray(x)), rounded * stride_value, jnp.nan)


def _scalar_value(x):
    return jnp.ravel(jnp.asarray(x))[0]


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


OP_FACTORIES: dict[tuple[str, int], Callable[[], Op]] = {
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
}

from trading_dsl_engine.jax_flat.ops_groupby import *

__all__ = ["GroupByOp"]
