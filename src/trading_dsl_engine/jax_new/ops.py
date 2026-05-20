from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class EmptyState:
    value: jax.Array


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


class Op:
    output_kind: str = "vector"

    def init_state(self, sample: jax.Array):
        raise NotImplementedError

    def step(self, state: Any, *child_states: Any) -> Any:
        raise NotImplementedError


def state_value(state):
    return state.value


def _nan_cmp(a, b, pred):
    return jnp.where(jnp.isnan(a) | jnp.isnan(b), jnp.nan, jnp.where(pred, 1.0, 0.0))


@dataclass(frozen=True)
class InputOp(Op):
    input_index: int
    output_kind: str = "vector"

    def init_state(self, sample: jax.Array):
        return EmptyState(value=sample)

    def step(self, state: EmptyState, *child_states: Any) -> EmptyState:
        del child_states
        return state


@dataclass(frozen=True)
class LiteralOp(Op):
    value: float
    output_kind: str = "scalar"

    def init_state(self, sample: jax.Array):
        return EmptyState(value=jnp.asarray(self.value, dtype=jnp.float64))

    def step(self, state: EmptyState, *child_states: Any) -> EmptyState:
        del child_states
        return state


@dataclass(frozen=True)
class NaryOp(Op):
    fn: Callable[..., jax.Array]
    output_kind: str = "vector"

    def init_state(self, sample: jax.Array):
        return EmptyState(value=sample)

    def step(self, state: EmptyState, *child_states: Any) -> EmptyState:
        return EmptyState(value=self.fn(*(state_value(child_state) for child_state in child_states)))


@dataclass(frozen=True)
class EwmOp(Op):
    span: float
    output_kind: str = "vector"

    def init_state(self, sample: jax.Array):
        return EwmState(value=jnp.zeros_like(sample), initialized=jnp.zeros_like(sample, dtype=bool))

    def step(self, state: EwmState, *child_states: Any) -> EwmState:
        x = state_value(child_states[0])
        alpha = 2.0 / (self.span + 1.0)
        valid = jnp.isfinite(x)
        # one boolean blend path to avoid repeated is_valid|is_init checks
        init_or_valid = state.initialized | valid
        blended = alpha * x + (1.0 - alpha) * state.value
        out = jnp.where(state.initialized, blended, x)
        out = jnp.where(init_or_valid, jnp.where(valid, out, state.value), jnp.nan)
        return EwmState(value=out, initialized=init_or_valid)


@dataclass(frozen=True)
class CumsumOp(Op):
    output_kind: str = "vector"

    def init_state(self, sample: jax.Array):
        return CumsumState(value=jnp.zeros_like(sample), initialized=jnp.zeros_like(sample, dtype=bool))

    def step(self, state: CumsumState, *child_states: Any) -> CumsumState:
        x = state_value(child_states[0])
        valid = jnp.isfinite(x)
        init_or_valid = state.initialized | valid
        prev = jnp.where(state.initialized, state.value, 0.0)
        accum = prev + jnp.where(valid, x, 0.0)
        out = jnp.where(init_or_valid, jnp.where(valid, accum, state.value), jnp.nan)
        return CumsumState(value=out, initialized=init_or_valid)

OP_FACTORIES = {
    ("abs", 1): lambda: NaryOp(jnp.abs),
    ("ln", 1): lambda: NaryOp(jnp.log),
    ("ceil", 1): lambda: NaryOp(jnp.ceil),
    ("floor", 1): lambda: NaryOp(jnp.floor),
    ("exp", 1): lambda: NaryOp(jnp.exp),
    ("sign", 1): lambda: NaryOp(jnp.sign),
    ("arctan", 1): lambda: NaryOp(jnp.arctan),
    ("isnan", 1): lambda: NaryOp(lambda x: jnp.where(jnp.isnan(x), 1.0, 0.0)),
    ("purify", 1): lambda: NaryOp(lambda x: jnp.where(jnp.isfinite(x), x, jnp.nan)),
    ("fraction", 1): lambda: NaryOp(lambda x: x - jnp.floor(x)),
    ("xstd", 1): lambda: NaryOp(_xstd),
    ("xs_rank", 1): lambda: NaryOp(_xs_rank),
    ("mean", 1): lambda: NaryOp(lambda x: jnp.nanmean(x), output_kind="scalar"),
    ("outer", 1): lambda: NaryOp(lambda x: x[:, None] * x[None, :], output_kind="matrix"),
    ("cumsum", 1): lambda: CumsumOp(),
    ("add", 2): lambda: NaryOp(lambda l, r: l + r),
    ("sub", 2): lambda: NaryOp(lambda l, r: l - r),
    ("mul", 2): lambda: NaryOp(lambda l, r: l * r),
    ("mod", 2): lambda: NaryOp(jnp.mod),
    ("pow", 2): lambda: NaryOp(lambda l, r: l**r),
    ("div", 2): lambda: NaryOp(lambda l, r: jnp.where(r == 0.0, jnp.nan, l / r)),
    ("floordiv", 2): lambda: NaryOp(lambda l, r: jnp.where(r == 0.0, jnp.nan, l // r)),
    ("eq", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l == r)),
    ("ne", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l != r)),
    ("lt", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l < r)),
    ("gt", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, l > r)),
    ("and", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) & (r != 0.0))),
    ("or", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) | (r != 0.0))),
    ("xor", 2): lambda: NaryOp(lambda l, r: _nan_cmp(l, r, (l != 0.0) ^ (r != 0.0))),
    ("fillna", 2): lambda: NaryOp(lambda l, r: jnp.where(jnp.isnan(l), r, l)),
    ("where", 3): lambda: NaryOp(lambda c, t, f: jnp.where(c != 0.0, t, f)),
}

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
    """NaN-masked cross-sectional percentile rank for a 1D vector.

    Tie semantics match the historical JAX backend: ties map to the
    right-edge rank (equivalent to searchsorted(..., side="right")).
    """
    valid = jnp.isfinite(x)
    n_valid = jnp.sum(valid).astype(jnp.int32)
    compact = jnp.where(valid, x, jnp.inf)
    sorted_compact = jnp.sort(compact)
    le_counts = jnp.minimum(jnp.searchsorted(sorted_compact, x, side="right"), n_valid)
    ranks = le_counts.astype(jnp.float64) / jnp.maximum(n_valid, 1).astype(jnp.float64)
    return jnp.where(valid, ranks, jnp.nan)

