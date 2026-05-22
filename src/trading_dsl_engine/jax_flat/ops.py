from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)


class Op:
    output_kind: str = "vector"
    is_stateful: bool = False

    def state_spec(self, sample: jax.Array) -> tuple[tuple[str, jax.Array], ...]:
        return ()

    def lower_stream_step(
        self,
        child_values: tuple[jax.Array, ...],
        state_fields: tuple[jax.Array, ...],
    ) -> tuple[tuple[jax.Array, ...], jax.Array]:
        del child_values, state_fields
        raise NotImplementedError


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

    def lower_stream_step(self, child_values, state_fields):
        del state_fields
        return (), self.fn(*child_values)


@dataclass(frozen=True)
class EwmOp(Op):
    span: float
    output_kind: str = "vector"
    is_stateful: bool = True

    def state_spec(self, sample: jax.Array):
        return (
            ("value", jnp.zeros_like(sample)),
            ("initialized", jnp.zeros_like(sample, dtype=bool)),
        )

    def lower_stream_step(self, child_values, state_fields):
        x = child_values[0]
        value, initialized = state_fields
        alpha = 2.0 / (self.span + 1.0)
        valid = jnp.isfinite(x)
        init_or_valid = initialized | valid
        blended = alpha * x + (1.0 - alpha) * value
        out = jnp.where(initialized, blended, x)
        out = jnp.where(init_or_valid, jnp.where(valid, out, value), jnp.nan)
        return (out, init_or_valid), out


@dataclass(frozen=True)
class CumsumOp(Op):
    output_kind: str = "vector"
    is_stateful: bool = True

    def state_spec(self, sample: jax.Array):
        return (
            ("value", jnp.zeros_like(sample)),
            ("initialized", jnp.zeros_like(sample, dtype=bool)),
        )

    def lower_stream_step(self, child_values, state_fields):
        x = child_values[0]
        value, initialized = state_fields
        valid = jnp.isfinite(x)
        init_or_valid = initialized | valid
        prev = jnp.where(initialized, value, 0.0)
        accum = prev + jnp.where(valid, x, 0.0)
        out = jnp.where(init_or_valid, jnp.where(valid, accum, value), jnp.nan)
        return (out, init_or_valid), out


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
