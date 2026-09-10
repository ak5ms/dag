from __future__ import annotations

import math

from trading_dsl_engine.base import dsl
from trading_dsl_engine.base.parser import Expr, Number


_HASH_SCALE = 43_758.545_312_3
_HASH_A = 12.9898
_HASH_B = 78.233
_HASH_STREAM = 37.719
_EPS = 1e-12


def _seed_value(seed: int | float | Number | None) -> int:
    if seed is None:
        return 0
    if isinstance(seed, Number):
        seed = seed.value
    if isinstance(seed, bool):
        raise TypeError("random seed must be an integer, not bool")
    value = float(seed)
    if not math.isfinite(value) or not value.is_integer():
        raise ValueError("random seed must be a finite integer")
    integer = int(value)
    if abs(integer) > 2**31 - 1:
        raise ValueError("random seed magnitude must be <= 2**31 - 1")
    return integer


def _select_key(key, *parameters) -> Expr:
    if key is not None:
        return dsl.ensure_expr(key)
    for parameter in parameters:
        expr = dsl.ensure_expr(parameter)
        if not isinstance(expr, Number):
            return expr
    # Static distribution parameters need a changing key. Trading datasets carry
    # _ev_ts by convention; callers working outside that convention can pass key=.
    return dsl.var("_ev_ts")


def _unit_interval(key: Expr, *, seed: int, stream: int) -> Expr:
    phase = dsl.add(
        dsl.mul(key, _HASH_A + stream * 0.6180339887498949),
        seed * _HASH_B + stream * _HASH_STREAM,
    )
    raw = dsl.fraction(dsl.abs(dsl.mul(dsl.sin(phase), _HASH_SCALE)))
    return dsl.minimum(dsl.maximum(raw, _EPS), 1.0 - _EPS)


def _normal_z(key: Expr, *, seed: int) -> Expr:
    u1 = _unit_interval(key, seed=seed, stream=0)
    u2 = _unit_interval(dsl.add(key, 0.7548776662466927), seed=seed, stream=1)
    radius = dsl.sqrt(dsl.mul(-2.0, dsl.ln(u1)))
    angle = dsl.mul(2.0 * math.pi, u2)
    return dsl.mul(radius, dsl.cos(angle))


@dsl.register_dsl_function("uniform")
def uniform(low=0.0, high=1.0, *, key=None, seed: int | None = 0) -> Expr:
    """Seeded keyed uniform draw with dynamic ``low``/``high`` parameters.

    ``key`` controls where draws vary. If omitted, the first dynamic parameter is
    used; if all parameters are static the conventional ``_ev_ts`` field is used.
    The implementation is a pure DSL composition, so it works on every backend
    that supports the underlying arithmetic operators.
    """

    seed_value = _seed_value(seed)
    key_expr = _select_key(key, low, high)
    u = _unit_interval(key_expr, seed=seed_value, stream=0)
    return dsl.add(low, dsl.mul(dsl.sub(high, low), u))


@dsl.register_dsl_function("normal")
def normal(mu=0.0, sigma=1.0, *, key=None, seed: int | None = 0) -> Expr:
    """Seeded keyed normal draw with dynamic ``mu`` and ``sigma``."""

    seed_value = _seed_value(seed)
    key_expr = _select_key(key, mu, sigma)
    return dsl.add(mu, dsl.mul(sigma, _normal_z(key_expr, seed=seed_value)))


@dsl.register_dsl_function("lognormal")
def lognormal(mu=0.0, sigma=1.0, *, key=None, seed: int | None = 0) -> Expr:
    """Seeded keyed log-normal draw with dynamic log-space parameters."""

    seed_value = _seed_value(seed)
    key_expr = _select_key(key, mu, sigma)
    return dsl.exp(dsl.add(mu, dsl.mul(sigma, _normal_z(key_expr, seed=seed_value))))


@dsl.register_dsl_function("exponential")
def exponential(scale=1.0, *, key=None, seed: int | None = 0) -> Expr:
    """Seeded keyed exponential draw with a dynamic ``scale`` parameter."""

    seed_value = _seed_value(seed)
    key_expr = _select_key(key, scale)
    u = _unit_interval(key_expr, seed=seed_value, stream=0)
    return dsl.mul(-1.0, dsl.mul(scale, dsl.ln(dsl.sub(1.0, u))))


# Make the helpers available through the familiar ``from ...base import dsl``
# namespace in addition to direct imports from this module.
dsl.uniform = uniform
dsl.normal = normal
dsl.lognormal = lognormal
dsl.exponential = exponential


__all__ = ["exponential", "lognormal", "normal", "uniform"]
