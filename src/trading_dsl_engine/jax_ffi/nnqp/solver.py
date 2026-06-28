from __future__ import annotations

import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_ffi.nnqp import _eigen_nnqp


# Important: register before first CPU backend/JIT execution.
for name, capsule in _eigen_nnqp.registrations().items():
    jax.ffi.register_ffi_target(name, capsule, platform="cpu", api_version=1)


solve_direct = _eigen_nnqp.solve_direct


def _check(A, c):
    A = jnp.asarray(A)
    c = jnp.asarray(c)

    if A.dtype != jnp.float64 or c.dtype != jnp.float64:
        raise TypeError("float64 only; call jax.config.update('jax_enable_x64', True)")

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must have shape (p, p)")

    if c.ndim != 1 or c.shape[0] != A.shape[0]:
        raise ValueError("c must have shape (p,)")

    return A, c


def nnqp_raw(A, c):
    """
    Raw forward FFI call.

    Solves:
        min_beta 0.5 beta.T @ A @ beta - c.T @ beta
        s.t. beta >= 0
    """
    A, c = _check(A, c)
    p = c.shape[0]

    return jax.ffi.ffi_call(
        "nnqp_eigen_fwd",
        jax.ShapeDtypeStruct((p,), A.dtype),
        input_layouts=[(1, 0), (0,)],
        output_layouts=(0,),
        vmap_method="sequential",
    )(A, c)


@jax.custom_vjp
def nnqp(A, c):
    return nnqp_raw(A, c)


def _nnqp_fwd(A, c):
    A, c = _check(A, c)
    beta = nnqp_raw(A, c)
    return beta, (A, c, beta)


def _nnqp_bwd(res, g):
    A, c, beta = res
    p = c.shape[0]

    dA, dc = jax.ffi.ffi_call(
        "nnqp_eigen_bwd",
        (
            jax.ShapeDtypeStruct((p, p), A.dtype),
            jax.ShapeDtypeStruct((p,), c.dtype),
        ),
        input_layouts=[(1, 0), (0,), (0,), (0,)],
        output_layouts=[(1, 0), (0,)],
        vmap_method="sequential",
    )(A, c, beta, jnp.asarray(g, dtype=c.dtype))

    return dA, dc


nnqp.defvjp(_nnqp_fwd, _nnqp_bwd)
