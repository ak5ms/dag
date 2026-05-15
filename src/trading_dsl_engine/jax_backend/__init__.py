"""JAX + Equinox backend for the trading DSL.

The public API mirrors the Numba runtime helpers while keeping live and batch
hot paths inside JAX-compiled functions.
"""

from trading_dsl_engine.jax_backend.engine import (
    JaxEngineHandle,
    build_engine,
    build_jax_engine,
    compile_formula,
    run_batch_from_mapping,
    update_from_mapping,
)

__all__ = [
    "JaxEngineHandle",
    "build_engine",
    "build_jax_engine",
    "compile_formula",
    "run_batch_from_mapping",
    "update_from_mapping",
]
