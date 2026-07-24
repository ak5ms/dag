"""Compatibility facade for the JAX-flat engine.

Implementation lives in sibling modules:

- ``program`` — streaming program IR (``StreamingProgram``, ``DagNode``, ...)
- ``compile`` — formula lowering (``compile_formula``)
- ``runtime`` — live tick and batch execution (``JaxFlatRuntime``)

Import from this module for backward compatibility. To tune batch chunk size at
runtime, assign ``trading_dsl_engine.jax_flat.runtime._BATCH_CHUNK_SIZE`` (not
a copy re-exported here).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.jax_flat.compile import compile_formula
from trading_dsl_engine.jax_flat.program import (
    CacheWriteTarget,
    DagNode,
    InnerGraphOp,
    JitCompileTracker,
    MemmapPathTracker,
    StateFieldRef,
    StateLayout,
    StreamingProgram,
)
from trading_dsl_engine.jax_flat.runtime import (
    JaxFlatRuntime,
    _BATCH_CHUNK_SIZE,
    _CPP_ACCELERATOR_CACHE,
    _cache_output_array,
    _finalize_batch_result,
    _flush_memmap_output,
    _fresh_memmap_path,
    _has_disk_cache_node,
    _has_memmap_input,
    _input_chunk,
    _is_cpp_flat_state,
    _is_groupby_capacity_error,
    _jit_batch,
    _jit_batch_from_initial_state,
    _materialize_cache_value,
    _normalize_batch_inputs,
    _output_array,
    _run_chunked_batch,
    _runtime_with_cache_write_callbacks,
    _scan_batch,
    _scan_batch_chunk,
    _store_cache_arrays,
    _tracked_cache_memmap,
    _warn_cpp_fallback,
    _double_groupby_capacities,
)
from trading_dsl_engine.jax_flat.compile import _build_state_layout

__all__ = [
    "CacheWriteTarget",
    "DagNode",
    "InnerGraphOp",
    "JaxFlatRuntime",
    "JitCompileTracker",
    "MemmapPathTracker",
    "StateFieldRef",
    "StateLayout",
    "StreamingProgram",
    "compile_formula",
    "jax",
    "jnp",
    "np",
    "_BATCH_CHUNK_SIZE",
    "_CPP_ACCELERATOR_CACHE",
    "_build_state_layout",
    "_cache_output_array",
    "_finalize_batch_result",
    "_flush_memmap_output",
    "_fresh_memmap_path",
    "_has_disk_cache_node",
    "_has_memmap_input",
    "_input_chunk",
    "_is_cpp_flat_state",
    "_is_groupby_capacity_error",
    "_jit_batch",
    "_jit_batch_from_initial_state",
    "_materialize_cache_value",
    "_normalize_batch_inputs",
    "_output_array",
    "_run_chunked_batch",
    "_runtime_with_cache_write_callbacks",
    "_scan_batch",
    "_scan_batch_chunk",
    "_store_cache_arrays",
    "_tracked_cache_memmap",
    "_warn_cpp_fallback",
    "_double_groupby_capacities",
]
