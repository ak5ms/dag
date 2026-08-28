"""JAX-flat live tick and batch execution runtime.

``JaxFlatRuntime`` drives one streaming timestep via JIT-compiled ``tick`` and
full-history replay via ``run_batch``. The hot path uses ``lax.scan``-style
chunked batch scans; disk/memmap inputs or disk ``cache(...)`` nodes fall back
to a host-side chunked loop. Diagnostics (``inspect_jaxpr``,
``inspect_compiled_hlo``) trace the same tick ABI but stay off the hot path.
"""
from __future__ import annotations

import mmap
import os
import tempfile
import time
import warnings
from dataclasses import replace
from io import BytesIO
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from graphviz import Source

from trading_dsl_engine.jax_flat.ops import CacheOp, GroupByOp, InputOp, LiteralOp
from trading_dsl_engine.jax_flat.program import (
    CacheWriteTarget,
    DagNode,
    JitCompileTracker,
    MemmapPathTracker,
    StreamingProgram,
)


class JaxFlatRuntime(eqx.Module):
    program: StreamingProgram = eqx.field(static=True)
    runtimes: list[float] = eqx.field(default_factory=list, static=True)
    _jit_compile_tracker: JitCompileTracker = eqx.field(default_factory=JitCompileTracker, static=True)
    cpp: bool = eqx.field(default=True, static=True)
    cpp_workers: int | None = eqx.field(default=None, static=True)
    cpp_fallback_warnings: list[str] = eqx.field(default_factory=list, static=True)
    cached_values: dict[int, np.ndarray] = eqx.field(default_factory=dict, static=True)
    cache_memmap_tracker: MemmapPathTracker = eqx.field(default_factory=MemmapPathTracker, static=True)
    block: bool = True # False disables waiting (for runtime measurement)

    @property
    def jit_compile_count(self) -> int:
        """Return the number of observed JIT traces for this runtime.

        A trace is recorded when the tick transition receives a JAX tracer. This
        is a lightweight per-runtime proxy for JIT cache misses/compilations;
        it covers live ``tick`` calls, compiled-HLO inspection, and the
        transition used by batch scans. JAXPR inspection does not affect this
        count because it traces without compiling.
        """
        return self._jit_compile_tracker.count

    def reset_jit_compile_count(self) -> None:
        """Clear this runtime's observed JIT compilation counter."""
        self._jit_compile_tracker.reset()

    def inspect_jaxpr(self, state_leaves, *input_rows):
        """Return the closed JAXPR for one streaming tick transition.

        ``state_leaves`` and ``input_rows`` use the same ABI as :meth:`tick`.
        This is intended for diagnostics and does not execute a live update.
        """
        transition = lambda state, *rows: self._tick_impl(state, *rows)
        return self._inspect_without_count(lambda: jax.make_jaxpr(transition)(state_leaves, *input_rows))

    def inspect_compiled_hlo(self, state_leaves, *input_rows) -> str:
        """Compile one tick specialization and return its optimized HLO text.

        ``state_leaves`` and ``input_rows`` use the same ABI as :meth:`tick`.
        The returned text is XLA's compiled executable representation, rather
        than the pre-compilation HLO emitted by ``compiler_ir``.
        """
        transition = lambda state, *rows: self._tick_impl(state, *rows)
        return jax.jit(transition).lower(state_leaves, *input_rows).compile().as_text()

    # ``get_*`` aliases make these diagnostics convenient in notebooks and
    # preserve a conventional runtime-inspection spelling.
    get_jaxpr = inspect_jaxpr
    get_compiled_hlo = inspect_compiled_hlo

    def explain(self, format: str = "text") -> str:
        """Explain C++/JAX lowering islands as text, JSON, or Graphviz DOT."""
        from trading_dsl_engine.jax_flat.engine_cpp import explain_cpp_plan

        return explain_cpp_plan(self.program).format(format)

    def display(self, figsize=(24,14)):
        dot = self.explain("dot")
        png = Source(dot).pipe(format="png")
        img = plt.imread(BytesIO(png), format="png")
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    def get_lowering_plan(self):
        """Return the structured, serializable C++/JAX lowering plan."""
        from trading_dsl_engine.jax_flat.engine_cpp import explain_cpp_plan

        return explain_cpp_plan(self.program)

    def _inspect_without_count(self, inspect):
        """Run a non-compiling tracing diagnostic without recording its trace."""
        count_before = self._jit_compile_tracker.count
        try:
            return inspect()
        finally:
            self._jit_compile_tracker.count = count_before

    def _record_jit_trace(self, state_leaves, input_rows) -> None:
        if any(
            isinstance(value, jax.core.Tracer)
            for value in jax.tree.leaves((state_leaves, input_rows))
        ):
            self._jit_compile_tracker.record()

    def get_cached_values(self):
        """Return batch-materialized values for top-level cache(...) nodes keyed by node id."""
        return dict(self.cached_values)

    def clear_cached_values(self) -> None:
        """Drop materialized batch cache values from previous runs and remove disk cache files."""
        self.cached_values.clear()
        self.cache_memmap_tracker.cleanup()

    def get_units(self):
        """Return static unit exponents inferred for this formula output."""
        if self.program.metadata is None:
            return None
        return self.program.metadata.get_units()

    def get_range(self):
        """Return the static domain/range interval inferred for this formula output."""
        if self.program.metadata is None:
            return None
        return self.program.metadata.get_range()

    def get_types(self):
        """Return semantic output types after applying configured type relations."""
        if self.program.metadata is None:
            return frozenset()
        return self.program.metadata.get_types()

    def get_node_metadata(self, label: str | None = None):
        """Return static metadata for analyzed formula nodes, optionally filtered by label."""
        if self.program.metadata is None:
            return ()
        return self.program.metadata.get_node_metadata(label)

    def get_node_types(self, label: str):
        """Return semantic type sets for analyzed formula nodes with the given label."""
        if self.program.metadata is None:
            return ()
        return self.program.metadata.get_node_types(label)

    def get_type_relations(self):
        """Return the configured semantic type relation graph."""
        if self.program.metadata is None:
            return None
        return self.program.metadata.type_graph

    def init_state(self, n_instruments: int):
        vector_sample = jnp.zeros((n_instruments,), dtype=jnp.float64)
        values: list[Any] = [vector_sample] * len(self.program.nodes)
        states = []
        for idx, node in enumerate(self.program.nodes):
            op = node.op
            if isinstance(op, InputOp):
                if op.output_kind == "matrix":
                    values[idx] = jnp.zeros((n_instruments, op.output_width or n_instruments), dtype=jnp.float64)
                else:
                    values[idx] = vector_sample
                continue
            if isinstance(op, LiteralOp):
                values[idx] = jnp.asarray(op.value, dtype=jnp.float64)
                continue

            child_values = tuple(values[cid] for cid in node.child_ids)
            if op.is_stateful:
                sample = child_values[0] if child_values else vector_sample
                if jnp.asarray(sample).ndim == 0:
                    sample = vector_sample
                state = op.init_state(sample)
                states.append(state)
                _, value = op.tick(state, *child_values)
            else:
                _, value = op.tick(None, *child_values)
            values[idx] = value
        return tuple(states)

    def _tick_impl(self, state_leaves, *input_rows):
        self._record_jit_trace(state_leaves, input_rows)
        values: list[jax.Array] = [jnp.array(0.0)] * len(self.program.nodes)
        new_state = list(state_leaves)

        for idx, node in enumerate(self.program.nodes):
            op = node.op
            if isinstance(op, InputOp):
                values[idx] = input_rows[op.input_index]
                continue
            if isinstance(op, LiteralOp):
                values[idx] = jnp.asarray(op.value, dtype=jnp.float64)
                continue

            child_values = tuple(values[cid] for cid in node.child_ids)
            field = self.program.state_layout.node_fields[idx]
            node_state = None if field.index < 0 else state_leaves[field.index]
            next_state, value = op.tick(node_state, *child_values)
            if field.index >= 0:
                new_state[field.index] = next_state
            values[idx] = value

        outs = tuple(values[i] for i in self.program.outputs)
        return tuple(new_state), outs[0] if len(outs) == 1 else jnp.stack(outs, axis=0)

    @jax.jit
    def tick(self, state_leaves, *input_rows):
        return self._tick_impl(state_leaves, *input_rows)

    def run_batch(self, inputs, states=None, out_path: str | bool = False):

        runtime = self

        while True:
            start = time.perf_counter()
            try:
                result = runtime._run_batch_once(inputs, states, out_path)
                if runtime.block:
                    jax.block_until_ready(result)
                end = time.perf_counter()
                runtime.runtimes.append(end - start)
                return result

            except Exception as exc:
                if states or not _is_groupby_capacity_error(exc):
                    raise
                next_runtime = _double_groupby_capacities(runtime)
                if next_runtime is runtime:
                    raise
                warnings.warn(
                    "jax_flat groupby capacity/hash table exhausted; retrying run_batch with 2x group key capacity",
                    RuntimeWarning,
                    stacklevel=2,
                )
                runtime = next_runtime

    def _run_batch_once(self, inputs, states=None, out_path: str | bool = False):
        inputs = _normalize_batch_inputs(self, inputs)
        if not inputs:
            raise ValueError("run_batch requires at least one input array")
        n_steps = inputs[0].shape[0]
        n_instruments = inputs[0].shape[1]
        for arr in inputs[1:]:
            if arr.shape[0] != n_steps or arr.shape[1] != n_instruments:
                raise ValueError("All inputs must share aligned shape (time, n_instruments)")
        if self.program.cache_nodes:
            self.clear_cached_values()
        if self.cpp and not self.program.cache_nodes and not states:
            try:
                from trading_dsl_engine.jax_flat.engine_cpp import _try_cpp_hybrid_batch
            except Exception as exc:
                _warn_cpp_fallback(self, f"C++ jax_flat accelerator unavailable ({type(exc).__name__}: {exc}); falling back to JAX-flat")
            else:
                hybrid = _try_cpp_hybrid_batch(
                    self, inputs, _CPP_ACCELERATOR_CACHE, _warn_cpp_fallback,
                    out_path=out_path, workers=self.cpp_workers
                )
                if hybrid is not None:
                    if isinstance(hybrid[1], np.memmap):
                        hybrid[1].flush()
                    return hybrid
        if _has_memmap_input(inputs) or out_path or _has_disk_cache_node(self):
            return _run_chunked_batch(self, inputs, states, out_path)
        result = _jit_batch_from_initial_state(self, inputs) if not states else _jit_batch(self, states, inputs)
        return _finalize_batch_result(self, result)



def _is_cpp_flat_state(states) -> bool:
    return type(states).__name__ == "CppFlatState"



def _warn_cpp_fallback(runtime: JaxFlatRuntime, message: str) -> None:
    if message in runtime.cpp_fallback_warnings:
        return
    runtime.cpp_fallback_warnings.append(message)
    warnings.warn(message, RuntimeWarning, stacklevel=3)


def _normalize_batch_inputs(runtime: JaxFlatRuntime, inputs):
    external_cache_inputs = runtime.program.external_cache_inputs or {}
    if isinstance(inputs, dict):
        missing = [
            name
            for name in runtime.program.input_names
            if name not in inputs and name not in external_cache_inputs
        ]
        if missing:
            raise ValueError(f"Missing jax_flat run_batch input(s): {missing}")
        if len(runtime.program.input_names) == 0:
            inputs = tuple(inputs.values())
        else:
            inputs = tuple(
                external_cache_inputs[name] if name in external_cache_inputs else inputs[name]
                for name in runtime.program.input_names
            )
    else:
        user_inputs = tuple(inputs)
        if external_cache_inputs:
            if len(user_inputs) == len(runtime.program.input_names):
                inputs = user_inputs
            else:
                user_iter = iter(user_inputs)
                ordered = []
                for name in runtime.program.input_names:
                    if name in external_cache_inputs:
                        ordered.append(external_cache_inputs[name])
                    else:
                        ordered.append(next(user_iter))
                remaining = tuple(user_iter)
                if remaining:
                    raise ValueError(
                        "run_batch received more positional input array(s) than non-cached formula inputs"
                    )
                inputs = tuple(ordered)
        else:
            inputs = user_inputs

    if len(inputs) != len(runtime.program.input_names):
        if len(runtime.program.input_names) == 0 and len(inputs) == 1:
            pass
        else:
            raise ValueError(
                "run_batch expected "
                f"{len(runtime.program.input_names)} input array(s) "
                f"({runtime.program.input_names}), got {len(inputs)}"
            )
    for name, arr in zip(runtime.program.input_names, inputs):
        if arr.ndim != 2 and not name.startswith("__cpp_subgraph_"):
            raise ValueError(f"Expected 2D input for '{name}', got shape {arr.shape}")
    return inputs


def _has_memmap_input(inputs) -> bool:
    return any(isinstance(arr, np.memmap) for arr in inputs)


def _has_disk_cache_node(runtime: JaxFlatRuntime) -> bool:
    return any(
        isinstance(runtime.program.nodes[node_id].op, CacheOp)
        and runtime.program.nodes[node_id].op.storage == "disk"
        for node_id in runtime.program.cache_nodes
    )


def _fresh_memmap_path(prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".memmap")
    os.close(fd)
    return path


def _tracked_cache_memmap(runtime: JaxFlatRuntime, node_id: int, dtype, shape):
    run_index = len(runtime.runtimes)
    prefix = f"trading_dsl_engine_jax_flat_cache_pid{os.getpid()}_rt{id(runtime):x}_run{run_index}_node{node_id}_"
    path = _fresh_memmap_path(prefix)
    out = np.memmap(path, mode="w+", dtype=dtype, shape=shape)
    runtime.cache_memmap_tracker.add(out)
    return out


def _input_chunk(arr, start: int, stop: int):
    chunk = arr[start:stop]
    if isinstance(chunk, np.memmap) and chunk.dtype == np.float64:
        return chunk
    if isinstance(chunk, np.ndarray):
        return np.asarray(chunk, dtype=np.float64)
    return chunk


def _output_array(out_path: str | bool, n_steps: int, chunk_out: np.ndarray):
    shape = (n_steps,) + chunk_out.shape[1:]
    if out_path is False or out_path is None:
        return np.empty(shape, dtype=chunk_out.dtype)
    if out_path is True:
        out_path = _fresh_memmap_path("trading_dsl_engine_jax_flat_out_")
    if isinstance(out_path, str):
        return np.memmap(out_path, mode="w+", dtype=chunk_out.dtype, shape=shape)
    raise ValueError("out_path must be False, True, or a filesystem path string")


def _flush_memmap_output(out) -> None:
    if not isinstance(out, np.memmap):
        return
    out.flush()
    mapped = getattr(out, "_mmap", None)
    madvise = getattr(mapped, "madvise", None)
    if madvise is not None and hasattr(mmap, "MADV_DONTNEED"):
        madvise(mmap.MADV_DONTNEED)


def _run_chunked_batch(runtime: JaxFlatRuntime, inputs, states=None, out_path: str | bool = False):
    n_steps = inputs[0].shape[0]
    n_instruments = inputs[0].shape[1]
    chunk_size = min(n_steps, _BATCH_CHUNK_SIZE)
    states = runtime.init_state(n_instruments) if not states else states
    cache_outs = tuple(_cache_output_array(runtime, node_id, n_steps, n_instruments) for node_id in runtime.program.cache_nodes)
    callback_runtime = _runtime_with_cache_write_callbacks(runtime, cache_outs)
    root_cache_idx = runtime.program.cache_nodes.index(runtime.program.outputs[0]) if runtime.program.outputs[0] in runtime.program.cache_nodes else None
    out = cache_outs[root_cache_idx] if root_cache_idx is not None and isinstance(cache_outs[root_cache_idx], np.memmap) else None
    root_cache_output = out is not None

    for start in range(0, n_steps, chunk_size):
        stop = min(start + chunk_size, n_steps)
        chunk_inputs = tuple(jnp.asarray(_input_chunk(arr, start, stop)) for arr in inputs)
        states, chunk_out, _ = _scan_batch_chunk(callback_runtime, states, chunk_inputs, start)
        if root_cache_output:
            jax.block_until_ready(chunk_out)
            continue
        chunk_out_np = np.asarray(jax.block_until_ready(chunk_out))
        out = _output_array(out_path, n_steps, chunk_out_np) if out is None else out
        out[start:stop] = chunk_out_np
        _flush_memmap_output(out)

    _store_cache_arrays(runtime, cache_outs or ())
    return states, out



def _runtime_with_cache_write_callbacks(runtime: JaxFlatRuntime, cache_outs) -> JaxFlatRuntime:
    if not cache_outs:
        return runtime
    targets = dict(zip(runtime.program.cache_nodes, map(CacheWriteTarget, cache_outs)))
    return replace(
        runtime,
        program=replace(
            runtime.program,
            nodes=tuple(
                replace(node, op=replace(node.op, cache_write_target=targets[node_id]))
                if node_id in targets and isinstance(node.op, CacheOp)
                else node
                for node_id, node in enumerate(runtime.program.nodes)
            ),
        ),
    )


def _cache_output_array(runtime: JaxFlatRuntime, node_id: int, n_steps: int, n_instruments: int):
    op = runtime.program.nodes[node_id].op
    if not isinstance(op, CacheOp):
        raise TypeError(f"cache node {node_id} is {type(op).__name__}")
    if op.output_kind == "scalar":
        shape = (n_steps,)
    elif op.output_kind == "vector":
        shape = (n_steps, n_instruments)
    elif op.output_kind == "matrix" and op.output_width is not None:
        shape = (n_steps, n_instruments, int(op.output_width))
    else:
        raise ValueError(f"Cannot infer cache output shape for cache node {node_id}")
    return _tracked_cache_memmap(runtime, node_id, np.float64, shape) if op.storage == "disk" else np.empty(shape, dtype=np.float64)


def _store_cache_arrays(runtime: JaxFlatRuntime, values) -> None:
    runtime.cached_values.clear()
    for node_id, value in zip(runtime.program.cache_nodes, values):
        runtime.cached_values[int(node_id)] = value


def _finalize_batch_result(runtime: JaxFlatRuntime, result):
    states, out, cache_outs = result
    materialized = tuple(_materialize_cache_value(runtime, node_id, value) for node_id, value in zip(runtime.program.cache_nodes, cache_outs))
    _store_cache_arrays(runtime, materialized)
    return states, out


def _materialize_cache_value(runtime: JaxFlatRuntime, node_id: int, value):
    value_np = np.asarray(jax.block_until_ready(value))
    op = runtime.program.nodes[node_id].op
    if isinstance(op, CacheOp) and op.storage == "disk":
        out = _tracked_cache_memmap(runtime, node_id, value_np.dtype, value_np.shape)
        out[:] = value_np
        _flush_memmap_output(out)
        return out
    return value_np

def _is_groupby_capacity_error(exc: Exception) -> bool:
    message = str(exc)
    return (
        "jax_flat groupby capacity exceeded" in message
        or "jax_flat groupby hash table exhausted" in message
    )


def _double_groupby_capacities(runtime: JaxFlatRuntime) -> JaxFlatRuntime:
    changed = False
    nodes = []
    for node in runtime.program.nodes:
        op = node.op
        if isinstance(op, GroupByOp):
            op = replace(
                op,
                capacity=op.capacity * 2,
                hash_capacity=max(op.hash_capacity * 2, op.capacity * 4),
            )
            changed = True
        nodes.append(DagNode(op=op, child_ids=node.child_ids))
    if not changed:
        return runtime
    return JaxFlatRuntime(program=replace(runtime.program, nodes=tuple(nodes)))


@jax.jit
def _jit_batch_from_initial_state(runtime: JaxFlatRuntime, inputs):
    state0 = runtime.init_state(inputs[0].shape[1])
    return _scan_batch(runtime, state0, inputs)


@jax.jit
def _jit_batch(runtime: JaxFlatRuntime, state0, inputs):
    return _scan_batch(runtime, state0, inputs)


_BATCH_CHUNK_SIZE = int(os.environ.get("TRADING_DSL_JAX_FLAT_BATCH_CHUNK_SIZE", "65536"))
_CPP_ACCELERATOR_CACHE: dict[tuple[Any, ...], Any] = {}


@jax.jit
def _scan_batch(runtime: JaxFlatRuntime, state0, inputs):
    n_steps = inputs[0].shape[0]
    chunk_size = min(n_steps, _BATCH_CHUNK_SIZE)
    n_full_chunks = n_steps // chunk_size
    remainder = n_steps - n_full_chunks * chunk_size

    def scan_chunk(states, start, size: int):
        chunk_inputs = tuple(
            jax.lax.dynamic_slice_in_dim(arr, start, size, axis=0)
            for arr in inputs
        )
        return _scan_batch_chunk(runtime, states, chunk_inputs, start)

    def set_chunk(out, start, value):
        return jax.tree_util.tree_map(
            lambda dst, src: jax.lax.dynamic_update_slice(
                dst,
                src,
                (start,) + (0,) * (jnp.asarray(dst).ndim - 1),
            ),
            out,
            value,
        )

    states, chunk0_out, chunk0_cache = scan_chunk(state0, 0, chunk_size)

    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.empty((n_steps,) + leaf.shape[1:], dtype=leaf.dtype)

    out0 = set_chunk(jax.tree_util.tree_map(alloc, chunk0_out), 0, chunk0_out)
    cache0 = set_chunk(jax.tree_util.tree_map(alloc, chunk0_cache), 0, chunk0_cache)

    def body(chunk_i, carry):
        states_c, out_c, cache_c = carry
        start = chunk_i * chunk_size
        states_n, chunk_out, chunk_cache = scan_chunk(states_c, start, chunk_size)
        return states_n, set_chunk(out_c, start, chunk_out), set_chunk(cache_c, start, chunk_cache)

    states, out, cache_out = jax.lax.fori_loop(
        1,
        n_full_chunks,
        body,
        (states, out0, cache0),
    )

    if remainder:
        start = n_full_chunks * chunk_size
        states, tail_out, tail_cache = scan_chunk(states, start, remainder)
        out = set_chunk(out, start, tail_out)
        cache_out = set_chunk(cache_out, start, tail_cache)

    return states, out, cache_out


@jax.jit
def _scan_batch_chunk(runtime: JaxFlatRuntime, state_leaves, inputs, batch_start: int = 0):
    n_steps = inputs[0].shape[0]
    values: list[Any] = [jnp.array(0.0)] * len(runtime.program.nodes)
    new_state = list(state_leaves)

    for idx, node in enumerate(runtime.program.nodes):
        op = node.op
        if isinstance(op, InputOp):
            values[idx] = inputs[op.input_index]
            continue
        if isinstance(op, LiteralOp):
            values[idx] = jnp.full((n_steps,), op.value, dtype=jnp.float64)
            continue

        child_values = tuple(values[cid] for cid in node.child_ids)
        field = runtime.program.state_layout.node_fields[idx]
        node_state = None if field.index < 0 else state_leaves[field.index]
        next_state, value = (
            op.scan_batch_with_start(node_state, jnp.asarray(batch_start, dtype=jnp.int64), *child_values)
            if isinstance(op, CacheOp)
            else op.scan_batch(node_state, *child_values)
        )
        if field.index >= 0:
            new_state[field.index] = next_state
        values[idx] = value

    outs = tuple(values[i] for i in runtime.program.outputs)
    cache_outs = tuple(values[i] for i in runtime.program.cache_nodes)
    return tuple(new_state), outs[0] if len(outs) == 1 else jnp.stack(outs, axis=0), cache_outs
