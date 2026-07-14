from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from functools import partial
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.base.parser import Expr, parse_formula
from trading_dsl_engine.jax_flat.engine import (
    DagNode,
    JaxFlatRuntime,
    StreamingProgram,
    _build_state_layout,
    _compile_node,
    _double_groupby_capacities,
    _expand_dsl,
    _external_cache_inputs,
    _is_groupby_capacity_error,
    _normalize_batch_inputs,
    _normalize_static_jax_flat_kwargs,
)
from trading_dsl_engine.jax_flat.ops import (
    CacheOp,
    CumsumOp,
    EwmOp,
    EwmState,
    InputOp,
    LiteralOp,
    Op,
)

_DEFAULT_CHUNK_SIZE = int(os.environ.get("TRADING_DSL_JAX_FLAT_BATCH_CHUNK_SIZE", "65536"))
_DEFAULT_MAX_IN_FLIGHT = int(os.environ.get("TRADING_DSL_JAX_FLAT_MAX_IN_FLIGHT", "2"))


def _invalid_like(value: jax.Array) -> jax.Array:
    value = jnp.asarray(value)
    if jnp.issubdtype(value.dtype, jnp.inexact):
        return jnp.full_like(value, jnp.nan)
    if jnp.issubdtype(value.dtype, jnp.bool_):
        return jnp.zeros_like(value, dtype=bool)
    return jnp.zeros_like(value)


def _value_template(op: Op, n_instruments: int) -> jax.Array:
    if op.output_kind == "scalar":
        return jnp.asarray(0.0, dtype=jnp.float64)
    if op.output_kind == "vector":
        return jnp.zeros((n_instruments,), dtype=jnp.float64)
    if op.output_kind == "matrix" and op.output_width is not None:
        return jnp.zeros((n_instruments, int(op.output_width)), dtype=jnp.float64)
    raise ValueError(f"Optimized batch roots must be scalar/vector/matrix, got {op.output_kind!r}")


@dataclass(frozen=True)
class AssociativeEwmOp(EwmOp):
    """Parallel batch lowering for the affine EWM mode.

    This preserves the existing tick semantics for static-span,
    adjust=False, ignore_na=True EWM, including first-observation
    initialization and NaN rows leaving state unchanged.
    """

    @property
    def batch_parallel(self) -> bool:
        return self.span is not None and self.ignore_na and not self.adjust

    def scan_batch(self, state: EwmState, *child_sequences: jax.Array):
        if not self.batch_parallel:
            return super().scan_batch(state, *child_sequences)

        x = jnp.asarray(child_sequences[0])
        alpha = jnp.asarray(2.0 / (float(self.span) + 1.0), dtype=x.dtype)
        decay = jnp.asarray(1.0, dtype=x.dtype) - alpha
        valid = jnp.isfinite(x)

        # A segment summary is a transform on an incoming initialized value:
        #   y_out = a * y_in + b
        # and u is the segment result if the incoming state was uninitialized.
        has = valid
        a = jnp.where(valid, decay, jnp.asarray(1.0, dtype=x.dtype))
        b = jnp.where(valid, alpha * x, jnp.asarray(0.0, dtype=x.dtype))
        u = jnp.where(valid, x, jnp.asarray(0.0, dtype=x.dtype))
        count = valid.astype(jnp.int64)

        def combine(left, right):
            h1, a1, b1, u1, c1 = left
            h2, a2, b2, u2, c2 = right
            return (
                h1 | h2,
                a2 * a1,
                a2 * b1 + b2,
                jnp.where(h1, a2 * u1 + b2, u2),
                c1 + c2,
            )

        has_p, a_p, b_p, u_p, count_p = jax.lax.associative_scan(
            combine,
            (has, a, b, u, count),
            axis=0,
        )

        initialized0 = jnp.asarray(state.initialized)
        value_from_initialized = a_p * state.value + b_p
        value_from_uninitialized = jnp.where(has_p, u_p, state.value)
        values = jnp.where(initialized0, value_from_initialized, value_from_uninitialized)
        initialized = initialized0 | has_p
        counts = state.count + count_p

        if self.min_periods is None:
            enough = jnp.ones_like(initialized, dtype=bool)
        else:
            enough = counts >= int(round(float(self.min_periods)))
        out = jnp.where(initialized & enough, values, jnp.nan)

        final_initialized = initialized[-1]
        final_value = values[-1]
        final_count = counts[-1]
        final_weight = jnp.where(final_initialized, jnp.ones_like(state.weight), state.weight)
        return (
            EwmState(
                value=final_value,
                weight=final_weight,
                initialized=final_initialized,
                count=final_count,
            ),
            out,
        )


def _replace_parallel_ops(program: StreamingProgram) -> StreamingProgram:
    changed = False
    nodes: list[DagNode] = []
    for node in program.nodes:
        op = node.op
        if isinstance(op, EwmOp) and not isinstance(op, AssociativeEwmOp):
            candidate = AssociativeEwmOp(
                span=op.span,
                min_periods=op.min_periods,
                ignore_na=op.ignore_na,
                adjust=op.adjust,
                output_kind=op.output_kind,
                output_width=op.output_width,
            )
            if candidate.batch_parallel:
                op = candidate
                changed = True
        nodes.append(DagNode(op=op, child_ids=node.child_ids))
    return replace(program, nodes=tuple(nodes)) if changed else program


def _stateful_depth(program: StreamingProgram) -> int:
    depths: list[int] = []
    for node in program.nodes:
        parent_depth = max((depths[cid] for cid in node.child_ids), default=0)
        depths.append(parent_depth + int(bool(node.op.is_stateful)))
    return max((depths[i] for i in program.outputs), default=0)


def _pad_safe(program: StreamingProgram) -> bool:
    for node in program.nodes:
        op = node.op
        if not op.is_stateful:
            continue
        if isinstance(op, AssociativeEwmOp) and op.batch_parallel:
            continue
        if isinstance(op, CumsumOp):
            continue
        return False
    return True


def _choose_strategy(program: StreamingProgram, requested: str) -> str:
    if requested not in {"auto", "compound", "node_batch"}:
        raise ValueError("strategy must be 'auto', 'compound', or 'node_batch'")
    if requested != "auto":
        return requested
    if _stateful_depth(program) <= 1 and _pad_safe(program):
        return "node_batch"
    return "compound"


def _tick_program(runtime: JaxFlatRuntime, state_leaves, input_rows):
    values: list[Any] = [jnp.asarray(0.0)] * len(runtime.program.nodes)
    new_state = list(state_leaves)

    for idx, node in enumerate(runtime.program.nodes):
        op = node.op
        if isinstance(op, InputOp):
            values[idx] = input_rows[op.input_index]
            continue
        if isinstance(op, LiteralOp):
            values[idx] = jnp.asarray(op.value, dtype=jnp.float64)
            continue

        child_values = tuple(values[cid] for cid in node.child_ids)
        field = runtime.program.state_layout.node_fields[idx]
        node_state = None if field.index < 0 else state_leaves[field.index]
        next_state, value = op.tick(node_state, *child_values)
        if field.index >= 0:
            new_state[field.index] = next_state
        values[idx] = value

    outputs = tuple(values[i] for i in runtime.program.outputs)
    cache_outputs = tuple(values[i] for i in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


@partial(jax.jit, donate_argnums=(1,))
def _jit_tick_program(runtime: JaxFlatRuntime, state_leaves, input_rows):
    return _tick_program(runtime, state_leaves, input_rows)


@partial(jax.jit, donate_argnums=(1,))
def _compound_scan_chunk(
    runtime: JaxFlatRuntime,
    state_leaves,
    inputs,
    valid_length: jax.Array,
    invalid_outputs,
    invalid_cache_outputs,
):
    n_steps = inputs[0].shape[0]
    row_indices = jnp.arange(n_steps, dtype=jnp.int32)

    def step(states, xs):
        rows = xs[:-1]
        row_idx = xs[-1]

        def active(_):
            return _tick_program(runtime, states, rows)

        def inactive(_):
            return states, (invalid_outputs, invalid_cache_outputs)

        return jax.lax.cond(row_idx < valid_length, active, inactive, operand=None)

    return jax.lax.scan(step, state_leaves, (*inputs, row_indices), unroll=1)


@partial(jax.jit, donate_argnums=(1,))
def _node_batch_chunk(runtime: JaxFlatRuntime, state_leaves, inputs, batch_start: jax.Array):
    n_steps = inputs[0].shape[0]
    values: list[Any] = [jnp.asarray(0.0)] * len(runtime.program.nodes)
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
            op.scan_batch_with_start(node_state, batch_start, *child_values)
            if isinstance(op, CacheOp)
            else op.scan_batch(node_state, *child_values)
        )
        if field.index >= 0:
            new_state[field.index] = next_state
        values[idx] = value

    outputs = tuple(values[i] for i in runtime.program.outputs)
    cache_outputs = tuple(values[i] for i in runtime.program.cache_nodes)
    return tuple(new_state), (outputs, cache_outputs)


def _pad_chunk(array, start: int, stop: int, chunk_size: int) -> np.ndarray:
    src = np.asarray(array[start:stop], dtype=np.float64)
    if src.shape[0] == chunk_size:
        return src
    dst = np.full((chunk_size,) + src.shape[1:], np.nan, dtype=np.float64)
    dst[: src.shape[0]] = src
    return dst


def _output_path(base: str, name: str | None, multiple: bool) -> str:
    if not multiple:
        return base
    path = Path(base)
    suffix = path.suffix or ".memmap"
    stem = path.name[: -len(suffix)] if suffix else path.name
    return str(path.with_name(f"{stem}.{name}{suffix}"))


def _allocate_output(template, n_steps: int, out_path, name: str | None, multiple: bool):
    shape = (n_steps,) + tuple(np.asarray(template).shape)
    if out_path is False or out_path is None:
        return np.empty(shape, dtype=np.asarray(template).dtype)
    if out_path is True:
        import tempfile

        fd, path = tempfile.mkstemp(prefix=f"trading_dsl_engine_{name or 'out'}_", suffix=".memmap")
        os.close(fd)
    elif isinstance(out_path, str):
        path = _output_path(out_path, name, multiple)
    else:
        raise ValueError("out_path must be False, True, None, or a filesystem path")
    return np.memmap(path, mode="w+", dtype=np.asarray(template).dtype, shape=shape)


def _flush(array) -> None:
    if isinstance(array, np.memmap):
        array.flush()


def _materialize_pending(pending, output_arrays, cache_arrays) -> None:
    start, valid_length, output_values, cache_values = pending
    output_np = tuple(np.asarray(jax.device_get(value))[:valid_length] for value in output_values)
    cache_np = tuple(np.asarray(jax.device_get(value))[:valid_length] for value in cache_values)
    stop = start + valid_length
    for target, value in zip(output_arrays, output_np):
        target[start:stop] = value
        _flush(target)
    for target, value in zip(cache_arrays, cache_np):
        target[start:stop] = value
        _flush(target)


@dataclass
class OptimizedJaxFlatRuntime:
    runtime: JaxFlatRuntime
    output_names: tuple[str, ...]
    strategy: str = "auto"
    chunk_size: int = _DEFAULT_CHUNK_SIZE
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT
    block: bool = True

    @property
    def program(self) -> StreamingProgram:
        return self.runtime.program

    def init_state(self, n_instruments: int):
        return self.runtime.init_state(n_instruments)

    def tick(self, state_leaves, *input_rows):
        next_state, (outputs, _) = _jit_tick_program(self.runtime, state_leaves, input_rows)
        return next_state, outputs[0] if len(outputs) == 1 else dict(zip(self.output_names, outputs, strict=True))

    def execution_strategy(self) -> str:
        return _choose_strategy(self.program, self.strategy)

    def run_batch(self, inputs, states=None, out_path: str | bool | None = False):
        runtime = self.runtime
        original_states = states
        while True:
            try:
                return self._run_batch_once(runtime, inputs, states, out_path)
            except Exception as exc:
                if original_states is not None or not _is_groupby_capacity_error(exc):
                    raise
                next_runtime = _double_groupby_capacities(runtime)
                if next_runtime is runtime:
                    raise
                runtime = next_runtime
                states = None

    def _run_batch_once(self, runtime: JaxFlatRuntime, inputs, states, out_path):
        inputs = _normalize_batch_inputs(runtime, inputs)
        if not inputs:
            raise ValueError("run_batch requires at least one input array")
        n_steps, n_instruments = inputs[0].shape[:2]
        if any(arr.shape[:2] != (n_steps, n_instruments) for arr in inputs[1:]):
            raise ValueError("All inputs must share aligned shape (time, n_instruments)")

        state = runtime.init_state(n_instruments) if states is None else states
        chunk_size = max(1, min(int(self.chunk_size), max(1, n_steps)))
        output_templates = tuple(_value_template(runtime.program.nodes[i].op, n_instruments) for i in runtime.program.outputs)
        cache_templates = tuple(_value_template(runtime.program.nodes[i].op, n_instruments) for i in runtime.program.cache_nodes)
        invalid_outputs = tuple(_invalid_like(value) for value in output_templates)
        invalid_caches = tuple(_invalid_like(value) for value in cache_templates)

        multiple = len(output_templates) > 1
        output_arrays = tuple(
            _allocate_output(template, n_steps, out_path, name, multiple)
            for name, template in zip(self.output_names, output_templates, strict=True)
        )
        cache_arrays = tuple(np.empty((n_steps,) + tuple(np.asarray(template).shape), dtype=np.asarray(template).dtype) for template in cache_templates)

        strategy = _choose_strategy(runtime.program, self.strategy)
        pending = deque()
        max_in_flight = max(1, int(self.max_in_flight))

        for start in range(0, n_steps, chunk_size):
            stop = min(start + chunk_size, n_steps)
            valid_length = stop - start
            chunk_inputs = tuple(jnp.asarray(_pad_chunk(arr, start, stop, chunk_size)) for arr in inputs)

            if strategy == "node_batch":
                state, (chunk_outputs, chunk_caches) = _node_batch_chunk(
                    runtime,
                    state,
                    chunk_inputs,
                    jnp.asarray(start, dtype=jnp.int64),
                )
            else:
                state, (chunk_outputs, chunk_caches) = _compound_scan_chunk(
                    runtime,
                    state,
                    chunk_inputs,
                    jnp.asarray(valid_length, dtype=jnp.int32),
                    invalid_outputs,
                    invalid_caches,
                )

            pending.append((start, valid_length, chunk_outputs, chunk_caches))
            if len(pending) >= max_in_flight:
                _materialize_pending(pending.popleft(), output_arrays, cache_arrays)

        while pending:
            _materialize_pending(pending.popleft(), output_arrays, cache_arrays)

        if self.block:
            jax.block_until_ready(state)
        runtime.cached_values.clear()
        for node_id, value in zip(runtime.program.cache_nodes, cache_arrays, strict=True):
            runtime.cached_values[int(node_id)] = value

        output = output_arrays[0] if len(output_arrays) == 1 else dict(zip(self.output_names, output_arrays, strict=True))
        return state, output


def _compile_expressions(
    expressions: Sequence[tuple[str, str | Expr]],
    *,
    dsl_registry: DSLFunctionRegistry | None,
    runtimes,
) -> tuple[JaxFlatRuntime, tuple[str, ...]]:
    external_cache_names, external_cache_values = _external_cache_inputs(runtimes)
    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    input_names: list[str] = []
    outputs: list[int] = []
    names: list[str] = []

    for name, formula in expressions:
        expr = parse_formula(formula) if isinstance(formula, str) else formula
        expr = _normalize_static_jax_flat_kwargs(expr)
        expr = _expand_dsl(expr, dsl_registry or DEFAULT_DSL_REGISTRY)
        expr = _normalize_static_jax_flat_kwargs(expr)
        outputs.append(_compile_node(expr, memo, nodes, input_names, external_cache_names))
        names.append(str(name))

    node_tuple = tuple(nodes)
    cache_nodes = tuple(idx for idx, node in enumerate(node_tuple) if isinstance(node.op, CacheOp))
    program = StreamingProgram(
        nodes=node_tuple,
        outputs=tuple(outputs),
        input_names=tuple(input_names),
        state_layout=_build_state_layout(node_tuple),
        metadata=None,
        cache_nodes=cache_nodes,
        cache_expr_keys=(),
        external_cache_inputs=external_cache_values or None,
    )
    program = _replace_parallel_ops(program)
    return JaxFlatRuntime(program=program, cpp=False), tuple(names)


def compile_formula(
    formula: str | Expr,
    *,
    dsl_registry: DSLFunctionRegistry | None = None,
    runtimes=None,
    strategy: str = "auto",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
) -> OptimizedJaxFlatRuntime:
    runtime, names = _compile_expressions(
        (("output", formula),),
        dsl_registry=dsl_registry,
        runtimes=runtimes,
    )
    return OptimizedJaxFlatRuntime(
        runtime=runtime,
        output_names=names,
        strategy=strategy,
        chunk_size=chunk_size,
        max_in_flight=max_in_flight,
    )


def compile_features(
    formulas: Mapping[str, str | Expr],
    *,
    dsl_registry: DSLFunctionRegistry | None = None,
    runtimes=None,
    strategy: str = "auto",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    max_in_flight: int = _DEFAULT_MAX_IN_FLIGHT,
) -> OptimizedJaxFlatRuntime:
    if not formulas:
        raise ValueError("compile_features requires at least one named formula")
    runtime, names = _compile_expressions(
        tuple(formulas.items()),
        dsl_registry=dsl_registry,
        runtimes=runtimes,
    )
    return OptimizedJaxFlatRuntime(
        runtime=runtime,
        output_names=names,
        strategy=strategy,
        chunk_size=chunk_size,
        max_in_flight=max_in_flight,
    )


__all__ = [
    "AssociativeEwmOp",
    "OptimizedJaxFlatRuntime",
    "compile_formula",
    "compile_features",
]
