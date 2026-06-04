from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any
import mmap
import os
import time
import tempfile
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, String, Universe, parse_formula
from trading_dsl_engine.jax_flat.custom import StatelessJaxCall
from trading_dsl_engine.jax_flat.ops import (
    BufferShiftOp,
    EwmOp,
    FFillOp,
    GroupByOp,
    InputOp,
    LiteralOp,
    NaryOp,
    ANY_ARITY,
    OP_FACTORIES,
    Op,
    RidgeOp,
    ShiftOp,
    _bspline,
    _cat,
    _col,
)




@dataclass(frozen=True)
class DagNode:
    op: Any
    child_ids: tuple[int, ...]


@dataclass(frozen=True)
class StateFieldRef:
    index: int


@dataclass(frozen=True)
class StateLayout:
    node_fields: tuple[StateFieldRef, ...]
    total_leaves: int


@dataclass(frozen=True)
class StreamingProgram:
    nodes: tuple[DagNode, ...]
    outputs: tuple[int, ...]
    input_names: tuple[str, ...]
    state_layout: StateLayout


@dataclass(frozen=True)
class InnerGraphOp(Op):
    """A groupby-local operator DAG.

    The flat runtime normally stores state per top-level DAG node. A groupby RHS,
    however, must keep any nested stateful operators inside the group bucket. This
    op wraps the RHS sub-DAG so GroupByOp can allocate one complete RHS state per
    universe/dynamic-key slot.
    """

    nodes: tuple[DagNode, ...]
    output_id: int
    state_layout: StateLayout
    n_inputs: int
    is_stateful: bool = False
    output_kind: str = "vector"

    def init_state(self, sample: jax.Array):
        states = []
        for node in self.nodes:
            if node.op.is_stateful:
                states.append(node.op.init_state(sample))
        return tuple(states)

    def tick(self, state_leaves, *input_values: jax.Array):
        values: list[jax.Array] = [jnp.array(0.0)] * len(self.nodes)
        new_state = list(()) if state_leaves is None else list(state_leaves)

        for idx, node in enumerate(self.nodes):
            op = node.op
            if isinstance(op, InputOp):
                values[idx] = input_values[op.input_index]
                continue
            if isinstance(op, LiteralOp):
                values[idx] = jnp.asarray(op.value, dtype=jnp.float64)
                continue

            child_values = tuple(values[cid] for cid in node.child_ids)
            field = self.state_layout.node_fields[idx]
            node_state = None if field.index < 0 else state_leaves[field.index]
            next_state, value = op.tick(node_state, *child_values)
            if field.index >= 0:
                new_state[field.index] = next_state
            values[idx] = value

        return tuple(new_state), values[self.output_id]


class JaxFlatRuntime(eqx.Module):
    program: StreamingProgram = eqx.field(static=True)
    runtimes: list[float] = eqx.field(default_factory=list, static=True)
    block: bool = True # False disables waiting (for runtime measurement)

    def init_state(self, n_instruments: int):
        vector_sample = jnp.zeros((n_instruments,), dtype=jnp.float64)
        values: list[Any] = [vector_sample] * len(self.program.nodes)
        states = []
        for idx, node in enumerate(self.program.nodes):
            op = node.op
            if isinstance(op, InputOp):
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
        if _has_memmap_input(inputs) or out_path:
            return _run_chunked_batch(self, inputs, states, out_path)
        if not states:
            return _jit_batch_from_initial_state(self, inputs)
        return _jit_batch(self, states, inputs)


def _normalize_batch_inputs(runtime: JaxFlatRuntime, inputs):
    if isinstance(inputs, dict):
        missing = [name for name in runtime.program.input_names if name not in inputs]
        if missing:
            raise ValueError(f"Missing jax_flat run_batch input(s): {missing}")
        if len(runtime.program.input_names) == 0:
            inputs = tuple(inputs.values())
        else:
            inputs = tuple(inputs[name] for name in runtime.program.input_names)
    else:
        inputs = tuple(inputs)

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
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input for '{name}', got shape {arr.shape}")
    return inputs


def _has_memmap_input(inputs) -> bool:
    return any(isinstance(arr, np.memmap) for arr in inputs)


def _fresh_memmap_path(prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".memmap")
    os.close(fd)
    return path


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
    out = None

    for start in range(0, n_steps, chunk_size):
        stop = min(start + chunk_size, n_steps)
        chunk_inputs = tuple(jnp.asarray(_input_chunk(arr, start, stop)) for arr in inputs)
        states, chunk_out = _scan_batch_chunk(runtime, states, chunk_inputs)
        chunk_out_np = np.asarray(jax.block_until_ready(chunk_out))
        if out is None:
            out = _output_array(out_path, n_steps, chunk_out_np)
        out[start:stop] = chunk_out_np
        _flush_memmap_output(out)

    return states, out


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


_BATCH_CHUNK_SIZE = 2560


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
        return _scan_batch_chunk(runtime, states, chunk_inputs)

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

    states, chunk0_out = scan_chunk(state0, 0, chunk_size)

    def alloc(leaf):
        leaf = jnp.asarray(leaf)
        return jnp.empty((n_steps,) + leaf.shape[1:], dtype=leaf.dtype)

    out0 = set_chunk(jax.tree_util.tree_map(alloc, chunk0_out), 0, chunk0_out)

    def body(chunk_i, carry):
        states_c, out_c = carry
        start = chunk_i * chunk_size
        states_n, chunk_out = scan_chunk(states_c, start, chunk_size)
        return states_n, set_chunk(out_c, start, chunk_out)

    states, out = jax.lax.fori_loop(
        1,
        n_full_chunks,
        body,
        (states, out0),
    )

    if remainder:
        start = n_full_chunks * chunk_size
        states, tail_out = scan_chunk(states, start, remainder)
        out = set_chunk(out, start, tail_out)

    return states, out


@jax.jit
def _scan_batch_chunk(runtime: JaxFlatRuntime, state_leaves, inputs):
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
        next_state, value = op.scan_batch(node_state, *child_values)
        if field.index >= 0:
            new_state[field.index] = next_state
        values[idx] = value

    outs = tuple(values[i] for i in runtime.program.outputs)
    return tuple(new_state), outs[0] if len(outs) == 1 else jnp.stack(outs, axis=0)



def _normalize_static_jax_flat_kwargs(node: Expr) -> Expr:
    if isinstance(node, Call):
        args = tuple(_normalize_static_jax_flat_kwargs(arg) for arg in node.args)
        kwargs = tuple((key, _normalize_static_jax_flat_kwargs(value)) for key, value in node.kwargs)
        if not kwargs:
            return Call(node.fn, args)
        kw = dict(kwargs)
        if node.fn == "shift":
            unsupported = set(kw) - {"lag", "nlag", "max_lag", "max_size"}
            if unsupported:
                raise ValueError(f"Unsupported shift keyword argument(s): {sorted(unsupported)}")
            lag = kw.pop("lag") if "lag" in kw else kw.pop("nlag", None)
            max_lag = kw.pop("max_lag") if "max_lag" in kw else kw.pop("max_size", None)
            if len(args) > 1 or (len(args) == 1 and lag is None):
                raise ValueError("shift keyword form expects shift(x, lag=..., max_lag=...)")
            if len(args) != 1 or lag is None or max_lag is None:
                raise ValueError("shift keyword form requires x, lag, and max_lag")
            return Call("shift", (args[0], lag, max_lag))
        if node.fn == "buffer":
            unsupported = set(kw) - {"min", "max"}
            if unsupported:
                raise ValueError(f"Unsupported buffer keyword argument(s): {sorted(unsupported)}")
            min_lag = kw.pop("min", None)
            max_lag = kw.pop("max", None)
            if len(args) > 1 or (len(args) == 1 and min_lag is None):
                raise ValueError("buffer keyword form expects buffer(shift_expr, min=..., max=...)")
            if len(args) != 1 or min_lag is None or max_lag is None:
                raise ValueError("buffer keyword form requires shift_expr, min, and max")
            return Call("buffer", (args[0], min_lag, max_lag))
        return Call(node.fn, args, kwargs)
    if isinstance(node, StatelessJaxCall):
        return StatelessJaxCall(
            fn=node.fn,
            args=tuple(_normalize_static_jax_flat_kwargs(arg) for arg in node.args),
            output_kind=node.output_kind,
            output_width=node.output_width,
            name=node.name,
        )
    if isinstance(node, KeyTuple):
        return KeyTuple(tuple(_normalize_static_jax_flat_kwargs(item) for item in node.items))
    return node

def _expr_key(node: Expr):
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", float(node.value))
    if isinstance(node, String):
        return ("str", node.value)
    if isinstance(node, Call):
        return (
            "call",
            node.fn,
            tuple(_expr_key(a) for a in node.args),
            tuple((k, _expr_key(v)) for k, v in node.kwargs),
        )
    if isinstance(node, StatelessJaxCall):
        return (
            "jax_stateless",
            id(node.fn),
            node.output_kind,
            node.output_width,
            node.name,
            tuple(_expr_key(a) for a in node.args),
        )
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_expr_key(item) for item in node.items))
    if isinstance(node, Universe):
        return ("univ", node.groups)
    raise ValueError(f"Unsupported expression: {node}")


def _canonical_groupby_key_items(key: Expr) -> tuple[Expr, ...]:
    if not isinstance(key, KeyTuple):
        key = KeyTuple((key,))
    if sum(1 for item in key.items if isinstance(item, Universe)) > 1:
        raise ValueError("groupby key tuple may contain at most one univ(...) element")
    return key.items


def _resolve_universe_groups(universe: Universe) -> tuple[tuple[int, ...], ...]:
    groups = []
    for g in universe.groups:
        cols = []
        for m in g:
            if not isinstance(m, int):
                raise ValueError("jax_flat univ currently supports integer column indexes only")
            cols.append(m)
        groups.append(tuple(cols))
    return tuple(groups)


def _validate_groupby_canonical_form(expr: Call) -> None:
    if len(expr.args) != 3:
        raise ValueError(
            "groupby only supports canonical form: "
            "groupby((key1, ..., maybe_univ, ...), lhs, op_using_self_)"
        )
    unsupported_kwargs = {name for name, _ in expr.kwargs} - {"capacity", "hash_capacity"}
    if unsupported_kwargs:
        raise ValueError(f"Unsupported groupby keyword argument(s): {sorted(unsupported_kwargs)}")
    for name, value in expr.kwargs:
        if not isinstance(value, Number):
            raise ValueError(f"groupby {name} must be a numeric literal")
    _canonical_groupby_key_items(expr.args[0])


def _replace_self(node: Expr, lhs: Expr) -> Expr:
    if isinstance(node, Identifier) and node.name == "self_":
        return lhs
    if isinstance(node, Call):
        return Call(
            node.fn,
            tuple(_replace_self(a, lhs) for a in node.args),
            tuple((key, _replace_self(value, lhs)) for key, value in node.kwargs),
        )
    if isinstance(node, StatelessJaxCall):
        return StatelessJaxCall(
            fn=node.fn,
            args=tuple(_replace_self(a, lhs) for a in node.args),
            output_kind=node.output_kind,
            output_width=node.output_width,
            name=node.name,
        )
    if isinstance(node, KeyTuple):
        return KeyTuple(tuple(_replace_self(a, lhs) for a in node.items))
    return node


def _literal_int_arg(expr: Expr, fn: str, position: int) -> int:
    if not isinstance(expr, Number):
        raise ValueError(f"{fn} argument {position} must be a numeric literal")
    return int(round(float(expr.value)))


def _op_width(op: Op) -> int:
    if op.output_width is None:
        raise ValueError(f"Cannot infer static output width for {op.output_kind} op in jax_flat")
    return int(op.output_width)



def _build_stateless_jax_op(expr: StatelessJaxCall, child_ops: tuple[Op, ...]) -> Op:
    if not child_ops:
        raise ValueError("stateless JAX call expects at least one child")
    output_kind = expr.output_kind if expr.output_kind is not None else child_ops[0].output_kind
    output_width = expr.output_width if expr.output_width is not None else child_ops[0].output_width
    return NaryOp(expr.fn, output_kind=output_kind, output_width=output_width)


def _build_op(expr: Call, child_ops: tuple[Op, ...] = ()) -> tuple[Op, int | None]:
    if expr.fn == "groupby":
        raise ValueError("groupby must be compiled with its RHS subgraph")
    if expr.kwargs:
        raise ValueError(f"Keyword arguments are not supported for {expr.fn}")
    if expr.fn == "cat" and len(expr.args) >= 1:
        return NaryOp(_cat, output_kind="matrix", output_width=sum(_op_width(op) for op in child_ops), cpp_name="cat"), None
    variadic_builder = OP_FACTORIES.get((expr.fn, ANY_ARITY))
    if variadic_builder is not None:
        static_args = tuple(arg.value for arg in expr.args if isinstance(arg, String))
        return variadic_builder(*static_args), None
    if expr.fn == "round":
        if len(expr.args) == 1:
            return NaryOp(jnp.round, cpp_name="round"), None
        if len(expr.args) == 2 and isinstance(expr.args[1], Number):
            decimals = int(round(float(expr.args[1].value)))
            return NaryOp(lambda x, decimals=decimals: jnp.round(x, decimals=decimals)), 1
    if expr.fn == "ewm" and len(expr.args) == 2:
        if isinstance(expr.args[1], Number):
            return EwmOp(span=float(expr.args[1].value)), 1
        return EwmOp(), None
    if expr.fn == "ffill" and len(expr.args) in (1, 2):
        limit = None
        dynamic_limit = False
        drop_child_idx = None
        if len(expr.args) == 2:
            if isinstance(expr.args[1], Number):
                limit = int(round(float(expr.args[1].value)))
                if limit < 0:
                    raise ValueError("ffill limit must be >= 0")
                drop_child_idx = 1
            else:
                dynamic_limit = True
        return (
            FFillOp(
                limit=limit,
                dynamic_limit=dynamic_limit,
                output_kind=child_ops[0].output_kind,
                output_width=child_ops[0].output_width,
            ),
            drop_child_idx,
        )
    if expr.fn == "shift" and len(expr.args) in (2, 3):
        max_size_arg = expr.args[2] if len(expr.args) == 3 else expr.args[1]
        max_size = max(0, _literal_int_arg(max_size_arg, "shift", 3 if len(expr.args) == 3 else 2))
        return (
            ShiftOp(
                max_size=max_size,
                output_kind=child_ops[0].output_kind,
                output_width=child_ops[0].output_width,
            ),
            2 if len(expr.args) == 3 else None,
        )
    if expr.fn == "bspline" and len(expr.args) == 2:
        n_basis = _literal_int_arg(expr.args[1], "bspline", 2)
        if n_basis <= 0:
            raise ValueError("bspline n_basis must be >= 1")
        return NaryOp(lambda x, n_basis=n_basis: _bspline(x, n_basis), output_kind="matrix", output_width=n_basis, cpp_name="bspline", cpp_int_param=n_basis), 1
    if expr.fn == "col" and len(expr.args) == 2:
        index = _literal_int_arg(expr.args[1], "col", 2)
        if index < 0:
            raise ValueError("col index must be >= 0")
        return NaryOp(lambda x, index=index: _col(x, index), output_kind="vector", output_width=1, cpp_name="col", cpp_int_param=index), 1
    if expr.fn == "Ridge" and len(expr.args) >= 4:
        has_weights = len(expr.args) >= 5
        feature_ops = child_ops[:-4] if has_weights else child_ops[:-3]
        if not feature_ops:
            raise ValueError("Ridge expects at least one feature arg")
        feature_widths = tuple(_op_width(op) for op in feature_ops)
        return RidgeOp(feature_widths=feature_widths, has_weights=has_weights), None
    builder = OP_FACTORIES.get((expr.fn, len(expr.args)))
    if builder is None:
        raise ValueError(f"Unsupported function {expr.fn}/{len(expr.args)} in jax_flat")
    return builder(), None


def _build_state_layout(nodes: tuple[DagNode, ...], sample: jax.Array | None = None) -> StateLayout:
    del sample
    refs = []
    offset = 0
    for node in nodes:
        if node.op.is_stateful:
            refs.append(StateFieldRef(index=offset))
            offset += 1
        else:
            refs.append(StateFieldRef(index=-1))
    return StateLayout(node_fields=tuple(refs), total_leaves=offset)


def _compile_groupby_inner_op(rhs: Expr, lhs: Expr) -> tuple[InnerGraphOp, tuple[Expr, ...]]:
    if not isinstance(rhs, Call):
        raise ValueError("groupby rhs must be a call expression")

    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    feed_exprs: list[Expr] = []
    feed_index_by_key: dict[tuple[Any, ...], int] = {}

    def feed_node(feed_expr: Expr) -> int:
        feed_key = _expr_key(feed_expr)
        input_index = feed_index_by_key.get(feed_key)
        if input_index is None:
            input_index = len(feed_exprs)
            feed_index_by_key[feed_key] = input_index
            feed_exprs.append(feed_expr)

        node_key = ("feed", feed_key)
        cached = memo.get(node_key)
        if cached is not None:
            return cached
        idx = len(nodes)
        nodes.append(DagNode(op=InputOp(input_index=input_index), child_ids=()))
        memo[node_key] = idx
        return idx

    def build(node: Expr) -> int:
        if isinstance(node, Identifier):
            return feed_node(lhs if node.name == "self_" else node)

        key = ("expr", _expr_key(_replace_self(node, lhs)))
        cached = memo.get(key)
        if cached is not None:
            return cached

        if isinstance(node, Number):
            idx = len(nodes)
            nodes.append(DagNode(op=LiteralOp(float(node.value)), child_ids=()))
            memo[key] = idx
            return idx
        if isinstance(node, String):
            raise ValueError(f"String literal {node.value!r} is only supported as a static keyword argument")

        if isinstance(node, StatelessJaxCall):
            child_ids = tuple(build(a) for a in node.args)
            idx = len(nodes)
            nodes.append(
                DagNode(
                    op=_build_stateless_jax_op(node, tuple(nodes[cid].op for cid in child_ids)),
                    child_ids=child_ids,
                )
            )
            memo[key] = idx
            return idx

        if isinstance(node, Call):
            if node.fn == "groupby":
                raise ValueError("nested groupby inside a groupby rhs is not supported in jax_flat")
            child_ids = tuple(
                build(a)
                for a in node.args
                if not ((node.fn, ANY_ARITY) in OP_FACTORIES and isinstance(a, String))
            )
            op, drop_child_idx = _build_op(node, tuple(nodes[cid].op for cid in child_ids))
            if drop_child_idx is not None:
                child_ids = tuple(cid for i, cid in enumerate(child_ids) if i != drop_child_idx)
            idx = len(nodes)
            nodes.append(DagNode(op=op, child_ids=child_ids))
            memo[key] = idx
            return idx

        raise ValueError(f"Unsupported groupby rhs node: {node}")

    output_id = build(rhs)
    node_tuple = tuple(nodes)
    state_layout = _build_state_layout(node_tuple)
    return (
        InnerGraphOp(
            nodes=node_tuple,
            output_id=output_id,
            state_layout=state_layout,
            n_inputs=len(feed_exprs),
            is_stateful=state_layout.total_leaves > 0,
            output_kind=node_tuple[output_id].op.output_kind,
        ),
        tuple(feed_exprs),
    )


def _groupby_capacity_kwargs(expr: Call, has_universe_groups: bool) -> dict[str, int]:
    capacity = 2048 if has_universe_groups else 4096
    hash_capacity = None
    for name, value in expr.kwargs:
        literal = int(value.value)
        if literal <= 0:
            raise ValueError(f"groupby {name} must be positive")
        if name == "capacity":
            capacity = literal
        elif name == "hash_capacity":
            hash_capacity = literal
    return {
        "capacity": capacity,
        "hash_capacity": max(hash_capacity if hash_capacity is not None else capacity * 2, capacity),
    }


def _compile_groupby_node(
    expr: Call,
    memo: dict[tuple[Any, ...], int],
    nodes: list[DagNode],
    input_names: list[str],
) -> int:
    _validate_groupby_canonical_form(expr)
    key_items = _canonical_groupby_key_items(expr.args[0])
    universe_items = [item for item in key_items if isinstance(item, Universe)]

    dynamic_items = [item for item in key_items if not isinstance(item, Universe)]
    inner_op, feed_exprs = _compile_groupby_inner_op(expr.args[2], expr.args[1])
    universe_groups = _resolve_universe_groups(universe_items[0]) if universe_items else None

    child_ids = tuple(_compile_node(k, memo, nodes, input_names) for k in dynamic_items) + tuple(
        _compile_node(feed, memo, nodes, input_names) for feed in feed_exprs
    )
    idx = len(nodes)
    groupby_kwargs = _groupby_capacity_kwargs(expr, universe_groups is not None)
    nodes.append(
        DagNode(
            op=GroupByOp(
                inner_op=inner_op,
                n_keys=len(dynamic_items),
                universe_groups=universe_groups,
                **groupby_kwargs,
            ),
            child_ids=child_ids,
        )
    )
    memo[_expr_key(expr)] = idx
    return idx


def _compile_buffer_shift_node(
    expr: Call,
    memo: dict[tuple[Any, ...], int],
    nodes: list[DagNode],
    input_names: list[str],
) -> int:
    if len(expr.args) != 3:
        raise ValueError("buffer expects args: shift_expr, min_lag, max_lag")
    shift_expr = expr.args[0]
    if not isinstance(shift_expr, Call) or shift_expr.fn != "shift" or len(shift_expr.args) not in (2, 3):
        raise ValueError("buffer first arg must be a direct shift(...) expression")
    max_size_arg = shift_expr.args[2] if len(shift_expr.args) == 3 else shift_expr.args[1]
    max_size = max(0, _literal_int_arg(max_size_arg, "shift", 3 if len(shift_expr.args) == 3 else 2))
    max_lag = _literal_int_arg(expr.args[2], "buffer", 3)
    if max_lag < 1:
        raise ValueError("buffer max_lag must be >= 1")
    if max_lag > max_size:
        raise ValueError("buffer max_lag must be <= shift max_size")

    child_ids = (
        _compile_node(shift_expr.args[0], memo, nodes, input_names),
        _compile_node(shift_expr.args[1], memo, nodes, input_names),
        _compile_node(expr.args[1], memo, nodes, input_names),
    )
    idx = len(nodes)
    nodes.append(DagNode(op=BufferShiftOp(max_size=max_size, max_lag=max_lag), child_ids=child_ids))
    memo[_expr_key(expr)] = idx
    return idx

def _compile_node(expr: Expr, memo: dict[tuple[Any, ...], int], nodes: list[DagNode], input_names: list[str]) -> int:
    key = _expr_key(expr)
    if key in memo:
        return memo[key]
    if isinstance(expr, Identifier):
        if expr.name not in input_names:
            input_names.append(expr.name)
        idx = len(nodes)
        nodes.append(DagNode(op=InputOp(input_index=input_names.index(expr.name)), child_ids=()))
        memo[key] = idx
        return idx
    if isinstance(expr, Number):
        idx = len(nodes)
        nodes.append(DagNode(op=LiteralOp(float(expr.value)), child_ids=()))
        memo[key] = idx
        return idx
    if isinstance(expr, String):
        raise ValueError(f"String literal {expr.value!r} is only supported as a static keyword argument")
    if isinstance(expr, StatelessJaxCall):
        child_ids = tuple(_compile_node(a, memo, nodes, input_names) for a in expr.args)
        idx = len(nodes)
        nodes.append(
            DagNode(
                op=_build_stateless_jax_op(expr, tuple(nodes[cid].op for cid in child_ids)),
                child_ids=child_ids,
            )
        )
        memo[key] = idx
        return idx
    if isinstance(expr, Call) and expr.fn == "groupby":
        return _compile_groupby_node(expr, memo, nodes, input_names)
    if isinstance(expr, Call) and expr.fn == "buffer":
        return _compile_buffer_shift_node(expr, memo, nodes, input_names)
    if not isinstance(expr, Call):
        raise ValueError(f"Unsupported node {expr}")

    child_ids = tuple(
        _compile_node(a, memo, nodes, input_names)
        for a in expr.args
        if not ((expr.fn, ANY_ARITY) in OP_FACTORIES and isinstance(a, String))
    )
    op, drop_child_idx = _build_op(expr, tuple(nodes[cid].op for cid in child_ids))
    if drop_child_idx is not None:
        child_ids = tuple(cid for i, cid in enumerate(child_ids) if i != drop_child_idx)
    idx = len(nodes)
    nodes.append(DagNode(op=op, child_ids=child_ids))
    memo[key] = idx
    return idx

def _expand_dsl(node: Expr, dsl_registry: DSLFunctionRegistry, depth: int = 0) -> Expr:
    if depth > 256:
        raise ValueError("Exceeded max DSL expansion depth (256)")
    if isinstance(node, Call):
        expanded_args = tuple(_expand_dsl(arg, dsl_registry, depth) for arg in node.args)
        expanded_kwargs = tuple((key, _expand_dsl(value, dsl_registry, depth)) for key, value in node.kwargs)
        expanded_node = Call(node.fn, expanded_args, expanded_kwargs)
        macro = dsl_registry.get(expanded_node.fn)
        if macro is None:
            return expanded_node
        result = macro(*expanded_node.args, **dict(expanded_node.kwargs))
        if _expr_key(result) == _expr_key(expanded_node):
            return expanded_node
        return _expand_dsl(result, dsl_registry, depth + 1)
    if isinstance(node, StatelessJaxCall):
        return StatelessJaxCall(
            fn=node.fn,
            args=tuple(_expand_dsl(arg, dsl_registry, depth) for arg in node.args),
            output_kind=node.output_kind,
            output_width=node.output_width,
            name=node.name,
        )
    if isinstance(node, KeyTuple):
        return KeyTuple(tuple(_expand_dsl(item, dsl_registry, depth) for item in node.items))
    return node



def compile_formula(formula: str | Expr, dsl_registry: DSLFunctionRegistry | None = None) -> JaxFlatRuntime:
    expr = parse_formula(formula) if isinstance(formula, str) else formula
    expr = _expand_dsl(expr, dsl_registry or DEFAULT_DSL_REGISTRY)
    expr = _normalize_static_jax_flat_kwargs(expr)
    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    input_names: list[str] = []
    out = _compile_node(expr, memo, nodes, input_names)
    layout = _build_state_layout(tuple(nodes))
    return JaxFlatRuntime(
        program=StreamingProgram(
            nodes=tuple(nodes),
            outputs=(out,),
            input_names=tuple(input_names),
            state_layout=layout,
        )
    )
