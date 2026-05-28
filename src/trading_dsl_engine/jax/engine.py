from __future__ import annotations

from dataclasses import dataclass
import os
import tempfile
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

_DEFAULT_MEMMAP_OUT = "__trading_dsl_engine_jax_default_memmap__"


def _fresh_memmap_path(prefix: str) -> str:
    fd, path = tempfile.mkstemp(prefix=prefix, suffix=".memmap")
    os.close(fd)
    return path


from trading_dsl_engine.base.compiler import CompileStats, FormulaCompileError
from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, Universe, parse_formula
from trading_dsl_engine.jax.ops import (
    GroupByOp,
    InputOp,
    LiteralOp,
    LocalValueOp,
    TupleKeyOp,
    UniverseDynamicGroupByOp,
    UniverseGroupByOp,
    _make_call_op,
    _project_output,
)


@dataclass(frozen=True)
class JaxCompiledArtifact:
    compiled: "JaxProgram"
    input_names: tuple[str, ...]
    output_kind: str
    stats: CompileStats


class JaxProgram(eqx.Module):
    """Static Equinox module containing a compiled operator tree."""

    root: Any = eqx.field(static=True)
    n_inputs: int = eqx.field(static=True)
    output_kind: str = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        return self.root.init_state(n_instruments)

    def tick(self, state, frame2d):
        new_state, out = self.root.tick(state, frame2d)
        return new_state, _project_output(out, self.output_kind)

    def run_batch(self, inputs):
        n_instruments = inputs[0].shape[1]
        state0 = self.init_state(n_instruments)

        def step(state, rows):
            frame = jnp.stack(rows, axis=0)
            new_state, out = self.tick(state, frame)
            return new_state, out

        _, outputs = jax.lax.scan(step, state0, inputs)
        return outputs


@eqx.filter_jit
def _jit_tick(program: JaxProgram, state, frame2d):
    return program.tick(state, frame2d)


@eqx.filter_jit
def _jit_batch(program: JaxProgram, inputs):
    return program.run_batch(inputs)


class JaxEngineHandle:
    def __init__(self, compiled: JaxProgram, input_names: tuple[str, ...], output_kind: str):
        self.compiled = compiled
        self.input_names = input_names
        self.output_kind = output_kind
        self.output_code = {"scalar": 0, "vector": 1, "matrix": 2, "object": 3}[output_kind]
        self._state = None
        self._n_instruments = None

    def on_data(self, frame2d):
        frame = jnp.asarray(frame2d, dtype=jnp.float64)
        if self._state is None or self._n_instruments != frame.shape[1]:
            self._state = self.compiled.init_state(frame.shape[1])
            self._n_instruments = frame.shape[1]
        self._state, self._last = _jit_tick(self.compiled, self._state, frame)
        return self._last

    def emit(self):
        return np.asarray(self._last)


def _expr_key(node: Expr):
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", float(node.value))
    if isinstance(node, Universe):
        return ("univ", node.groups)
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_expr_key(item) for item in node.items))
    if isinstance(node, Call):
        return (
            "call",
            node.fn,
            tuple(_expr_key(a) for a in node.args),
            tuple((k, _expr_key(v)) for k, v in node.kwargs),
        )
    raise FormulaCompileError(f"Unsupported expression node: {node}")


def _resolve_universe_groups(universe: Universe, column_names):
    name_to_idx = {name: i for i, name in enumerate(column_names or ())}
    groups = []
    seen = set()
    for group in universe.groups:
        resolved = []
        for item in group:
            if isinstance(item, int):
                idx = item
            else:
                if item not in name_to_idx:
                    raise FormulaCompileError(
                        f"Unknown universe column {item!r}. Pass column_names to compile_formula/build_engine."
                    )
                idx = name_to_idx[item]
            if idx in seen:
                raise FormulaCompileError(f"Universe column index {idx} appears in more than one group")
            seen.add(idx)
            resolved.append(int(idx))
        groups.append(tuple(resolved))
    return tuple(groups)



def _canonical_groupby_key_items(key: Expr) -> tuple[Expr, ...]:
    if not isinstance(key, KeyTuple):
        key = KeyTuple((key,))
    if sum(1 for item in key.items if isinstance(item, Universe)) > 1:
        raise FormulaCompileError("groupby key tuple may contain at most one univ(...) element")
    return key.items


def _replace_self_placeholder(node: Expr, lhs: Expr) -> Expr:
    if isinstance(node, Identifier) and node.name == "self_":
        return lhs
    if isinstance(node, Call):
        return Call(
            node.fn,
            tuple(_replace_self_placeholder(arg, lhs) for arg in node.args),
            tuple((key, _replace_self_placeholder(value, lhs)) for key, value in node.kwargs),
        )
    if isinstance(node, KeyTuple):
        return KeyTuple(tuple(_replace_self_placeholder(item, lhs) for item in node.items))
    return node

def compile_formula(
    formula: str | Expr,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
) -> JaxCompiledArtifact:
    started_at = perf_counter()
    ast_expr = parse_formula(formula) if isinstance(formula, str) else formula
    dsl_registry = dsl_registry or DEFAULT_DSL_REGISTRY
    inputs: dict[str, int] = {}
    cache: dict[tuple, Any] = {}
    cache_hits = 0
    expanded_nodes = 0

    def build(expr: Expr, local_inputs: dict[str, Any] | None = None) -> Any:
        nonlocal cache_hits, expanded_nodes
        use_cache = local_inputs is None
        key = _expr_key(expr)
        if use_cache and key in cache:
            cache_hits += 1
            return cache[key]
        expanded_nodes += 1
        if isinstance(expr, Identifier):
            if local_inputs is not None:
                if expr.name in local_inputs:
                    op = local_inputs[expr.name]
                else:
                    inputs.setdefault(expr.name, len(inputs))
                    op = InputOp(inputs[expr.name])
            else:
                inputs.setdefault(expr.name, len(inputs))
                op = InputOp(inputs[expr.name])
        elif isinstance(expr, Number):
            op = LiteralOp(float(expr.value))
        elif isinstance(expr, Call):
            if expr.kwargs:
                unsupported_kwargs = {name for name, _ in expr.kwargs} - {"capacity", "hash_capacity"}
                if expr.fn != "groupby" or unsupported_kwargs:
                    raise FormulaCompileError(f"Keyword arguments are not supported for {expr.fn}")
            macro = dsl_registry.get(expr.fn)
            if macro is not None:
                op = build(macro(*expr.args), local_inputs)
            elif expr.fn == "groupby" and len(expr.args) == 3:
                key_items = _canonical_groupby_key_items(expr.args[0])
                universe_items = [item for item in key_items if isinstance(item, Universe)]
                dynamic_items = [item for item in key_items if not isinstance(item, Universe)]
                if universe_items:
                    op_expr = _replace_self_placeholder(expr.args[2], expr.args[1])
                    op_child = build(op_expr, local_inputs)
                    output_kind = "vector" if op_child.output_kind == "scalar" else op_child.output_kind
                    groups = _resolve_universe_groups(universe_items[0], column_names)
                    if len(dynamic_items) == 0:
                        op = UniverseGroupByOp(op_child, groups)
                    else:
                        key_children = tuple(build(key, local_inputs) for key in dynamic_items)
                        key_child = key_children[0] if len(key_children) == 1 else TupleKeyOp(key_children)
                        op = UniverseDynamicGroupByOp(key_child, op_child, groups, output_kind=output_kind)
                    if use_cache:
                        cache[key] = op
                    return op
                key_children = tuple(build(key, local_inputs) for key in key_items)
                key_child = key_children[0] if len(key_children) == 1 else TupleKeyOp(key_children)
                lhs_child = build(expr.args[1], local_inputs)
                local_value = LocalValueOp(lhs_child.output_kind, lhs_child.output_kind)
                rhs_child = build(expr.args[2], {"self_": local_value})
                output_kind = "vector" if rhs_child.output_kind == "scalar" else rhs_child.output_kind
                op = GroupByOp(key_child, rhs_child, len(inputs), lhs=lhs_child, output_kind=output_kind)
            elif expr.fn == "groupby":
                raise FormulaCompileError(
                    "groupby only supports canonical form: groupby((key1, ..., maybe_univ, ...), lhs, op_using_self_)"
                )
            else:
                children = tuple(build(arg, local_inputs) for arg in expr.args)
                op = _make_call_op(expr.fn, expr.args, children)
        else:
            raise FormulaCompileError(f"Unsupported expression node: {expr}")
        if use_cache:
            cache[key] = op
        return op

    root = build(ast_expr)
    return JaxCompiledArtifact(
        compiled=JaxProgram(root, len(inputs), root.output_kind),
        input_names=tuple(inputs.keys()),
        output_kind=root.output_kind,
        stats=CompileStats(
            expanded_nodes=expanded_nodes,
            cache_hits=cache_hits,
            compile_seconds=perf_counter() - started_at,
        ),
    )


def build_jax_engine(
    formula: str | Expr,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
):
    artifact = compile_formula(formula, dsl_registry=dsl_registry, column_names=column_names)
    return JaxEngineHandle(artifact.compiled, artifact.input_names, artifact.output_kind)


build_engine = build_jax_engine


def _as_aligned_inputs(engine: JaxEngineHandle, data: dict[str, np.ndarray]):
    arrays = []
    for name in engine.input_names:
        arr = np.asarray(data[name], dtype=np.float64)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D input for '{name}', got shape {arr.shape}")
        arrays.append(jnp.asarray(arr))
    return tuple(arrays)


def run_batch_from_mapping(
    engine: JaxEngineHandle,
    data: dict[str, np.ndarray],
    out=None,
    out_path: str | None = _DEFAULT_MEMMAP_OUT,
    chunk_size: int = 8192,
):
    inputs = _as_aligned_inputs(engine, data)
    result = np.asarray(_jit_batch(engine.compiled, inputs))
    if out is not None:
        out[...] = result
        return out
    if out_path == _DEFAULT_MEMMAP_OUT:
        out_path = _fresh_memmap_path("trading_dsl_engine_jax_out_")
    if out_path is not None:
        mapped = np.memmap(out_path, mode="w+", dtype=np.float64, shape=result.shape)
        mapped[...] = result
        return mapped
    return result


def update_from_mapping(engine: JaxEngineHandle, data: dict[str, np.ndarray]):
    frame = np.empty(
        (len(engine.input_names), np.asarray(data[engine.input_names[0]]).shape[0]),
        dtype=np.float64,
    )
    for i, name in enumerate(engine.input_names):
        frame[i, :] = np.asarray(data[name], dtype=np.float64)
    engine.on_data(frame)
    return engine.emit()
