from __future__ import annotations

from dataclasses import dataclass
import tempfile
from time import perf_counter
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from trading_dsl_engine.base.compiler import CompileStats, FormulaCompileError
from trading_dsl_engine.base.dsl import DEFAULT_DSL_REGISTRY, DSLFunctionRegistry
from trading_dsl_engine.base.parser import Call, Expr, Identifier, Number, Universe, parse_formula
from trading_dsl_engine.jax.ops import (
    GroupByOp,
    InputOp,
    KeyValidatedOp,
    LiteralOp,
    LocalValueOp,
    ScopedGroupByOp,
    UniverseGroupByOp,
    _make_call_op,
    _op_is_stateless,
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
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(a) for a in node.args))
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
                if expr.name not in local_inputs:
                    raise FormulaCompileError("groupby local op expressions may only reference the 'self_' lhs placeholder")
                op = local_inputs[expr.name]
            else:
                inputs.setdefault(expr.name, len(inputs))
                op = InputOp(inputs[expr.name])
        elif isinstance(expr, Number):
            op = LiteralOp(float(expr.value))
        elif isinstance(expr, Call):
            macro = dsl_registry.get(expr.fn)
            if macro is not None:
                op = build(macro(*expr.args), local_inputs)
            elif expr.fn == "groupby" and len(expr.args) == 2 and isinstance(expr.args[0], Universe):
                child = build(expr.args[1], local_inputs)
                op = UniverseGroupByOp(child, _resolve_universe_groups(expr.args[0], column_names))
            elif expr.fn == "groupby" and len(expr.args) == 2:
                key_child = build(expr.args[0], local_inputs)
                op_child = build(expr.args[1], local_inputs)
                if _op_is_stateless(op_child):
                    op = KeyValidatedOp(key_child, op_child, output_kind=op_child.output_kind)
                else:
                    op = GroupByOp(key_child, op_child, len(inputs), output_kind=op_child.output_kind)
            elif expr.fn == "groupby" and len(expr.args) == 3:
                key_child = build(expr.args[0], local_inputs)
                lhs_child = build(expr.args[1], local_inputs)
                local_value = LocalValueOp(lhs_child.output_kind, lhs_child.output_kind)
                rhs_child = build(expr.args[2], {"self_": local_value})
                op = ScopedGroupByOp(key_child, lhs_child, rhs_child, output_kind=rhs_child.output_kind)
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
    out_path: str | None = f"{tempfile.gettempdir()}/trading_dsl_engine_jax_out.memmap",
    chunk_size: int = 8192,
):
    inputs = _as_aligned_inputs(engine, data)
    result = np.asarray(_jit_batch(engine.compiled, inputs))
    if out is not None:
        out[...] = result
        return out
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
