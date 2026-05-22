from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from trading_dsl_engine.base.parser import Call, Expr, Identifier, Number, parse_formula
from trading_dsl_engine.jax_flat.ops import CumsumOp, EwmOp, InputOp, LiteralOp, OP_FACTORIES, Op


@dataclass(frozen=True)
class DagNode:
    op: Any
    child_ids: tuple[int, ...]


@dataclass(frozen=True)
class StateFieldRef:
    start: int
    size: int


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


class JaxFlatRuntime(eqx.Module):
    program: StreamingProgram = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        sample = jnp.zeros((n_instruments,), dtype=jnp.float64)
        leaves = []
        for node in self.program.nodes:
            if not node.op.is_stateful:
                continue
            for _, init_leaf in node.op.state_spec(sample):
                leaves.append(init_leaf)
        return tuple(leaves)

    def tick_stream(self, state_leaves, *input_rows):
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
            state_fields = tuple(state_leaves[field.start + k] for k in range(field.size))
            next_fields, value = op.lower_stream_step(child_values, state_fields)
            if field.size:
                for k in range(field.size):
                    new_state[field.start + k] = next_fields[k]
            values[idx] = value

        outs = tuple(values[i] for i in self.program.outputs)
        return tuple(new_state), outs[0] if len(outs) == 1 else jnp.stack(outs, axis=0)

    def tick(self, state_leaves, *input_rows):
        return self.tick_stream(state_leaves, *input_rows)


def _expr_key(node: Expr):
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", float(node.value))
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(a) for a in node.args))
    raise ValueError(f"Unsupported expression: {node}")


def _build_op(expr: Call) -> tuple[Op, int | None]:
    if expr.fn == "ewm" and len(expr.args) == 2 and isinstance(expr.args[1], Number):
        return EwmOp(span=float(expr.args[1].value)), 1
    builder = OP_FACTORIES.get((expr.fn, len(expr.args)))
    if builder is None:
        raise ValueError(f"Unsupported function {expr.fn}/{len(expr.args)} in jax_flat")
    return builder(), None


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
    child_ids = tuple(_compile_node(a, memo, nodes, input_names) for a in expr.args)
    op, drop_child_idx = _build_op(expr)
    if drop_child_idx is not None:
        child_ids = tuple(cid for i, cid in enumerate(child_ids) if i != drop_child_idx)
    idx = len(nodes)
    nodes.append(DagNode(op=op, child_ids=child_ids))
    memo[key] = idx
    return idx


def _build_state_layout(nodes: tuple[DagNode, ...], sample: jax.Array) -> StateLayout:
    refs = []
    offset = 0
    for node in nodes:
        if node.op.is_stateful:
            size = len(node.op.state_spec(sample))
            refs.append(StateFieldRef(start=offset, size=size))
            offset += size
        else:
            refs.append(StateFieldRef(start=offset, size=0))
    return StateLayout(node_fields=tuple(refs), total_leaves=offset)


def compile_formula(formula: str | Expr) -> JaxFlatRuntime:
    expr = parse_formula(formula) if isinstance(formula, str) else formula
    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    input_names: list[str] = []
    out = _compile_node(expr, memo, nodes, input_names)
    sample = jnp.zeros((1,), dtype=jnp.float64)
    layout = _build_state_layout(tuple(nodes), sample)
    return JaxFlatRuntime(
        program=StreamingProgram(
            nodes=tuple(nodes),
            outputs=(out,),
            input_names=tuple(input_names),
            state_layout=layout,
        )
    )


@eqx.filter_jit
def jit_tick_stream(runtime: JaxFlatRuntime, state_leaves, *input_rows):
    return runtime.tick_stream(state_leaves, *input_rows)


@eqx.filter_jit
def jit_tick(runtime: JaxFlatRuntime, state_leaves, *input_rows):
    return runtime.tick(state_leaves, *input_rows)
