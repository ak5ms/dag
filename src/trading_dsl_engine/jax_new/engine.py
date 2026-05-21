from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from trading_dsl_engine.base.parser import Call, Expr, Identifier, Number, parse_formula
from trading_dsl_engine.jax_new.ops import (
    EwmOp,
    EmptyState,
    InputOp,
    LiteralOp,
    Op,
    OP_FACTORIES,
    state_value,
)


@dataclass(frozen=True)
class DagNode:
    op: Any
    child_ids: tuple[int, ...]


@dataclass(frozen=True)
class DagProgram:
    nodes: tuple[DagNode, ...]
    outputs: tuple[int, ...]
    input_names: tuple[str, ...]
    stateful_node_ids: tuple[int, ...]
    stateful_pos_by_node: tuple[int, ...]


class JaxDagRuntime(eqx.Module):
    program: DagProgram = eqx.field(static=True)

    def init_state(self, n_instruments: int):
        sample = jnp.zeros((n_instruments,), dtype=jnp.float64)
        return tuple(node.op.init_state(sample) for node in self.program.nodes)

    def tick(self, states, *input_rows):
        new_states: list[Any] = []
        for idx, node in enumerate(self.program.nodes):
            op = node.op
            if isinstance(op, InputOp):
                new_states.append(EmptyState(value=input_rows[op.input_index]))
                continue
            if isinstance(op, LiteralOp):
                new_states.append(states[idx])
                continue
            child_states = tuple(new_states[cid] for cid in node.child_ids)
            new_states.append(op.step(states[idx], *child_states))
        outputs = tuple(_format_output(state_value(new_states[i]), self.program.nodes[i].op.output_kind) for i in self.program.outputs)
        return tuple(new_states), outputs[0] if len(outputs) == 1 else jnp.stack(outputs, axis=0)

    def _tick_with_compact_carry(self, carry_states, full_states, *input_rows):
        new_states: list[Any] = []
        for idx, node in enumerate(self.program.nodes):
            op = node.op
            if isinstance(op, InputOp):
                new_states.append(EmptyState(value=input_rows[op.input_index]))
                continue
            if isinstance(op, LiteralOp):
                new_states.append(full_states[idx])
                continue
            child_states = tuple(new_states[cid] for cid in node.child_ids)
            state_pos = self.program.stateful_pos_by_node[idx]
            if state_pos >= 0:
                prev_state = carry_states[state_pos]
            else:
                prev_state = full_states[idx]
            new_states.append(op.step(prev_state, *child_states))
        outputs = tuple(_format_output(state_value(new_states[i]), self.program.nodes[i].op.output_kind) for i in self.program.outputs)
        new_carry = tuple(new_states[nid] for nid in self.program.stateful_node_ids)
        return new_carry, outputs[0] if len(outputs) == 1 else jnp.stack(outputs, axis=0)

    def run_batch(self, states, inputs):
        if not inputs:
            raise ValueError("run_batch requires at least one input array")
        n_steps = inputs[0].shape[0]
        for arr in inputs[1:]:
            if arr.shape[0] != n_steps:
                raise ValueError("All inputs must have identical timestep length")
        if not states:
            state0 = self.init_state(inputs[0].shape[1])
        else:
            n_instruments = state_value(states[0]).shape[0]
            if inputs[0].shape[1] != n_instruments:
                raise ValueError("Input instrument width must match provided state")
            state0 = states

        if not self.program.stateful_node_ids:
            row_fn = lambda *rows: self.tick(state0, *rows)[1]
            ys = jax.vmap(row_fn, in_axes=0)(*inputs)
            return state0, ys

        carry0 = tuple(state0[nid] for nid in self.program.stateful_node_ids)
        out = jax.lax.scan(lambda c, rows: self._tick_with_compact_carry(c, state0, *rows), carry0, xs=inputs, unroll=1)
        carry_out, ys = out
        state_out = list(state0)
        for i, nid in enumerate(self.program.stateful_node_ids):
            state_out[nid] = carry_out[i]
        return tuple(state_out), ys


def _format_output(value, output_kind: str):
    if output_kind == "vector":
        return value[:, None]
    if output_kind == "scalar":
        return value.reshape(1, 1) if value.ndim == 0 else value
    return value

@eqx.filter_jit
def jit_tick(runtime: JaxDagRuntime, states, *input_rows):
    return runtime.tick(states, *input_rows)

@eqx.filter_jit
def jit_batch(runtime: JaxDagRuntime, inputs, states=tuple()):
    return runtime.run_batch(states=states, inputs=inputs)

def _expr_key(node: Expr):
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", float(node.value))
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(a) for a in node.args))
    raise ValueError(f"Unsupported expression: {node}")

def _build_op(expr: Call) -> tuple[Op, int | None]:
    fn = expr.fn
    if fn == "ewm" and len(expr.args) == 2 and isinstance(expr.args[1], Number):
        return EwmOp(span=float(expr.args[1].value)), 1
    builder = OP_FACTORIES.get((fn, len(expr.args)))
    if builder is None:
        raise ValueError(f"Unsupported function {expr.fn}/{len(expr.args)} in jax_new")
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
    if not isinstance(expr, Call):
        raise ValueError(f"Unsupported node {expr}")
    child_ids = tuple(_compile_node(a, memo, nodes, input_names) for a in expr.args)
    op, drop_child_idx = _build_op(expr)
    if drop_child_idx is not None:
        child_ids = tuple(cid for i, cid in enumerate(child_ids) if i != drop_child_idx)
    idx = len(nodes)
    nodes.append(DagNode(op=op, child_ids=child_ids))
    memo[key] = idx
    return idx


def compile_formula(formula: str | Expr) -> JaxDagRuntime:
    expr = parse_formula(formula) if isinstance(formula, str) else formula
    nodes: list[DagNode] = []
    memo: dict[tuple[Any, ...], int] = {}
    input_names: list[str] = []
    out = _compile_node(expr, memo, nodes, input_names)
    stateful_node_ids = tuple(i for i, node in enumerate(nodes) if node.op.stateful)
    stateful_pos = {nid: i for i, nid in enumerate(stateful_node_ids)}
    stateful_pos_by_node = tuple(stateful_pos.get(i, -1) for i in range(len(nodes)))
    program = DagProgram(nodes=tuple(nodes), outputs=(out,), input_names=tuple(input_names), stateful_node_ids=stateful_node_ids, stateful_pos_by_node=stateful_pos_by_node)
    return JaxDagRuntime(program=program)