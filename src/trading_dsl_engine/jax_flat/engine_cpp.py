from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trading_dsl_engine.base.dsl import DSLFunctionRegistry
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.jax_flat.engine import (
    DagNode,
    InnerGraphOp,
    StateLayout,
    StreamingProgram,
    compile_formula as compile_formula_jax,
)
from trading_dsl_engine.jax_flat.ops import (
    CacheOp,
    CumsumOp,
    EwmOp,
    FFillOp,
    FutureRbfBasisSumOp,
    GroupByOp,
    InputOp,
    InstrumentBasisMeanOp,
    LiteralOp,
    NaryOp,
    Op,
    RbfBasisOp,
    RidgeOp,
    ShiftOp,
)


@dataclass(frozen=True)
class CppFlatRuntime:
    """Allocation-conscious native tick-path runtime for supported flat formulas.

    The C++ core owns flattened node specs and preallocates per-node scratch in
    ``init_state``. ``run_batch`` mirrors the JAX-flat runtime API for batch
    execution. ``tick_into`` remains the lowest-allocation row hot path for
    callers that can provide reusable state and output buffers.
    """

    program: StreamingProgram
    core: Any
    supported_ops: tuple[str, ...]

    def init_state(self, n_instruments: int):
        return self.core.init_state(n_instruments)

    def tick(self, state, *input_rows):
        return self.core.tick(state, *input_rows)

    def tick_into(self, state, out, *input_rows) -> None:
        self.core.tick_into(state, out, *input_rows)

    def run_batch(self, inputs, states=None, out=None):
        inputs = _normalize_batch_inputs_for_program(self.program, inputs)
        if not inputs:
            raise ValueError("run_batch requires at least one input array")
        n_steps, n_instruments = inputs[0].shape
        for arr in inputs[1:]:
            if arr.shape != (n_steps, n_instruments):
                raise ValueError("All inputs must share aligned shape (time, n_instruments)")
        state = states or self.init_state(n_instruments)
        np_inputs = tuple(np.asarray(arr, dtype=np.float64) for arr in inputs)
        if out is None:
            raw = self.core.run_batch(state, *np_inputs)
            return state, _reshape_cpp_batch_output(self.program, raw, n_steps, n_instruments)
        self.core.run_batch_into(state, out, *np_inputs)
        return state, out

    def run_batch_into(self, state, out, inputs):
        inputs = _normalize_batch_inputs_for_program(self.program, inputs)
        np_inputs = tuple(np.asarray(arr, dtype=np.float64) for arr in inputs)
        self.core.run_batch_into(state, out, *np_inputs)
        return state, out


def compile_formula(formula: str | Expr, dsl_registry: DSLFunctionRegistry | None = None) -> CppFlatRuntime:
    """Compile a supported jax_flat formula to the native C++ tick runtime.

    This backend intentionally covers the allocation-sensitive scalar/vector tick
    subset first. Unsupported operators raise a clear ``NotImplementedError`` so
    callers can keep using the JAX-flat runtime for full DSL coverage.
    """
    from trading_dsl_engine.jax_flat import _cpp_flat

    runtime = compile_formula_jax(formula, dsl_registry=dsl_registry, cpp=False)
    node_specs, supported = _cpp_node_specs(runtime.program)
    core = _cpp_flat.make_runtime(node_specs, runtime.program.outputs[0], runtime.program.state_layout.total_leaves)
    return CppFlatRuntime(program=runtime.program, core=core, supported_ops=tuple(sorted(set(supported))))


def _reshape_cpp_batch_output(program: StreamingProgram, raw, n_steps: int, n_instruments: int):
    root = program.nodes[program.outputs[0]].op
    arr = np.asarray(raw)
    if root.output_kind == "matrix":
        width = int(root.output_width) if root.output_width is not None else n_instruments
        if isinstance(root, NaryOp) and root.cpp_name == "einsum":
            subscripts = root.cpp_str_param
            output = subscripts.split("->", 1)[1] if "->" in subscripts else ""
            if "i" not in output and len(output) == 2:
                input_terms = subscripts.split("->", 1)[0].split(",")
                row_width = next(
                    (program.nodes[cid].op.output_width for term, cid in zip(input_terms, program.nodes[program.outputs[0]].child_ids) if len(term) == 2 and term[1] == output[0]),
                    None,
                )
                if row_width is not None:
                    return arr.reshape((n_steps, int(row_width), width))
        return arr.reshape((n_steps, n_instruments, width))
    if root.output_kind == "scalar" and arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _normalize_batch_inputs_for_program(program: StreamingProgram, inputs):
    if isinstance(inputs, dict):
        missing = [name for name in program.input_names if name not in inputs]
        if missing:
            raise ValueError(f"Missing jax_flat C++ input(s): {missing}")
        inputs = tuple(inputs[name] for name in program.input_names)
    else:
        inputs = tuple(inputs)
    if len(inputs) != len(program.input_names):
        raise ValueError(f"expected {len(program.input_names)} input array(s), got {len(inputs)}")
    for name, arr in zip(program.input_names, inputs):
        if np.asarray(arr).ndim != 2:
            raise ValueError(f"Expected 2D input for '{name}', got shape {np.asarray(arr).shape}")
    return inputs


def _cpp_node_specs(program: StreamingProgram):
    if len(program.outputs) != 1:
        raise NotImplementedError("C++ jax_flat currently supports exactly one output")
    specs = []
    supported = []

    def spec_tuple(
        name,
        children=(),
        input_index=-1,
        state_index=-1,
        literal=0.0,
        param=0.0,
        int_param=0,
        width=1,
        feature_widths=(),
        inner_specs=(),
        inner_output_id=-1,
        str_param="",
    ):
        return (
            name,
            tuple(children),
            input_index,
            state_index,
            float(literal),
            float(param),
            int(int_param),
            int(1 if width is None else width),
            tuple(int(w) for w in feature_widths),
            tuple(inner_specs),
            int(inner_output_id),
            str(str_param),
        )

    for idx, node in enumerate(program.nodes):
        op = node.op
        field = program.state_layout.node_fields[idx]
        state_index = field.index
        width = op.output_width or 1
        if isinstance(op, InputOp):
            specs.append(spec_tuple("input", input_index=op.input_index, width=1))
            supported.append("input")
            continue
        if isinstance(op, LiteralOp):
            specs.append(spec_tuple("literal", literal=op.value, width=1))
            supported.append("literal")
            continue
        if isinstance(op, GroupByOp):
            group_spec, group_supported = _cpp_groupby_spec(op, node.child_ids, state_index, spec_tuple)
            specs.append(group_spec)
            supported.extend(group_supported)
            continue
        if isinstance(op, CacheOp):
            specs.append(spec_tuple("cache", node.child_ids, width=width))
            supported.append("cache")
            continue
        if isinstance(op, NaryOp) and op.cpp_name is not None:
            cpp_width = width
            if op.cpp_name == "outer":
                cpp_width = 0
            if op.cpp_name == "einsum" and op.output_width is None:
                cpp_width = 0
            if op.cpp_name == "get_beta":
                beta_child = program.nodes[node.child_ids[0]].op
                if isinstance(beta_child, RidgeOp):
                    cpp_width = sum(beta_child.feature_widths)
                elif isinstance(beta_child, InstrumentBasisMeanOp):
                    cpp_width = beta_child.feature_width
                else:
                    raise NotImplementedError("C++ jax_flat get_beta expects direct Ridge or InstrumentBasisMean child")
            feature_widths = ()
            if op.cpp_name == "einsum":
                feature_widths = tuple(program.nodes[cid].op.output_width or 1 for cid in node.child_ids)
            specs.append(spec_tuple(op.cpp_name, node.child_ids, int_param=op.cpp_int_param, param=op.cpp_param, width=cpp_width, feature_widths=feature_widths, str_param=getattr(op, "cpp_str_param", "")))
            supported.append(op.cpp_name)
            continue
        if isinstance(op, CumsumOp):
            specs.append(spec_tuple("cumsum", node.child_ids, state_index=state_index, width=width))
            supported.append("cumsum")
            continue
        if isinstance(op, EwmOp) and op.span is not None:
            specs.append(spec_tuple("ewm", node.child_ids, state_index=state_index, param=op.span, width=width))
            supported.append("ewm")
            continue
        if isinstance(op, FFillOp) and not op.dynamic_limit:
            limit = -1 if op.limit is None else op.limit
            specs.append(spec_tuple("ffill", node.child_ids, state_index=state_index, int_param=limit, width=width))
            supported.append("ffill")
            continue
        if isinstance(op, ShiftOp):
            specs.append(spec_tuple("shift", node.child_ids, state_index=state_index, int_param=op.max_size, width=width))
            supported.append("shift")
            continue
        if isinstance(op, RbfBasisOp):
            specs.append(spec_tuple("rbf_basis", node.child_ids, int_param=op.n_basis, width=op.n_basis))
            supported.append("rbf_basis")
            continue
        if isinstance(op, FutureRbfBasisSumOp):
            specs.append(spec_tuple("future_rbf_basis_sum", node.child_ids, int_param=op.n_basis, param=op.n_steps, width=op.n_basis))
            supported.append("future_rbf_basis_sum")
            continue
        if isinstance(op, InstrumentBasisMeanOp):
            specs.append(
                spec_tuple(
                    "instrument_basis_mean",
                    node.child_ids,
                    state_index=state_index,
                    int_param=1 if op.has_weights else 0,
                    width=1,
                    feature_widths=(op.feature_width,),
                )
            )
            supported.append("instrument_basis_mean")
            continue
        if isinstance(op, RidgeOp):
            k = sum(op.feature_widths)
            specs.append(spec_tuple("ridge", node.child_ids, state_index=state_index, width=1, feature_widths=op.feature_widths))
            supported.append("ridge")
            continue
        raise NotImplementedError(f"C++ jax_flat does not yet support node {idx}: {type(op).__name__}")
    return tuple(specs), supported



def _cpp_groupby_spec(op: GroupByOp, child_ids: tuple[int, ...], state_index: int, spec_tuple):
    inner = op.inner_op
    if not isinstance(inner, InnerGraphOp):
        raise NotImplementedError("C++ jax_flat groupby expects lowered inner graph")
    if inner.n_inputs != 1:
        raise NotImplementedError("C++ jax_flat groupby currently supports RHS graphs over self_ only")
    inner_specs, supported = _cpp_inner_node_specs(inner, spec_tuple)
    return (
        spec_tuple(
            "group",
            child_ids,
            state_index=state_index,
            param=op.capacity,
            int_param=op.n_keys,
            width=op.output_width or 1,
            feature_widths=_encode_universe_groups(op.universe_groups),
            inner_specs=inner_specs,
            inner_output_id=inner.output_id,
        ),
        ["group", *supported],
    )


def _encode_universe_groups(groups: tuple[tuple[int, ...], ...] | None) -> tuple[int, ...]:
    if groups is None:
        return ()
    encoded: list[int] = [len(groups)]
    for group in groups:
        encoded.append(len(group))
        encoded.extend(int(col) for col in group)
    return tuple(encoded)


def _cpp_inner_node_specs(inner: InnerGraphOp, spec_tuple):
    specs = []
    supported = []
    for idx, node in enumerate(inner.nodes):
        op = node.op
        field = inner.state_layout.node_fields[idx]
        state_index = field.index
        width = op.output_width or 1
        if isinstance(op, InputOp):
            specs.append(spec_tuple("input", input_index=op.input_index, width=1))
            supported.append("inner_input")
            continue
        if isinstance(op, LiteralOp):
            specs.append(spec_tuple("literal", literal=op.value, width=1))
            supported.append("inner_literal")
            continue
        if isinstance(op, CacheOp):
            specs.append(spec_tuple("cache", node.child_ids, width=width))
            supported.append("cache")
            continue
        if isinstance(op, NaryOp) and op.cpp_name is not None:
            cpp_width = 0 if op.cpp_name in {"outer", "einsum"} and op.output_width is None else width
            if cpp_width != 1:
                raise NotImplementedError(f"C++ jax_flat groupby inner op {op.cpp_name!r} currently supports vector/scalar width 1")
            specs.append(spec_tuple(op.cpp_name, node.child_ids, param=op.cpp_param, int_param=op.cpp_int_param, width=cpp_width, str_param=getattr(op, "cpp_str_param", "")))
            supported.append(f"inner_{op.cpp_name}")
            continue
        if isinstance(op, CumsumOp):
            specs.append(spec_tuple("cumsum", node.child_ids, state_index=state_index, width=width))
            supported.append("inner_cumsum")
            continue
        if isinstance(op, EwmOp) and op.span is not None:
            specs.append(spec_tuple("ewm", node.child_ids, state_index=state_index, param=op.span, width=width))
            supported.append("inner_ewm")
            continue
        if isinstance(op, FFillOp) and not op.dynamic_limit:
            limit = -1 if op.limit is None else op.limit
            specs.append(spec_tuple("ffill", node.child_ids, state_index=state_index, int_param=limit, width=width))
            supported.append("inner_ffill")
            continue
        raise NotImplementedError(f"C++ jax_flat groupby inner node {idx} unsupported: {type(op).__name__}")
    return tuple(specs), supported

def _require_vector_cpp_op(op: Op, name: str) -> None:
    if op.output_kind != "vector" or op.output_width not in (None, 1):
        raise NotImplementedError(f"C++ jax_flat op {name!r} currently supports vector outputs only")

