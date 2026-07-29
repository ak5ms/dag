"""Lower the canonical :class:`StreamingProgram` to cpp_new IR."""
from __future__ import annotations

from dataclasses import replace

from trading_dsl_engine.cpp_new.ir import Diagnostics, FormulaIR, GraphOutput, InputView, KernelNode, KernelTraits, ScratchSlot, StateSlot, ValueKind, ValueType
from trading_dsl_engine.cpp_new.registry import descriptor
from trading_dsl_engine.jax_flat.engine_cpp import lower_native_plan


def _align(value: int, alignment: int = 64) -> int:
    return (value + alignment - 1) // alignment * alignment


def _value_type(shape: str, width: int) -> ValueType:
    kind = {"scalar": ValueKind.SCALAR, "matrix": ValueKind.MATRIX, "object": ValueKind.MODEL}.get(shape, ValueKind.INSTRUMENT_VECTOR)
    return ValueType(kind, width)


def _parameters(node) -> tuple[tuple[str, str], ...]:
    spec = node.legacy_spec
    values = {
        "input_index": str(spec[2]),
        "literal": repr(spec[4]),
        "param": repr(spec[5]),
        "int_param": str(spec[6]),
    }
    if node.opcode == "ewm":
        flags = spec[6] % 4
        values.update(ignore_na="true" if flags & 1 else "false", adjust="true" if flags & 2 else "false")
    return tuple(sorted(values.items()))


def _state_size(node, n_instruments: int) -> int:
    if node.opcode == "ewm":
        return _align(n_instruments * (8 + 8 + 1 + 8))
    if node.opcode == "ridge":
        features = max(1, sum(node.legacy_spec[8]))
        doubles = 3 * features * features + 5 * features + n_instruments
        flags = features * features + features
        return _align(doubles * 8 + flags)
    return _align(max(1, n_instruments) * 8)


def _scratch_size(node, n_instruments: int, *, root: bool) -> int:
    output = 0 if root or node.value_type.shape in {"object", "scalar"} or node.opcode == "input" else n_instruments * max(1, node.value_type.width) * 8
    if node.opcode == "xs_rank":
        output += n_instruments * (16 + 8)
    elif node.opcode == "ridge":
        features = max(1, sum(node.legacy_spec[8]))
        output += (2 * features * features + 3 * features) * 8
    return _align(output) if output else 0


def _color_scratch(requests: list[tuple[int, int, int, int]]) -> tuple[tuple[ScratchSlot, ...], int]:
    """First-fit interval coloring with one offset per reusable color."""
    colors: list[tuple[int, int, int]] = []  # last use, offset, capacity
    slots: list[ScratchSlot] = []
    arena_end = 0
    for node, size, start, end in requests:
        choice = next((i for i, (last, _, capacity) in enumerate(colors) if last < start and capacity >= size), None)
        if choice is None:
            offset = _align(arena_end)
            choice = len(colors)
            colors.append((end, offset, size))
            arena_end = offset + size
        else:
            _, offset, capacity = colors[choice]
            colors[choice] = (end, offset, capacity)
        slots.append(ScratchSlot(node, offset, size, start, end, choice))
    return tuple(slots), _align(arena_end)


def lower(program, *, n_instruments: int | None = None) -> FormulaIR:
    plan, _ = lower_native_plan(program)
    instruments = n_instruments or 1
    state_end = 0
    states: list[StateSlot] = []
    requests: list[tuple[int, int, int, int]] = []
    output_id = plan.output_id
    for node in plan.nodes:
        desc = descriptor(node.opcode)
        if desc.state_family:
            state_end = _align(state_end)
            size = _state_size(node, instruments)
            states.append(StateSlot(node.node_id, desc.state_family, state_end, size))
            state_end += size
        size = _scratch_size(node, instruments, root=node.node_id == output_id)
        if size:
            requests.append((node.node_id, size, node.live_from, node.live_until))
    scratch, scratch_bytes = _color_scratch(requests)
    scratch_by_node = {slot.node: (index, slot) for index, slot in enumerate(scratch)}
    state_by_node = {slot.node: index for index, slot in enumerate(states)}
    nodes = []
    for node in plan.nodes:
        desc = descriptor(node.opcode)
        traits = KernelTraits(desc.pure, desc.deterministic_state, desc.fusion_barrier, desc.direct_root, desc.parallel)
        scratch_entry = scratch_by_node.get(node.node_id)
        parameters = dict(_parameters(node))
        if scratch_entry:
            parameters["scratch_offset"] = str(scratch_entry[1].offset)
        nodes.append(KernelNode(node.node_id, (node.node_id,), node.opcode, node.children, _value_type(node.value_type.shape, node.value_type.width), state_by_node.get(node.node_id), None if scratch_entry is None else scratch_entry[0], tuple(sorted(parameters.items())), traits))
    counts = dict(plan.optimizations)
    diagnostics = Diagnostics(
        tuple((node.node_id, (node.node_id,)) for node in plan.nodes),
        constant_folds=counts.get("constant_folds", 0),
        stateless_cse=counts.get("common_subexpressions", 0) - counts.get("stateful_common_subexpressions", 0),
        stateful_cse=counts.get("stateful_common_subexpressions", 0),
        dead_nodes=counts.get("dead_nodes", 0), aliases_removed=counts.get("aliases_removed", 0),
        schedule=tuple(node.traits.parallel for node in nodes),
    )
    inputs = tuple(InputView(name, index, ValueType(ValueKind.INSTRUMENT_VECTOR)) for index, name in enumerate(program.input_names))
    output = GraphOutput(output_id, None, nodes[output_id].traits.direct_root)
    return FormulaIR(1, inputs, tuple(nodes), (output,), tuple(states), scratch, _align(state_end), scratch_bytes, diagnostics)
