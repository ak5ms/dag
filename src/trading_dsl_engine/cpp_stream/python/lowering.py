from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping
import struct

from trading_dsl_engine.ir.ops import (
    CumsumOp,
    EwmOp,
    GroupByOp,
    GroupKeySpec,
    InputOp,
    LiteralOp,
    NaryOp,
    XsRankOp,
)
from trading_dsl_engine.ir.program import Program


class CppStreamLoweringError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Source:
    kind: str
    value: int | float
    row_scalar: bool = False

    @property
    def is_vector(self) -> bool:
        return self.kind != "literal"


@dataclass(frozen=True, slots=True)
class Dest:
    slot: int | None


@dataclass(frozen=True, slots=True)
class Stage:
    kind: str
    inputs: tuple[Source, ...]
    out: Dest
    lane_count: int
    op_name: str | None = None
    ewm: EwmOp | None = None
    group: "GroupStage | None" = None


@dataclass(frozen=True, slots=True)
class Plan:
    stages: tuple[Stage, ...]
    scratch_slots: int
    input_count: int


@dataclass(frozen=True, slots=True)
class GroupStage:
    inner: Plan
    partitions: tuple[int, ...]
    capacity: int
    hash_capacity: int
    key_sources: tuple[Source, ...]
    key_specs: tuple[GroupKeySpec, ...]
    feed_sources: tuple[Source, ...]
    dense: bool


_BINARY_CPP = {
    "add": "stackdsl::AddOp",
    "sub": "stackdsl::SubOp",
    "mul": "stackdsl::MulOp",
    "div": "stackdsl::DivOp",
    "mod": "stackdsl::ModOp",
}
_UNARY_CPP = {"floor": "stackdsl::FloorOp"}


def double_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def _partitions(groups: tuple[tuple[int, ...], ...] | None, n: int) -> tuple[int, ...]:
    if groups is None:
        return (0,) * n
    result = [-1] * n
    for group_id, group in enumerate(groups):
        for lane in group:
            if lane < 0 or lane >= n:
                raise CppStreamLoweringError(f"univ column {lane} is outside n_instruments={n}")
            if result[lane] != -1:
                raise CppStreamLoweringError(f"univ column {lane} appears in multiple groups")
            result[lane] = group_id
    missing = [i for i, value in enumerate(result) if value < 0]
    if missing:
        raise CppStreamLoweringError(
            f"cpp_stream requires univ(...) to partition every column; missing {missing}"
        )
    return tuple(result)


def _resolved_key_specs(
    program: Program,
    node_child_ids: tuple[int, ...],
    op: GroupByOp,
    key_cardinalities: Mapping[str, int] | None,
) -> tuple[GroupKeySpec, ...]:
    specs = list(op.key_specs)
    if key_cardinalities:
        for index, spec in enumerate(specs):
            if spec.num_keys is not None:
                continue
            key_op = program.nodes[node_child_ids[index]].op
            if isinstance(key_op, InputOp) and key_op.name in key_cardinalities:
                cardinality = int(key_cardinalities[key_op.name])
                if cardinality <= 0:
                    raise CppStreamLoweringError("key cardinality must be > 0")
                specs[index] = replace(spec, num_keys=cardinality)
    return tuple(specs)


def _dense_capacity(specs: tuple[GroupKeySpec, ...]) -> int:
    capacity = 1
    for spec in specs:
        assert spec.num_keys is not None
        capacity *= int(spec.num_keys) + 1
    if capacity > 65535:
        raise CppStreamLoweringError(
            f"dense composite key requires {capacity} slots, exceeding uint16 capacity"
        )
    return capacity


def _build_plan(
    program: Program,
    *,
    n_instruments: int,
    default_group_capacity: int,
    key_cardinalities: Mapping[str, int] | None,
    grouped: bool,
    row_scalar_nodes: frozenset[int],
) -> Plan:
    uses = [0] * len(program.nodes)
    root = program.output_id
    for node in program.nodes:
        for child in node.child_ids:
            uses[child] += 1
    uses[root] += 1
    sources: dict[int, Source] = {}
    free_slots: list[int] = []
    next_slot = 0
    stages: list[Stage] = []

    def allocate() -> int:
        nonlocal next_slot
        if free_slots:
            return free_slots.pop()
        slot = next_slot
        next_slot += 1
        return slot

    for node_id, node in enumerate(program.nodes):
        op = node.op
        row_scalar = node_id in row_scalar_nodes
        lane_count = 1 if row_scalar else n_instruments
        if isinstance(op, InputOp):
            sources[node_id] = Source("input", op.input_index, row_scalar=row_scalar)
            continue
        if isinstance(op, LiteralOp):
            sources[node_id] = Source("literal", op.value, row_scalar=True)
            continue

        children = tuple(sources[child] for child in node.child_ids)
        is_root = node_id == root
        reusable: int | None = None
        if not is_root and not isinstance(op, GroupByOp):
            for child_id, source in zip(node.child_ids, children):
                if (
                    source.kind == "slot"
                    and uses[child_id] == 1
                    and (row_scalar or not source.row_scalar)
                ):
                    reusable = int(source.value)
                    break
        out = Dest(None if is_root else (reusable if reusable is not None else allocate()))

        if isinstance(op, NaryOp):
            if op.arity == 2 and op.name in _BINARY_CPP:
                stage = Stage("binary", children, out, lane_count, op_name=op.name)
            elif op.arity == 1 and op.name in _UNARY_CPP:
                stage = Stage("unary", children, out, lane_count, op_name=op.name)
            else:
                raise CppStreamLoweringError(f"unsupported nary op {op.name!r}/{op.arity}")
        elif isinstance(op, CumsumOp):
            if not children[0].is_vector:
                raise CppStreamLoweringError("cpp_stream cumsum requires vector input")
            stage = Stage("cumsum", children, out, lane_count)
        elif isinstance(op, EwmOp):
            if not children[0].is_vector:
                raise CppStreamLoweringError("cpp_stream ewm requires vector input")
            stage = Stage("ewm", children, out, lane_count, ewm=op)
        elif isinstance(op, XsRankOp):
            if not children[0].is_vector:
                raise CppStreamLoweringError("cpp_stream xs_rank requires vector input")
            stage = Stage("xs_rank", children, out, lane_count)
        elif isinstance(op, GroupByOp):
            if grouped:
                raise CppStreamLoweringError("nested groupby is not supported")
            key_count = op.n_dynamic_keys
            key_sources = children[:key_count]
            feed_sources = children[key_count:]
            if any(not source.is_vector for source in key_sources + feed_sources):
                raise CppStreamLoweringError(
                    "cpp_stream groupby dynamic keys/lhs/captures must be vector-valued"
                )
            key_specs = _resolved_key_specs(program, node.child_ids, op, key_cardinalities)
            dense = bool(key_specs) and all(spec.num_keys is not None for spec in key_specs)
            capacity = _dense_capacity(key_specs) if dense else (op.capacity or default_group_capacity)
            inner = _build_plan(
                op.inner_program,
                n_instruments=n_instruments,
                default_group_capacity=default_group_capacity,
                key_cardinalities=key_cardinalities,
                grouped=True,
                row_scalar_nodes=frozenset(),
            )
            group_stage = GroupStage(
                inner=inner,
                partitions=_partitions(op.static_groups, n_instruments),
                capacity=capacity,
                hash_capacity=op.hash_capacity or 0,
                key_sources=key_sources,
                key_specs=key_specs,
                feed_sources=feed_sources,
                dense=dense,
            )
            stage = Stage("groupby", children, out, n_instruments, group=group_stage)
        else:
            raise CppStreamLoweringError(f"unsupported IR op {type(op).__name__}")

        stages.append(stage)
        sources[node_id] = Source(
            "slot",
            out.slot if out.slot is not None else -1,
            row_scalar=row_scalar,
        )
        for child_id, source in zip(node.child_ids, children):
            uses[child_id] -= 1
            if uses[child_id] == 0 and source.kind == "slot":
                slot = int(source.value)
                if slot != out.slot:
                    free_slots.append(slot)

    if isinstance(program.nodes[root].op, (InputOp, LiteralOp)):
        stages.append(Stage("copy", (sources[root],), Dest(None), n_instruments))
    return Plan(stages=tuple(stages), scratch_slots=next_slot, input_count=len(program.input_names))


def lower_program(
    program: Program,
    *,
    n_instruments: int,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    row_scalar_nodes: frozenset[int] | None = None,
) -> Plan:
    if n_instruments <= 0:
        raise CppStreamLoweringError("n_instruments must be > 0")
    if default_group_capacity <= 0:
        raise CppStreamLoweringError("default_group_capacity must be > 0")
    return _build_plan(
        program,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        grouped=False,
        row_scalar_nodes=row_scalar_nodes or frozenset(),
    )
