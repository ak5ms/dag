from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping
import struct

from trading_dsl_engine.ir.ops import CumsumOp, EwmOp, GroupByOp, InputOp, LiteralOp, NaryOp, XsRankOp
from trading_dsl_engine.ir.program import Program


class CppStreamLoweringError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class Source:
    kind: str
    value: int | float

    def cpp(self) -> str:
        if self.kind == "input":
            return f"stackdsl::InputSrc<{int(self.value)}>"
        if self.kind == "slot":
            return f"stackdsl::SlotSrc<{int(self.value)}>"
        if self.kind == "literal":
            value = float(self.value)
            if not (value == value and abs(value) != float("inf")):
                raise CppStreamLoweringError("non-finite literals are not yet supported by cpp_stream")
            return f"stackdsl::LiteralSrc<{value!r}>"
        raise AssertionError(self.kind)

    @property
    def is_vector(self) -> bool:
        return self.kind != "literal"


@dataclass(frozen=True, slots=True)
class Dest:
    slot: int | None

    def cpp(self) -> str:
        return "stackdsl::OutputDst" if self.slot is None else f"stackdsl::SlotDst<{self.slot}>"


@dataclass(frozen=True, slots=True)
class Stage:
    kind: str
    inputs: tuple[Source, ...]
    out: Dest
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
    feed_sources: tuple[Source, ...]
    dense_cardinality: int | None
    dense_offset: int = 0


_BINARY_CPP = {"add": "stackdsl::AddOp", "sub": "stackdsl::SubOp", "mul": "stackdsl::MulOp", "div": "stackdsl::DivOp"}


def double_bits(value: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", float(value)))[0]


def op_cpp_type(stage: Stage, n_expr: str = "N", *, grouped_capacity_expr: str | None = None) -> str:
    ins = [source.cpp() for source in stage.inputs]
    out = stage.out.cpp()
    if stage.kind == "copy":
        return f"stackdsl::CopyNode<{n_expr}, {ins[0]}, {out}>"
    if stage.kind == "binary":
        return f"stackdsl::BinaryNode<{n_expr}, {ins[0]}, {ins[1]}, {out}, {_BINARY_CPP[stage.op_name or '']}>"
    if stage.kind == "cumsum":
        if grouped_capacity_expr is None:
            return f"stackdsl::CumsumNode<{n_expr}, {ins[0]}, {out}>"
        return f"stackdsl::GroupedCumsumNode<{n_expr}, {grouped_capacity_expr}, {ins[0]}, {out}>"
    if stage.kind == "ewm":
        assert stage.ewm is not None
        op = stage.ewm
        bits = double_bits(op.span)
        common = f"{n_expr}, {ins[0]}, {out}, 0x{bits:016x}ULL, {op.min_periods}, {str(op.ignore_na).lower()}, {str(op.adjust).lower()}"
        if grouped_capacity_expr is None:
            return f"stackdsl::EwmNode<{common}>"
        return f"stackdsl::GroupedEwmNode<{n_expr}, {grouped_capacity_expr}, {ins[0]}, {out}, 0x{bits:016x}ULL, {op.min_periods}, {str(op.ignore_na).lower()}, {str(op.adjust).lower()}>"
    if stage.kind == "xs_rank":
        if grouped_capacity_expr is None:
            return f"stackdsl::XsRankNode<{n_expr}, {ins[0]}, {out}>"
        return f"stackdsl::GroupedXsRankNode<{n_expr}, {grouped_capacity_expr}, {ins[0]}, {out}>"
    if stage.kind == "groupby":
        raise AssertionError("groupby type is rendered by codegen")
    raise AssertionError(stage.kind)


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
        raise CppStreamLoweringError(f"cpp_stream requires univ(...) to partition every column; missing {missing}")
    return tuple(result)


def _build_plan(program: Program, *, n_instruments: int, default_group_capacity: int, key_cardinalities: Mapping[str, int] | None, grouped: bool) -> Plan:
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
        if isinstance(op, InputOp):
            sources[node_id] = Source("input", op.input_index)
            continue
        if isinstance(op, LiteralOp):
            sources[node_id] = Source("literal", op.value)
            continue
        children = tuple(sources[child] for child in node.child_ids)
        is_root = node_id == root
        reusable: int | None = None
        if not is_root and not isinstance(op, GroupByOp):
            for child_id, source in zip(node.child_ids, children):
                if source.kind == "slot" and uses[child_id] == 1:
                    reusable = int(source.value)
                    break
        out = Dest(None if is_root else (reusable if reusable is not None else allocate()))
        if isinstance(op, NaryOp):
            if op.name not in _BINARY_CPP or op.arity != 2:
                raise CppStreamLoweringError(f"unsupported nary op {op.name!r}")
            stage = Stage("binary", children, out, op_name=op.name)
        elif isinstance(op, CumsumOp):
            if not children[0].is_vector:
                raise CppStreamLoweringError("cpp_stream cumsum requires vector input")
            stage = Stage("cumsum", children, out)
        elif isinstance(op, EwmOp):
            if not children[0].is_vector:
                raise CppStreamLoweringError("cpp_stream ewm requires vector input")
            stage = Stage("ewm", children, out, ewm=op)
        elif isinstance(op, XsRankOp):
            if not children[0].is_vector:
                raise CppStreamLoweringError("cpp_stream xs_rank requires vector input")
            stage = Stage("xs_rank", children, out)
        elif isinstance(op, GroupByOp):
            if grouped:
                raise CppStreamLoweringError("nested groupby is not supported")
            key_count = op.n_dynamic_keys
            key_sources = children[:key_count]
            feed_sources = children[key_count:]
            if any(not source.is_vector for source in key_sources + feed_sources):
                raise CppStreamLoweringError("cpp_stream groupby dynamic keys/lhs/captures must be vector-valued")
            capacity = op.capacity or default_group_capacity
            dense_cardinality: int | None = None
            if key_count == 1 and key_cardinalities:
                key_node = program.nodes[node.child_ids[0]].op
                if isinstance(key_node, InputOp) and key_node.name in key_cardinalities:
                    dense_cardinality = int(key_cardinalities[key_node.name])
                    if dense_cardinality <= 0:
                        raise CppStreamLoweringError("key cardinality must be > 0")
                    capacity = dense_cardinality + 1
            inner = _build_plan(op.inner_program, n_instruments=n_instruments, default_group_capacity=default_group_capacity, key_cardinalities=key_cardinalities, grouped=True)
            group_stage = GroupStage(inner=inner, partitions=_partitions(op.static_groups, n_instruments), capacity=capacity, hash_capacity=op.hash_capacity or 0, key_sources=key_sources, feed_sources=feed_sources, dense_cardinality=dense_cardinality)
            stage = Stage("groupby", children, out, group=group_stage)
        else:
            raise CppStreamLoweringError(f"unsupported IR op {type(op).__name__}")
        stages.append(stage)
        sources[node_id] = Source("slot", out.slot) if out.slot is not None else Source("slot", -1)
        for child_id, source in zip(node.child_ids, children):
            uses[child_id] -= 1
            if uses[child_id] == 0 and source.kind == "slot":
                slot = int(source.value)
                if slot != out.slot:
                    free_slots.append(slot)
    if isinstance(program.nodes[root].op, (InputOp, LiteralOp)):
        stages.append(Stage("copy", (sources[root],), Dest(None)))
    return Plan(stages=tuple(stages), scratch_slots=next_slot, input_count=len(program.input_names))


def lower_program(program: Program, *, n_instruments: int, default_group_capacity: int = 64, key_cardinalities: Mapping[str, int] | None = None) -> Plan:
    if n_instruments <= 0:
        raise CppStreamLoweringError("n_instruments must be > 0")
    if default_group_capacity <= 0:
        raise CppStreamLoweringError("default_group_capacity must be > 0")
    return _build_plan(program, n_instruments=n_instruments, default_group_capacity=default_group_capacity, key_cardinalities=key_cardinalities, grouped=False)
