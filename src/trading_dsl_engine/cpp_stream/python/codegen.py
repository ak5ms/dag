from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from trading_dsl_engine.cpp_stream.python.lowering import GroupStage, Plan, Source, Stage, double_bits


@dataclass(frozen=True, slots=True)
class GeneratedSource:
    text: str


class CppType:
    def render(self) -> str:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Name(CppType):
    value: str

    def render(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class IntArg(CppType):
    value: int

    def render(self) -> str:
        return str(self.value)


@dataclass(frozen=True, slots=True)
class UInt64Arg(CppType):
    value: int

    def render(self) -> str:
        return f"0x{self.value:016x}ULL"


@dataclass(frozen=True, slots=True)
class BoolArg(CppType):
    value: bool

    def render(self) -> str:
        return "true" if self.value else "false"


@dataclass(frozen=True, slots=True)
class DoubleArg(CppType):
    value: float

    def render(self) -> str:
        text = repr(float(self.value))
        return text if any(char in text for char in ".eE") else text + ".0"


@dataclass(frozen=True, slots=True)
class TemplateType(CppType):
    name: str
    args: tuple[CppType, ...]

    def render(self) -> str:
        return f"{self.name}<" + ", ".join(arg.render() for arg in self.args) + ">"


def tmpl(name: str, *args: CppType) -> TemplateType:
    return TemplateType(name, tuple(args))


def _source_type(source: Source) -> CppType:
    if source.kind == "input":
        return tmpl("stackdsl::InputSrc", IntArg(int(source.value)))
    if source.kind == "slot":
        return tmpl("stackdsl::SlotSrc", IntArg(int(source.value)))
    if source.kind == "literal":
        return tmpl("stackdsl::LiteralSrc", DoubleArg(float(source.value)))
    raise AssertionError(source.kind)


def _dest_type(stage: Stage) -> CppType:
    return Name("stackdsl::OutputDst") if stage.out.slot is None else tmpl("stackdsl::SlotDst", IntArg(stage.out.slot))


_BINARY_POLICIES = {
    "add": "stackdsl::AddOp",
    "sub": "stackdsl::SubOp",
    "mul": "stackdsl::MulOp",
    "div": "stackdsl::DivOp",
    "mod": "stackdsl::ModOp",
}

_UNARY_POLICIES = {
    "floor": "stackdsl::FloorOp",
}


def _stage_type(stage: Stage, n: CppType, execution: CppType) -> CppType:
    """Render one operator type independent of whether its plan is grouped.

    Grouping changes only ``execution``. There are no Grouped* operator names or
    per-operator grouped branches in code generation.
    """
    inputs = tuple(_source_type(source) for source in stage.inputs)
    out = _dest_type(stage)
    if stage.kind == "copy":
        return tmpl("stackdsl::CopyNode", n, inputs[0], out, execution)
    if stage.kind == "binary":
        return tmpl(
            "stackdsl::BinaryNode",
            n,
            inputs[0],
            inputs[1],
            out,
            Name(_BINARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "unary":
        return tmpl(
            "stackdsl::UnaryNode",
            n,
            inputs[0],
            out,
            Name(_UNARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "cumsum":
        return tmpl("stackdsl::CumsumNode", n, inputs[0], out, execution)
    if stage.kind == "ewm":
        assert stage.ewm is not None
        return tmpl(
            "stackdsl::EwmNode",
            n,
            inputs[0],
            out,
            UInt64Arg(double_bits(stage.ewm.span)),
            IntArg(stage.ewm.min_periods),
            BoolArg(stage.ewm.ignore_na),
            BoolArg(stage.ewm.adjust),
            execution,
        )
    if stage.kind == "xs_rank":
        return tmpl("stackdsl::XsRankNode", n, inputs[0], out, execution)
    if stage.kind == "groupby":
        raise AssertionError("groupby type is rendered separately")
    raise AssertionError(stage.kind)


def _source_list(sources: tuple[Source, ...]) -> CppType:
    return tmpl("stackdsl::SourceList", *(_source_type(source) for source in sources))


@dataclass(frozen=True, slots=True)
class StageView:
    index: int
    cpp_type: str
    checked: bool = False


@dataclass(frozen=True, slots=True)
class InnerView:
    name: str
    input_count: int
    scratch_slots: int
    stages: tuple[StageView, ...]


@dataclass(frozen=True, slots=True)
class InputView:
    index: int


def _inner_view(name: str, group: GroupStage) -> InnerView:
    n = Name("N")
    execution = tmpl("stackdsl::GroupedExecution", n, Name("Capacity"))
    stages: list[StageView] = []
    for index, stage in enumerate(group.inner.stages):
        if stage.kind == "groupby":
            raise ValueError("nested groupby is not supported")
        stages.append(StageView(index, _stage_type(stage, n, execution).render()))
    return InnerView(name, group.inner.input_count, group.inner.scratch_slots, tuple(stages))


def _group_type(group_name: str, group: GroupStage, stage: Stage, n: int) -> CppType:
    if not group.key_sources:
        resolver = tmpl("stackdsl::NoKeyResolver", IntArg(n))
    elif group.dense_cardinality is not None:
        resolver = tmpl(
            "stackdsl::DenseGroupResolver",
            IntArg(n),
            IntArg(group.dense_cardinality),
            IntArg(group.dense_offset),
        )
    else:
        resolver = tmpl(
            "stackdsl::HashGroupResolver",
            IntArg(n),
            IntArg(len(group.key_sources)),
            IntArg(group.capacity),
            IntArg(group.hash_capacity),
        )
    partitions = tmpl(
        "stackdsl::StaticPartitions",
        IntArg(n),
        *(IntArg(value) for value in group.partitions),
    )
    inner = tmpl(group_name, IntArg(n), Name(f"{resolver.render()}::capacity"))
    return tmpl(
        "stackdsl::GroupByNode",
        IntArg(n),
        resolver,
        partitions,
        inner,
        _dest_type(stage),
        _source_list(group.key_sources),
        _source_list(group.feed_sources),
    )


def _environment() -> Environment:
    template_dir = Path(__file__).with_name("templates")
    return Environment(
        loader=FileSystemLoader(template_dir),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_translation_unit(plan: Plan, *, n_instruments: int, prefetch_rows: int) -> GeneratedSource:
    group_names: dict[int, str] = {}
    inners: list[InnerView] = []
    for index, stage in enumerate(plan.stages):
        if stage.kind != "groupby":
            continue
        assert stage.group is not None
        name = f"CppStreamInner{index}"
        group_names[index] = name
        inners.append(_inner_view(name, stage.group))

    n = IntArg(n_instruments)
    direct_execution = tmpl("stackdsl::DirectExecution", n)
    stages: list[StageView] = []
    for index, stage in enumerate(plan.stages):
        if stage.kind == "groupby":
            assert stage.group is not None
            cpp_type = _group_type(group_names[index], stage.group, stage, n_instruments)
            stages.append(StageView(index, cpp_type.render(), checked=True))
        else:
            stages.append(StageView(index, _stage_type(stage, n, direct_execution).render()))

    if plan.input_count == 0:
        raise ValueError("cpp_stream requires at least one file-backed input")

    template = _environment().get_template("runner.cpp.j2")
    text = template.render(
        n=n_instruments,
        input_count=plan.input_count,
        scratch_slots=plan.scratch_slots,
        prefetch_rows=prefetch_rows,
        inputs=tuple(InputView(index) for index in range(plan.input_count)),
        stages=tuple(stages),
        inners=tuple(inners),
    )
    return GeneratedSource(text)
