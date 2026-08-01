from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from trading_dsl_engine.cpp_stream.python.lowering import GroupStage, Plan, Source, Stage, double_bits
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.ir.ops import GroupKeySpec


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


def _source_type(
    source: Source,
    *,
    n: int | CppType,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    if source.kind == "input":
        index = int(source.value)
        if input_types is None:
            cpp_type = Name("double")
            row_width = n if isinstance(n, CppType) else IntArg(n)
        else:
            spec = input_types[index]
            cpp_type = Name(spec.cpp_type)
            row_width = IntArg(spec.row_width)
        return tmpl("stackdsl::InputSrc", IntArg(index), cpp_type, row_width)
    if source.kind == "slot":
        return tmpl(
            "stackdsl::SlotSrc",
            IntArg(int(source.value)),
            BoolArg(source.row_scalar),
        )
    if source.kind == "literal":
        return tmpl("stackdsl::LiteralSrc", DoubleArg(float(source.value)))
    raise AssertionError(source.kind)


def _dest_type(stage: Stage) -> CppType:
    return Name("stackdsl::OutputDst") if stage.out.slot is None else tmpl(
        "stackdsl::SlotDst", IntArg(stage.out.slot)
    )


_BINARY_POLICIES = {
    "add": "stackdsl::AddOp",
    "sub": "stackdsl::SubOp",
    "mul": "stackdsl::MulOp",
    "div": "stackdsl::DivOp",
    "mod": "stackdsl::ModOp",
}

_UNARY_POLICIES = {"floor": "stackdsl::FloorOp"}


def _stage_type(
    stage: Stage,
    n: CppType,
    execution: CppType,
    *,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    """Render one operator type independent of whether its plan is grouped."""
    stage_n: CppType = IntArg(1) if stage.lane_count == 1 else n
    inputs = tuple(
        _source_type(source, n=n, input_types=input_types)
        for source in stage.inputs
    )
    out = _dest_type(stage)
    if stage.kind == "copy":
        return tmpl("stackdsl::CopyNode", stage_n, inputs[0], out, execution)
    if stage.kind == "binary":
        return tmpl(
            "stackdsl::BinaryNode",
            stage_n,
            inputs[0],
            inputs[1],
            out,
            Name(_BINARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "unary":
        return tmpl(
            "stackdsl::UnaryNode",
            stage_n,
            inputs[0],
            out,
            Name(_UNARY_POLICIES[stage.op_name or ""]),
            execution,
        )
    if stage.kind == "cumsum":
        return tmpl("stackdsl::CumsumNode", stage_n, inputs[0], out, execution)
    if stage.kind == "ewm":
        assert stage.ewm is not None
        return tmpl(
            "stackdsl::EwmNode",
            stage_n,
            inputs[0],
            out,
            UInt64Arg(double_bits(stage.ewm.span)),
            IntArg(stage.ewm.min_periods),
            BoolArg(stage.ewm.ignore_na),
            BoolArg(stage.ewm.adjust),
            execution,
        )
    if stage.kind == "xs_rank":
        return tmpl("stackdsl::XsRankNode", stage_n, inputs[0], out, execution)
    if stage.kind == "groupby":
        raise AssertionError("groupby type is rendered separately")
    raise AssertionError(stage.kind)


def _source_list(
    sources: tuple[Source, ...],
    *,
    n: int | CppType,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    return tmpl(
        "stackdsl::SourceList",
        *(_source_type(source, n=n, input_types=input_types) for source in sources),
    )


def _key_type(
    source: Source,
    spec: GroupKeySpec,
    *,
    n: int,
    input_types: tuple[InputTypeSpec, ...],
) -> CppType:
    return tmpl(
        "stackdsl::KeySpec",
        _source_type(source, n=n, input_types=input_types),
        IntArg(0 if spec.num_keys is None else int(spec.num_keys)),
        IntArg(int(spec.offset)),
        BoolArg(bool(spec.row_scalar)),
    )


def _key_list(
    group: GroupStage,
    *,
    n: int,
    input_types: tuple[InputTypeSpec, ...],
) -> tuple[CppType, ...]:
    if len(group.key_sources) != len(group.key_specs):
        raise ValueError("group key source/spec length mismatch")
    return tuple(
        _key_type(source, spec, n=n, input_types=input_types)
        for source, spec in zip(group.key_sources, group.key_specs)
    )


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
    cpp_type: str
    row_width: int


def _inner_view(name: str, group: GroupStage) -> InnerView:
    n = Name("N")
    execution = tmpl("stackdsl::GroupedExecution", n, Name("Capacity"))
    stages: list[StageView] = []
    for index, stage in enumerate(group.inner.stages):
        if stage.kind == "groupby":
            raise ValueError("nested groupby is not supported")
        stages.append(
            StageView(index, _stage_type(stage, n, execution, input_types=None).render())
        )
    return InnerView(name, group.inner.input_count, group.inner.scratch_slots, tuple(stages))


def _group_type(
    group_name: str,
    group: GroupStage,
    stage: Stage,
    n: int,
    input_types: tuple[InputTypeSpec, ...],
) -> CppType:
    keys = _key_list(group, n=n, input_types=input_types)
    if not keys:
        resolver = tmpl("stackdsl::NoKeyResolver", IntArg(n))
    elif group.dense:
        resolver = tmpl("stackdsl::DenseTupleGroupResolver", IntArg(n), *keys)
    else:
        resolver = tmpl(
            "stackdsl::HashGroupResolver",
            IntArg(n),
            IntArg(group.capacity),
            IntArg(group.hash_capacity),
            *keys,
        )
    partitions = tmpl(
        "stackdsl::StaticPartitions",
        IntArg(n),
        *(IntArg(value) for value in group.partitions),
    )
    inner = tmpl(group_name, IntArg(n), Name(f"{resolver.render()}::capacity"))
    key_list = tmpl("stackdsl::KeyList", *keys)
    return tmpl(
        "stackdsl::GroupByNode",
        IntArg(n),
        resolver,
        partitions,
        inner,
        _dest_type(stage),
        key_list,
        _source_list(group.feed_sources, n=n, input_types=input_types),
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


def render_translation_unit(
    plan: Plan,
    *,
    n_instruments: int,
    prefetch_rows: int,
    input_types: tuple[InputTypeSpec, ...] | None = None,
) -> GeneratedSource:
    if input_types is None:
        input_types = tuple(
            InputTypeSpec("float64", n_instruments)
            for _ in range(plan.input_count)
        )
    if len(input_types) != plan.input_count:
        raise ValueError("input type count does not match the compiled program")
    for spec in input_types:
        if spec.row_width not in (1, n_instruments):
            raise ValueError(
                f"input row width must be 1 or n_instruments={n_instruments}, got {spec.row_width}"
            )

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
            cpp_type = _group_type(
                group_names[index], stage.group, stage, n_instruments, input_types
            )
            stages.append(StageView(index, cpp_type.render(), checked=True))
        else:
            stages.append(
                StageView(
                    index,
                    _stage_type(
                        stage,
                        n,
                        direct_execution,
                        input_types=input_types,
                    ).render(),
                )
            )

    if plan.input_count == 0:
        raise ValueError("cpp_stream requires at least one file-backed input")

    template = _environment().get_template("runner.cpp.j2")
    text = template.render(
        n=n_instruments,
        input_count=plan.input_count,
        scratch_slots=plan.scratch_slots,
        prefetch_rows=prefetch_rows,
        inputs=tuple(
            InputView(index, spec.cpp_type, spec.row_width)
            for index, spec in enumerate(input_types)
        ),
        stages=tuple(stages),
        inners=tuple(inners),
    )
    return GeneratedSource(text)
