from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from trading_dsl_engine.cpp_stream.python.lowering import GroupStage, Plan, Source, Stage, double_bits
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.ir.ops import GroupKeySpec


_CPP_TYPES = {
    "float32": "float",
    "float64": "double",
    "int32": "std::int32_t",
    "int64": "std::int64_t",
    "uint32": "std::uint32_t",
    "uint64": "std::uint64_t",
}


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
class SignedValueArg(CppType):
    value: int

    def render(self) -> str:
        return f"{self.value}LL"


@dataclass(frozen=True, slots=True)
class UnsignedValueArg(CppType):
    value: int

    def render(self) -> str:
        return f"{self.value}ULL"


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
class FloatArg(CppType):
    value: float

    def render(self) -> str:
        text = repr(float(self.value))
        if not any(char in text for char in ".eE"):
            text += ".0"
        return text + "f"


@dataclass(frozen=True, slots=True)
class TemplateType(CppType):
    name: str
    args: tuple[CppType, ...]

    def render(self) -> str:
        return f"{self.name}<" + ", ".join(arg.render() for arg in self.args) + ">"


def tmpl(name: str, *args: CppType) -> TemplateType:
    return TemplateType(name, tuple(args))


def _cpp_type(dtype: str) -> CppType:
    try:
        return Name(_CPP_TYPES[dtype])
    except KeyError as exc:
        raise ValueError(f"unsupported cpp_stream dtype {dtype!r}") from exc


def _literal_arg(source: Source) -> CppType:
    if source.dtype in {"int32", "int64"}:
        return SignedValueArg(int(source.value))
    if source.dtype in {"uint32", "uint64"}:
        return UnsignedValueArg(int(source.value))
    if source.dtype == "float32":
        return FloatArg(float(source.value))
    if source.dtype == "float64":
        return DoubleArg(float(source.value))
    raise ValueError(f"unsupported literal dtype {source.dtype!r}")


def _source_type(
    source: Source,
    *,
    n: int | CppType,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    if source.kind == "input":
        index = int(source.value)
        if input_types is None:
            cpp_type = _cpp_type(source.dtype)
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
            _cpp_type(source.dtype),
            BoolArg(source.row_scalar),
        )
    if source.kind == "literal":
        return tmpl("stackdsl::LiteralSrc", _literal_arg(source))
    raise ValueError(f"source kind {source.kind!r} is composite and must be flattened before codegen")


def _feature_list(
    sources: tuple[Source, ...],
    *,
    n: int | CppType,
    input_types: tuple[InputTypeSpec, ...] | None,
) -> CppType:
    return tmpl(
        "stackdsl::FeatureList",
        *(_source_type(source, n=n, input_types=input_types) for source in sources),
    )


def _dest_type(stage: Stage) -> CppType:
    if stage.out.slot is None:
        return Name("stackdsl::OutputDst")
    return tmpl("stackdsl::SlotDst", IntArg(stage.out.slot), _cpp_type(stage.dtype))


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
    """Render one operator type independent of grouped/direct execution."""
    stage_n: CppType = IntArg(1) if stage.lane_count == 1 else n
    inputs = tuple(_source_type(source, n=n, input_types=input_types) for source in stage.inputs)
    out = _dest_type(stage)
    if stage.kind == "copy":
        return tmpl("stackdsl::CopyNode", stage_n, inputs[0], out, execution)
    if stage.kind == "binary":
        return tmpl(
            "stackdsl::BinaryNode", stage_n, inputs[0], inputs[1], out,
            _cpp_type(stage.dtype), Name(_BINARY_POLICIES[stage.op_name or ""]), execution,
        )
    if stage.kind == "unary":
        return tmpl(
            "stackdsl::UnaryNode", stage_n, inputs[0], out,
            _cpp_type(stage.dtype), Name(_UNARY_POLICIES[stage.op_name or ""]), execution,
        )
    if stage.kind == "cat":
        return tmpl(
            "stackdsl::CatNode", n,
            _feature_list(stage.inputs, n=n, input_types=input_types), out, execution,
        )
    if stage.kind == "cumsum":
        return tmpl("stackdsl::CumsumNode", stage_n, inputs[0], out, execution)
    if stage.kind == "ewm":
        assert stage.ewm is not None
        return tmpl(
            "stackdsl::EwmNode", stage_n, inputs[0], out,
            UInt64Arg(double_bits(stage.ewm.span)), IntArg(stage.ewm.min_periods),
            BoolArg(stage.ewm.ignore_na), BoolArg(stage.ewm.adjust), execution,
        )
    if stage.kind == "xs_rank":
        return tmpl("stackdsl::XsRankNode", stage_n, inputs[0], out, execution)
    if stage.kind == "ridge":
        assert stage.ridge is not None
        assert stage.projection in {"beta", "preds"}
        assert stage.half_life is not None and stage.ridge_lambda is not None
        feature_count = stage.ridge.coefficient_width
        projection = "stackdsl::RidgeBetaProjection" if stage.projection == "beta" else "stackdsl::RidgePredsProjection"
        return tmpl(
            "stackdsl::RidgeNode", n,
            _feature_list(stage.inputs[:feature_count], n=n, input_types=input_types),
            _source_type(stage.inputs[feature_count], n=n, input_types=input_types),
            _source_type(stage.inputs[feature_count + 1], n=n, input_types=input_types),
            out,
            UInt64Arg(double_bits(stage.half_life)),
            UInt64Arg(double_bits(stage.ridge_lambda)),
            BoolArg(stage.ridge.nonneg),
            BoolArg(stage.ridge.is_stateful),
            Name(projection),
            execution,
        )
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
    execution = tmpl(
        "stackdsl::GroupedExecution",
        n,
        Name("Capacity"),
        Name("PartitionCount"),
    )
    stages = []
    for index, stage in enumerate(group.inner.stages):
        if stage.kind == "groupby":
            raise ValueError("nested groupby is not supported")
        stages.append(StageView(index, _stage_type(stage, n, execution, input_types=None).render()))
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
            "stackdsl::HashGroupResolver", IntArg(n), IntArg(group.capacity),
            IntArg(group.hash_capacity), *keys,
        )
    partitions = tmpl(
        "stackdsl::StaticPartitions",
        IntArg(n),
        *(IntArg(value) for value in group.partitions),
    )
    partition_count = max(group.partitions, default=0) + 1
    inner = tmpl(
        group_name,
        IntArg(n),
        Name(f"{resolver.render()}::capacity"),
        IntArg(partition_count),
    )
    return tmpl(
        "stackdsl::GroupByNode", IntArg(n), resolver, partitions, inner,
        _dest_type(stage), tmpl("stackdsl::KeyList", *keys),
        _source_list(group.feed_sources, n=n, input_types=input_types),
    )


def _environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(Path(__file__).with_name("templates")),
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
        input_types = tuple(InputTypeSpec("float64", n_instruments) for _ in range(plan.input_count))
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
        if stage.kind == "groupby":
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
            stages.append(
                StageView(
                    index,
                    _group_type(group_names[index], stage.group, stage, n_instruments, input_types).render(),
                    checked=True,
                )
            )
        else:
            stages.append(
                StageView(
                    index,
                    _stage_type(stage, n, direct_execution, input_types=input_types).render(),
                )
            )

    if plan.input_count == 0:
        raise ValueError("cpp_stream requires at least one file-backed input")

    text = _environment().get_template("runner.cpp.j2").render(
        n=n_instruments,
        input_count=plan.input_count,
        scratch_slots=plan.scratch_slots,
        output_row_width=plan.output_row_width,
        prefetch_rows=prefetch_rows,
        inputs=tuple(
            InputView(index, spec.cpp_type, spec.row_width)
            for index, spec in enumerate(input_types)
        ),
        stages=tuple(stages),
        inners=tuple(inners),
    )
    return GeneratedSource(text)
