from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
from contextlib import contextmanager
from time import perf_counter

from trading_dsl_engine.base.dsl import DSLFunctionRegistry
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.cpp_stream.python.codegen import render_translation_unit
from trading_dsl_engine.cpp_stream.python.compiler_support import (
    ReferencedSourceTypes,
    apply_input_key_hints,
    build_shared,
    infer_n,
    input_value_type,
    repair_value_types,
    row_scalar_analysis,
    validate_names,
)
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir
from trading_dsl_engine.cpp_stream.python.lowering_multi import lower_program
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.cpp_stream.python.output_projection import (
    optimize_public_projections,
)
from trading_dsl_engine.cpp_stream.python.outputs import build_output_layout
from trading_dsl_engine.cpp_stream.python.parallel import select_parallel_plan
from trading_dsl_engine.cpp_stream.python.runtime import (
    CompileMetrics,
    CppStreamRuntime,
)
from trading_dsl_engine.ir.ops import CvxpyProgramOp
from trading_dsl_engine.cpp_stream.python.sources import SourceValue


Formula = str | Expr
FormulaInput = Formula | list[Formula] | tuple[Formula, ...] | Mapping[object, object]


@contextmanager
def _measure(metrics: dict[str, float], stage: str):
    started = perf_counter()
    try:
        yield
    finally:
        metrics[stage] = metrics.get(stage, 0.0) + perf_counter() - started


def _flatten_formula_mapping(formula: Mapping[object, object]):
    flat: list[Formula] = []

    def visit(value, path):
        if isinstance(value, Mapping):
            if not value:
                raise ValueError(
                    f"formula mapping at {path or '<root>'} must not be empty"
                )
            return OrderedDict(
                (key, visit(child, path + (key,))) for key, child in value.items()
            )
        if not isinstance(value, (str, Expr)):
            raise TypeError(f"formula mapping leaf at {path} must be a string or Expr")
        index = len(flat)
        flat.append(value)
        return index

    structure = visit(formula, ())
    return tuple(flat), structure


def _compile_program(
    program,
    *,
    n_instruments: int,
    input_types: tuple[InputTypeSpec, ...],
    default_group_capacity: int,
    key_cardinalities: Mapping[str, int] | None,
    prefetch_rows: int,
    bound_sources: Mapping[str, SourceValue] | None,
    return_multiple: bool,
    result_structure=None,
    compile_stage_seconds: dict[str, float] | None = None,
    compile_started: float | None = None,
) -> CppStreamRuntime:
    metrics = compile_stage_seconds if compile_stage_seconds is not None else {}
    started = perf_counter() if compile_started is None else compile_started
    with _measure(metrics, "type_analysis"):
        program = repair_value_types(program)
        for root_id in program.outputs:
            root_kind = program.nodes[root_id].value_type.kind
            if root_kind == "object":
                raise ValueError(
                    "project object-valued operators before returning them from cpp_stream"
                )
            if root_kind not in {"scalar", "vector", "matrix", "fixed", "tensor"}:
                raise ValueError(f"unsupported cpp_stream root kind {root_kind!r}")
        program = apply_input_key_hints(program, input_types)
        scalar = row_scalar_analysis(program, input_types)
        row_scalar_nodes = frozenset(
            index for index in range(len(program.nodes)) if scalar(index)
        )
        layout = build_output_layout(program, n_instruments)
    with _measure(metrics, "lowering"):
        plan = optimize_public_projections(
            lower_program(
                program,
                n_instruments=n_instruments,
                default_group_capacity=default_group_capacity,
                key_cardinalities=key_cardinalities,
                row_scalar_nodes=row_scalar_nodes,
                input_dtypes=tuple(spec.dtype for spec in input_types),
            )
        )
    with _measure(metrics, "parallel_planning"):
        parallel_plan = select_parallel_plan(plan, n_instruments, output_layout=layout)
    generated_programs = []
    seen_programs = set()
    for node in program.nodes:
        if not isinstance(node.op, CvxpyProgramOp):
            continue
        artifact = node.op.program
        key = (str(artifact.root), artifact.class_name, artifact.prefix)
        if key not in seen_programs:
            seen_programs.add(key)
            generated_programs.append(artifact)
    if len(generated_programs) > 1:
        raise ValueError(
            "one cpp_stream translation unit currently supports one distinct "
            "generated optimizer artifact; reuse it for multiple projections"
        )
    native_headers = tuple(
        artifact.instance_header.name for artifact in generated_programs
    )
    with _measure(metrics, "code_generation"):
        generated = render_translation_unit(
            plan,
            n_instruments=n_instruments,
            prefetch_rows=prefetch_rows,
            input_types=input_types,
            output_layout=layout,
            native_headers=native_headers,
        )
    include_dirs = tuple(
        directory
        for artifact in generated_programs
        for directory in artifact.include_dirs
    )
    link_files = tuple(
        path for artifact in generated_programs for path in artifact.link_files
    )
    fingerprint_files = tuple(
        path for artifact in generated_programs for path in artifact.fingerprint_files
    )
    native_metrics: dict[str, float | bool] = {}
    with _measure(metrics, "native_build"):
        library_path, cpp_path = build_shared(
            generated.text,
            extra_include_dirs=include_dirs,
            extra_link_files=link_files,
            extra_fingerprint_files=fingerprint_files,
            metrics=native_metrics,
        )
    for name in ("dependency_fingerprint_seconds", "native_compile_seconds"):
        metrics[name] = float(native_metrics.get(name, 0.0))
    runtime_started = perf_counter()
    runtime = CppStreamRuntime(
        program=program,
        plan=plan,
        library_path=library_path,
        generated_cpp=cpp_path,
        n_instruments=n_instruments,
        input_types=input_types,
        bound_sources=bound_sources,
        parallel_plan=parallel_plan,
        output_layout=layout,
        return_multiple=return_multiple,
        result_structure=result_structure,
        compile_metrics=None,
    )
    metrics["runtime_setup"] = perf_counter() - runtime_started
    runtime.compile_metrics = CompileMetrics(
        stage_seconds=dict(metrics),
        total_seconds=perf_counter() - started,
        native_cache_hit=bool(native_metrics.get("native_cache_hit", False)),
    )
    return runtime


def compile_formula(
    formula: FormulaInput,
    data: Mapping[str, SourceValue] | None = None,
    *,
    n_instruments: int | None = None,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    prefetch_rows: int = 16,
    input_types: Mapping[str, InputTypeSpec] | None = None,
) -> CppStreamRuntime:
    """Compile one or many formulas into one CSE'd native streaming program."""

    compile_started = perf_counter()
    compile_stage_seconds: dict[str, float] = {}
    if prefetch_rows < 0:
        raise ValueError("prefetch_rows must be >= 0")
    result_structure = None
    if isinstance(formula, Mapping):
        formula, result_structure = _flatten_formula_mapping(formula)
        return_multiple = True
    else:
        return_multiple = isinstance(formula, (list, tuple))
    if return_multiple:
        formula = tuple(formula)
        if not formula:
            raise ValueError("compile_formula requires at least one formula")

    if data is not None:
        referenced_types = ReferencedSourceTypes(
            data,
            input_types,
            n_instruments,
        )
        with _measure(compile_stage_seconds, "frontend"):
            program = compile_ir(
                formula,
                dsl_registry=dsl_registry,
                column_names=column_names,
                input_value_types=referenced_types,
                n_instruments=n_instruments,
            )
        validate_names(program, data, what="source")
        infos = referenced_types.infos_for(program.input_names)
        if infos and len({info.rows for info in infos.values()}) != 1:
            details = {name: info.rows for name, info in infos.items()}
            raise ValueError(f"cpp_stream sources have different row counts: {details}")
        n = infer_n(infos, n_instruments)
        # Rebuild with exact N so all tensor and public-output extents become
        # compile-time constants before lowering and Jinja rendering.
        if n_instruments is None:
            with _measure(compile_stage_seconds, "frontend"):
                program = compile_ir(
                    formula,
                    dsl_registry=dsl_registry,
                    column_names=column_names,
                    input_value_types={
                        name: input_value_type(info.input_type, n)
                        for name, info in infos.items()
                    },
                    n_instruments=n,
                )
        validate_names(program, data, what="source")
        ordered = tuple(infos[name].input_type for name in program.input_names)
        bound_sources: Mapping[str, SourceValue] | None = {
            name: data[name] for name in program.input_names
        }
    else:
        if n_instruments is None:
            raise ValueError(
                "n_instruments is required when compile_formula is called without data"
            )
        n = int(n_instruments)
        if n <= 0:
            raise ValueError(f"invalid n_instruments={n}")
        input_value_types = (
            {name: input_value_type(spec, n) for name, spec in input_types.items()}
            if input_types is not None
            else None
        )
        with _measure(compile_stage_seconds, "frontend"):
            program = compile_ir(
                formula,
                dsl_registry=dsl_registry,
                column_names=column_names,
                input_value_types=input_value_types,
                n_instruments=n,
            )
        if input_types is None:
            ordered = tuple(InputTypeSpec("float64", n) for _ in program.input_names)
        else:
            validate_names(program, input_types, what="input_types")
            ordered = tuple(input_types[name] for name in program.input_names)
        bound_sources = None

    return _compile_program(
        program,
        n_instruments=n,
        input_types=ordered,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        prefetch_rows=prefetch_rows,
        bound_sources=bound_sources,
        return_multiple=return_multiple,
        result_structure=result_structure,
        compile_stage_seconds=compile_stage_seconds,
        compile_started=compile_started,
    )


__all__ = ["Formula", "FormulaInput", "compile_formula"]
