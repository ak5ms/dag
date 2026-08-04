from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
from typing import Mapping

import includeigen

from trading_dsl_engine.base.dsl import DSLFunctionRegistry
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.cpp_stream.python.codegen import render_translation_unit
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir
from trading_dsl_engine.cpp_stream.python.lowering import lower_program
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.cpp_stream.python.parallel import select_parallel_plan
from trading_dsl_engine.cpp_stream.python.runtime import CppStreamRuntime
from trading_dsl_engine.cpp_stream.python.sources import (
    SourceInfo,
    SourceValue,
    inspect_source_mapping,
)
from trading_dsl_engine.ir.ops import (
    CumsumOp,
    EwmOp,
    FFillOp,
    GroupByOp,
    InputOp,
    NaryOp,
    ShiftOp,
)
from trading_dsl_engine.ir.program import Node, Program
from trading_dsl_engine.ir.types import SCALAR, ValueType, tensor


_LANE_STATE_OPS = (CumsumOp, FFillOp, ShiftOp, EwmOp)


def _cpp_root() -> Path:
    return Path(__file__).resolve().parents[1] / "cpp"


def _eigen_include() -> Path:
    return Path(includeigen.get_include()).resolve()


def _cache_root() -> Path:
    configured = os.environ.get("TRADING_DSL_ENGINE_CPP_STREAM_CACHE")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache/trading_dsl_engine/cpp_stream"
    )


def _compiler() -> str:
    requested = os.environ.get("CXX", "g++")
    compiler = shutil.which(requested)
    if compiler is None:
        raise RuntimeError(
            f"cpp_stream requires a C++20 compiler; could not find {requested!r}"
        )
    return compiler


def _flags() -> tuple[list[str], list[str]]:
    if os.name == "nt":
        raise RuntimeError("cpp_stream currently targets POSIX/Linux")
    compile_flags = [
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        "-DEIGEN_NO_DEBUG",
        "-fPIC",
        "-shared",
        "-pthread",
        "-fno-math-errno",
        "-funroll-loops",
        "-DEIGEN_DONT_PARALLELIZE",
        "-DEIGEN_MPL2_ONLY",
    ]
    if sys.platform.startswith("linux"):
        compile_flags.append("-D_GNU_SOURCE")
    if os.environ.get("TRADING_DSL_ENGINE_CPP_NATIVE", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        compile_flags += ["-march=native", "-mtune=native"]
    if os.environ.get("TRADING_DSL_ENGINE_CPP_LTO", "1").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }:
        compile_flags.append("-flto")
    compile_flags += shlex.split(
        os.environ.get("TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS", "")
    )
    link_flags = [
        "-Wl,-O3",
        "-pthread",
        *shlex.split(
            os.environ.get("TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS", "")
        ),
    ]
    return compile_flags, link_flags


def _build_shared(source: str) -> tuple[Path, Path]:
    compiler = _compiler()
    compile_flags, link_flags = _flags()
    digest = hashlib.sha256(source.encode())
    for header in sorted(_cpp_root().rglob("*.hpp")):
        digest.update(header.relative_to(_cpp_root()).as_posix().encode())
        digest.update(header.read_bytes())
    eigen_include = _eigen_include()
    digest.update(str(eigen_include).encode())
    eigen_macros = eigen_include / "Eigen" / "src" / "Core" / "util" / "Macros.h"
    if eigen_macros.is_file():
        digest.update(eigen_macros.read_bytes())
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=False
    )
    digest.update((version.stdout or version.stderr).encode())
    digest.update("\0".join((*compile_flags, *link_flags)).encode())
    digest.update(
        f"{platform.platform()}|{platform.machine()}|{sys.implementation.cache_tag}".encode()
    )
    build_dir = _cache_root() / digest.hexdigest()
    cpp_path = build_dir / "formula.cpp"
    so_path = build_dir / "formula.so"
    if so_path.is_file():
        return so_path, cpp_path
    build_dir.mkdir(parents=True, exist_ok=True)
    temporary_cpp = build_dir / f"formula.{os.getpid()}.cpp"
    temporary_so = build_dir / f"formula.{os.getpid()}.so"
    temporary_cpp.write_text(source)
    command = [
        compiler,
        *compile_flags,
        f"-I{_cpp_root()}",
        f"-I{_eigen_include()}",
        str(temporary_cpp),
        *link_flags,
        "-o",
        str(temporary_so),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            "cpp_stream native compilation failed\n"
            + " ".join(command)
            + "\n"
            + result.stdout
            + result.stderr
        )
    temporary_cpp.replace(cpp_path)
    temporary_so.replace(so_path)
    return so_path, cpp_path


def _input_value_type(spec: InputTypeSpec, n_instruments: int) -> ValueType:
    shape = tuple(spec.row_shape or ())
    if not shape:
        return SCALAR
    logical_shape = (
        (None,) + shape[1:] if shape[0] == n_instruments else shape
    )
    return tensor(logical_shape, dtype=spec.dtype)


def _broadcast_shapes(shapes: tuple[tuple[int | None, ...], ...]) -> tuple[int | None, ...]:
    rank = max((len(shape) for shape in shapes), default=0)
    result: list[int | None] = []
    for output_axis in range(rank):
        aligned: list[int | None] = []
        for shape in shapes:
            input_axis = output_axis - (rank - len(shape))
            aligned.append(1 if input_axis < 0 else shape[input_axis])
        chosen: int | None = 1
        for extent in aligned:
            if extent == 1:
                continue
            if chosen == 1:
                chosen = extent
            elif extent != chosen:
                raise TypeError(
                    f"operands could not be broadcast together in cpp_stream: {shapes!r}"
                )
        result.append(chosen)
    return tuple(result)


def _broadcast_result_type(name: str, children: tuple[Node, ...]) -> ValueType:
    del name
    if any(child.value_type.kind == "object" for child in children):
        raise TypeError("elementwise operators cannot consume object values")
    shapes = tuple(child.value_type.logical_shape for child in children)
    dtype = (
        children[0].value_type.dtype
        if len({child.value_type.dtype for child in children}) == 1
        else "float64"
    )
    return tensor(_broadcast_shapes(shapes), dtype=dtype)


def _repair_value_types(program: Program) -> Program:
    """Propagate tensor broadcasting and shape-preserving temporal operators."""

    nodes: list[Node] = []
    for original in program.nodes:
        node = original
        op = node.op
        if isinstance(op, GroupByOp):
            inner = _repair_value_types(op.inner_program)
            op = replace(op, inner_program=inner)
            node = replace(
                node,
                op=op,
                value_type=inner.nodes[inner.output_id].value_type,
            )

        child_nodes = tuple(nodes[child_id] for child_id in node.child_ids)
        if isinstance(op, NaryOp):
            node = replace(
                node,
                value_type=_broadcast_result_type(op.name, child_nodes),
            )
        elif isinstance(op, _LANE_STATE_OPS):
            if len(child_nodes) != 1:
                raise TypeError(
                    f"{type(op).__name__} expected one child, got {len(child_nodes)}"
                )
            child_type = child_nodes[0].value_type
            if child_type.kind == "object":
                raise TypeError(
                    f"{type(op).__name__} cannot consume object values"
                )
            node = replace(node, value_type=child_type)
        nodes.append(node)
    return replace(program, nodes=tuple(nodes))


def _row_scalar_analysis(
    program: Program,
    input_types: tuple[InputTypeSpec, ...],
):
    """Return whether an outer-plan node has one value for the complete row."""

    memo: dict[int, bool] = {}

    def visit(node_id: int) -> bool:
        if node_id in memo:
            return memo[node_id]
        node = program.nodes[node_id]
        if isinstance(node.op, GroupByOp):
            value = False
        elif node.value_type.kind == "scalar":
            value = True
        elif isinstance(node.op, InputOp):
            value = input_types[node.op.input_index].row_scalar
        else:
            value = False
        memo[node_id] = value
        return value

    return visit


def _apply_input_key_hints(
    program: Program,
    input_types: tuple[InputTypeSpec, ...],
) -> Program:
    is_row_scalar = _row_scalar_analysis(program, input_types)
    nodes = []
    for node in program.nodes:
        if not isinstance(node.op, GroupByOp):
            nodes.append(node)
            continue
        specs = []
        for index, spec in enumerate(node.op.key_specs):
            child_id = node.child_ids[index]
            child_op = program.nodes[child_id].op
            row_scalar = (
                is_row_scalar(child_id)
                if spec.row_scalar is None
                else spec.row_scalar
            )
            dtype = spec.dtype
            if isinstance(child_op, InputOp):
                actual = input_types[child_op.input_index].dtype
                if dtype is None:
                    dtype = actual
                elif dtype != actual:
                    raise TypeError(
                        f"Key dtype {dtype!r} does not match input "
                        f"{child_op.name!r} dtype {actual!r}"
                    )
            specs.append(replace(spec, row_scalar=row_scalar, dtype=dtype))
        nodes.append(replace(node, op=replace(node.op, key_specs=tuple(specs))))
    return replace(program, nodes=tuple(nodes))


def _compile_program(
    program: Program,
    *,
    n_instruments: int,
    input_types: tuple[InputTypeSpec, ...],
    default_group_capacity: int,
    key_cardinalities: Mapping[str, int] | None,
    prefetch_rows: int,
    bound_sources: Mapping[str, SourceValue] | None,
) -> CppStreamRuntime:
    program = _repair_value_types(program)
    root_kind = program.nodes[program.output_id].value_type.kind
    if root_kind == "object":
        raise ValueError(
            "project Ridge with get_beta(...) or get_preds(...) before output"
        )
    if root_kind not in {"scalar", "vector", "matrix", "fixed", "tensor"}:
        raise ValueError(f"unsupported cpp_stream root kind {root_kind!r}")
    program = _apply_input_key_hints(program, input_types)
    scalar = _row_scalar_analysis(program, input_types)
    row_scalar_nodes = frozenset(
        index for index in range(len(program.nodes)) if scalar(index)
    )
    plan = lower_program(
        program,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        row_scalar_nodes=row_scalar_nodes,
        input_dtypes=tuple(spec.dtype for spec in input_types),
    )
    parallel_plan = select_parallel_plan(plan, n_instruments)
    generated = render_translation_unit(
        plan,
        n_instruments=n_instruments,
        prefetch_rows=prefetch_rows,
        input_types=input_types,
    )
    library_path, cpp_path = _build_shared(generated.text)
    return CppStreamRuntime(
        program=program,
        plan=plan,
        library_path=library_path,
        generated_cpp=cpp_path,
        n_instruments=n_instruments,
        input_types=input_types,
        bound_sources=bound_sources,
        parallel_plan=parallel_plan,
    )


def _infer_n(infos: Mapping[str, SourceInfo], requested: int | None) -> int:
    if requested is None:
        vector_widths = {
            info.input_type.row_shape[0]
            for info in infos.values()
            if len(info.input_type.row_shape or ()) == 1
            and info.input_type.row_shape[0] > 1
        }
        if len(vector_widths) != 1:
            raise ValueError(
                "pass n_instruments when source metadata has no unique rank-one row width"
            )
        requested = next(iter(vector_widths))
    n = int(requested)
    if n <= 0:
        raise ValueError(f"invalid n_instruments={n}")
    return n


def _validate_names(
    program: Program,
    data: Mapping[str, object],
    *,
    what: str,
) -> None:
    missing = [name for name in program.input_names if name not in data]
    extra = sorted(set(data) - set(program.input_names))
    if missing or extra:
        raise KeyError(f"{what} mismatch: missing={missing}, extra={extra}")


def compile_formula(
    formula: str | Expr,
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
    """Compile a formula for independently inferred heterogeneous sources."""
    if prefetch_rows < 0:
        raise ValueError("prefetch_rows must be >= 0")

    if data is not None:
        infos = inspect_source_mapping(data, expected_types=input_types)
        if len({info.rows for info in infos.values()}) != 1:
            details = {name: info.rows for name, info in infos.items()}
            raise ValueError(f"cpp_stream sources have different row counts: {details}")
        n = _infer_n(infos, n_instruments)
        program = compile_ir(
            formula,
            dsl_registry=dsl_registry,
            column_names=column_names,
            input_value_types={
                name: _input_value_type(info.input_type, n)
                for name, info in infos.items()
            },
        )
        _validate_names(program, data, what="source")
        ordered = tuple(infos[name].input_type for name in program.input_names)
        bound_sources: Mapping[str, SourceValue] | None = dict(data)
    else:
        if n_instruments is None:
            raise ValueError(
                "n_instruments is required when compile_formula is called without data"
            )
        n = int(n_instruments)
        if n <= 0:
            raise ValueError(f"invalid n_instruments={n}")
        input_value_types = (
            {
                name: _input_value_type(spec, n)
                for name, spec in input_types.items()
            }
            if input_types is not None
            else None
        )
        program = compile_ir(
            formula,
            dsl_registry=dsl_registry,
            column_names=column_names,
            input_value_types=input_value_types,
        )
        if input_types is None:
            ordered = tuple(
                InputTypeSpec("float64", n) for _ in program.input_names
            )
        else:
            _validate_names(program, input_types, what="input_types")
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
    )


__all__ = ["compile_formula"]
