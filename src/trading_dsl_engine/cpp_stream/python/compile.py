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
from trading_dsl_engine.cpp_stream.python.lowering import lower_program
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.cpp_stream.python.runtime import CppStreamRuntime
from trading_dsl_engine.cpp_stream.python.sources import (
    SourceInfo,
    SourceValue,
    inspect_source_mapping,
)
from trading_dsl_engine.ir.frontend import compile_ir
from trading_dsl_engine.ir.ops import CatOp, GroupByOp, InputOp, LiteralOp, NaryOp
from trading_dsl_engine.ir.program import Node, Program
from trading_dsl_engine.ir.types import SCALAR, ValueType, tensor


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
        "-fPIC",
        "-shared",
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


def _row_scalar_analysis(
    program: Program, input_types: tuple[InputTypeSpec, ...]
):
    memo: dict[int, bool] = {}

    def visit(node_id: int) -> bool:
        if node_id in memo:
            return memo[node_id]
        node = program.nodes[node_id]
        if isinstance(node.op, InputOp):
            value = input_types[node.op.input_index].row_scalar
        elif isinstance(node.op, LiteralOp):
            value = True
        elif isinstance(node.op, NaryOp):
            value = all(visit(child) for child in node.child_ids)
        elif isinstance(node.op, CatOp):
            value = False
        else:
            value = False
        memo[node_id] = value
        return value

    return visit


def _apply_input_key_hints(
    program: Program, input_types: tuple[InputTypeSpec, ...]
) -> Program:
    is_row_scalar = _row_scalar_analysis(program, input_types)
    nodes: list[Node] = []
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
    root_kind = program.nodes[program.output_id].value_type.kind
    if root_kind == "object":
        raise ValueError(
            "project Ridge with get_beta(...) or get_preds(...) before output"
        )
    if root_kind not in {"scalar", "vector", "matrix", "fixed", "tensor"}:
        raise ValueError(f"unsupported cpp_stream root kind {root_kind!r}")
    program = _apply_input_key_hints(program, input_types)
    scalar = _row_scalar_analysis(program, input_types)
    plan = lower_program(
        program,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        row_scalar_nodes=frozenset(
            index for index in range(len(program.nodes)) if scalar(index)
        ),
        input_dtypes=tuple(spec.dtype for spec in input_types),
    )
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
    """Compile one formula for independently inferred heterogeneous sources.

    When ``data`` is provided, each input is inspected through its own adapter.
    File extensions, URI schemes, and object types may therefore differ within the
    same formula. The resulting runtime binds these sources, although ``run`` may
    replace them with another compatible mapping.

    Headerless raw inputs need an ``InputTypeSpec`` either in ``input_types`` or in
    an ``InputSource`` wrapper. When ``data`` is omitted, ``n_instruments`` is
    required and the prior explicit ``input_types`` compilation mode remains valid.
    """
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
