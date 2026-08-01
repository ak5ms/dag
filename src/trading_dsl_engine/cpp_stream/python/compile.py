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

from trading_dsl_engine.base.dsl import DSLFunctionRegistry
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.cpp_stream.python.codegen import render_translation_unit
from trading_dsl_engine.cpp_stream.python.lowering import lower_program
from trading_dsl_engine.cpp_stream.python.npy import (
    InputTypeSpec,
    NpyArrayInfo,
    inspect_npy_mapping,
)
from trading_dsl_engine.cpp_stream.python.runtime import CppStreamRuntime
from trading_dsl_engine.ir.frontend import compile_ir
from trading_dsl_engine.ir.ops import GroupByOp, InputOp, LiteralOp, NaryOp
from trading_dsl_engine.ir.program import Node, Program


def _cpp_root() -> Path:
    return Path(__file__).resolve().parents[1] / "cpp"


def _cache_root() -> Path:
    configured = os.environ.get("TRADING_DSL_ENGINE_CPP_STREAM_CACHE")
    return Path(configured).expanduser() if configured else Path.home() / ".cache" / "trading_dsl_engine" / "cpp_stream"


def _compiler() -> str:
    cxx = os.environ.get("CXX", "g++")
    resolved = shutil.which(cxx)
    if resolved is None:
        raise RuntimeError(f"cpp_stream requires a C++20 compiler; could not find {cxx!r}")
    return resolved


def _compiler_identity(compiler: str) -> str:
    result = subprocess.run([compiler, "--version"], check=False, capture_output=True, text=True)
    first_line = (result.stdout or result.stderr).splitlines()
    return f"{compiler}\n{first_line[0] if first_line else 'unknown-version'}"


def _compile_flags() -> list[str]:
    if os.name == "nt":
        raise RuntimeError("cpp_stream mmap codegen currently targets POSIX/Linux; Windows support is not implemented yet")
    flags = ["-std=c++20", "-O3", "-DNDEBUG", "-fPIC", "-shared", "-fno-math-errno", "-funroll-loops"]
    if sys.platform.startswith("linux"):
        flags.append("-D_GNU_SOURCE")
    if os.environ.get("TRADING_DSL_ENGINE_CPP_NATIVE", "1").lower() not in {"0", "false", "no", "off"}:
        flags.extend(["-march=native", "-mtune=native"])
    if os.environ.get("TRADING_DSL_ENGINE_CPP_LTO", "1").lower() not in {"0", "false", "no", "off"}:
        flags.append("-flto")
    flags.extend(shlex.split(os.environ.get("TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS", "")))
    return flags


def _link_flags() -> list[str]:
    flags = ["-Wl,-O3"]
    flags.extend(shlex.split(os.environ.get("TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS", "")))
    return flags


def _fingerprint(source: str, compiler: str, flags: list[str], link_flags: list[str]) -> str:
    digest = hashlib.sha256()
    digest.update(source.encode())
    cpp_root = _cpp_root()
    for header in sorted(cpp_root.rglob("*.hpp")):
        digest.update(header.relative_to(cpp_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(header.read_bytes())
        digest.update(b"\0")
    digest.update(_compiler_identity(compiler).encode())
    digest.update("\0".join(flags).encode())
    digest.update("\0".join(link_flags).encode())
    digest.update(platform.platform().encode())
    digest.update(platform.machine().encode())
    digest.update(platform.processor().encode())
    digest.update(sys.implementation.cache_tag.encode())
    return digest.hexdigest()


def _build_shared(source: str) -> tuple[Path, Path]:
    compiler = _compiler()
    flags = _compile_flags()
    link_flags = _link_flags()
    fingerprint = _fingerprint(source, compiler, flags, link_flags)
    build_dir = _cache_root() / fingerprint
    cpp_path = build_dir / "formula.cpp"
    so_path = build_dir / "formula.so"
    if so_path.is_file():
        return so_path, cpp_path
    build_dir.mkdir(parents=True, exist_ok=True)
    temporary = build_dir / f"formula.{os.getpid()}.cpp"
    temporary.write_text(source)
    output_tmp = build_dir / f"formula.{os.getpid()}.so"
    command = [compiler, *flags, f"-I{_cpp_root()}", str(temporary), *link_flags, "-o", str(output_tmp)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("cpp_stream native compilation failed\n" + " ".join(command) + "\n" + exc.stdout + exc.stderr) from exc
    temporary.replace(cpp_path)
    output_tmp.replace(so_path)
    return so_path, cpp_path


def _row_scalar_analysis(program: Program, input_types: tuple[InputTypeSpec, ...]):
    memo: dict[int, bool] = {}

    def visit(node_id: int) -> bool:
        if node_id in memo:
            return memo[node_id]
        node = program.nodes[node_id]
        if isinstance(node.op, InputOp):
            result = input_types[node.op.input_index].row_scalar
        elif isinstance(node.op, LiteralOp):
            result = True
        elif isinstance(node.op, NaryOp):
            result = all(visit(child) for child in node.child_ids)
        else:
            result = False
        memo[node_id] = result
        return result

    return visit


def _row_scalar_node_ids(
    program: Program,
    input_types: tuple[InputTypeSpec, ...],
) -> frozenset[int]:
    is_row_scalar = _row_scalar_analysis(program, input_types)
    return frozenset(
        node_id
        for node_id in range(len(program.nodes))
        if is_row_scalar(node_id)
    )


def _apply_input_key_hints(
    program: Program,
    input_types: tuple[InputTypeSpec, ...],
) -> Program:
    is_row_scalar = _row_scalar_analysis(program, input_types)
    nodes: list[Node] = []
    for node in program.nodes:
        op = node.op
        if not isinstance(op, GroupByOp):
            nodes.append(node)
            continue
        specs = []
        for index, spec in enumerate(op.key_specs):
            child_id = node.child_ids[index]
            child_op = program.nodes[child_id].op
            row_scalar = spec.row_scalar
            if row_scalar is None:
                row_scalar = is_row_scalar(child_id)
            dtype = spec.dtype
            if dtype is None:
                dtype = (
                    input_types[child_op.input_index].dtype
                    if isinstance(child_op, InputOp)
                    else "float64"
                )
            elif isinstance(child_op, InputOp):
                actual = input_types[child_op.input_index].dtype
                if dtype != actual:
                    raise TypeError(
                        f"Key dtype hint {dtype!r} does not match direct input "
                        f"{child_op.name!r} dtype {actual!r}"
                    )
            specs.append(replace(spec, row_scalar=row_scalar, dtype=dtype))
        nodes.append(replace(node, op=replace(op, key_specs=tuple(specs))))
    return replace(program, nodes=tuple(nodes))


def _compile_program(
    program: Program,
    *,
    n_instruments: int,
    input_types: tuple[InputTypeSpec, ...],
    default_group_capacity: int,
    key_cardinalities: Mapping[str, int] | None,
    prefetch_rows: int,
) -> CppStreamRuntime:
    if program.nodes[program.output_id].value_type.kind != "vector":
        raise ValueError("cpp_stream currently requires a vector root output")
    program = _apply_input_key_hints(program, input_types)
    row_scalar_nodes = _row_scalar_node_ids(program, input_types)
    plan = lower_program(
        program,
        n_instruments=n_instruments,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        row_scalar_nodes=row_scalar_nodes,
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
    )


def compile_formula(
    formula: str | Expr,
    *,
    n_instruments: int,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    prefetch_rows: int = 16,
    input_types: Mapping[str, InputTypeSpec] | None = None,
) -> CppStreamRuntime:
    """Compile a formula-specialized raw-file runner.

    ``input_types`` may describe typed headerless inputs. When omitted, every
    input is row-major float64 with width ``n_instruments``. Per-key ``Key``
    descriptors are preferred over the legacy global ``key_cardinalities`` map.
    """
    if prefetch_rows < 0:
        raise ValueError("prefetch_rows must be >= 0")
    program = compile_ir(formula, dsl_registry=dsl_registry, column_names=column_names)
    if input_types is None:
        ordered_types = tuple(
            InputTypeSpec("float64", n_instruments)
            for _ in program.input_names
        )
    else:
        missing = [name for name in program.input_names if name not in input_types]
        extra = sorted(set(input_types) - set(program.input_names))
        if missing or extra:
            raise KeyError(f"input_types mismatch: missing={missing}, extra={extra}")
        ordered_types = tuple(input_types[name] for name in program.input_names)
    return _compile_program(
        program,
        n_instruments=n_instruments,
        input_types=ordered_types,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        prefetch_rows=prefetch_rows,
    )


def _infer_n_instruments(
    infos: Mapping[str, NpyArrayInfo],
    requested: int | None,
) -> int:
    vector_widths = {info.row_width for info in infos.values() if info.row_width > 1}
    if requested is None:
        if len(vector_widths) != 1:
            raise ValueError(
                "n_instruments could not be inferred uniquely from .npy shapes; "
                "pass n_instruments explicitly"
            )
        requested = next(iter(vector_widths))
    n = int(requested)
    if n <= 0:
        raise ValueError("n_instruments must be > 0")
    bad = {
        name: info.row_width
        for name, info in infos.items()
        if info.row_width not in (1, n)
    }
    if bad:
        raise ValueError(
            f".npy row widths must be 1 or n_instruments={n}; got {bad}"
        )
    return n


def compile_npy_formula(
    formula: str | Expr,
    data: Mapping[str, str | Path],
    *,
    n_instruments: int | None = None,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    prefetch_rows: int = 16,
) -> CppStreamRuntime:
    """Inspect mmap .npy headers, then compile exact input dtypes and widths."""
    if prefetch_rows < 0:
        raise ValueError("prefetch_rows must be >= 0")
    program = compile_ir(formula, dsl_registry=dsl_registry, column_names=column_names)
    missing = [name for name in program.input_names if name not in data]
    extra = sorted(set(data) - set(program.input_names))
    if missing or extra:
        raise KeyError(f".npy input mapping mismatch: missing={missing}, extra={extra}")
    infos = inspect_npy_mapping({name: data[name] for name in program.input_names})
    row_counts = {info.rows for info in infos.values()}
    if len(row_counts) != 1:
        raise ValueError(
            f".npy inputs have different row counts: "
            f"{ {name: info.rows for name, info in infos.items()} }"
        )
    n = _infer_n_instruments(infos, n_instruments)
    ordered_types = tuple(infos[name].input_type for name in program.input_names)
    return _compile_program(
        program,
        n_instruments=n,
        input_types=ordered_types,
        default_group_capacity=default_group_capacity,
        key_cardinalities=key_cardinalities,
        prefetch_rows=prefetch_rows,
    )
