from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import threading
from typing import Mapping
import warnings

import includeigen

from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.cpp_stream.python.sources import (
    SourceInfo,
    SourceValue,
    inspect_source,
)
from trading_dsl_engine.ir.ops import (
    CumsumOp,
    EwmOp,
    FFillOp,
    GroupByOp,
    InputOp,
    NaryOp,
    RollingOp,
    ShiftOp,
)
from trading_dsl_engine.ir.program import Node, Program
from trading_dsl_engine.ir.types import SCALAR, ValueType, tensor


_LANE_STATE_OPS = (CumsumOp, FFillOp, ShiftOp, EwmOp, RollingOp)
_PCH_BUILD_LOCK = threading.Lock()
_HEADER_DIGEST_LOCK = threading.Lock()
_HEADER_DIGEST_CACHE: dict[
    tuple[str, str],
    tuple[tuple[tuple[str, int, int], ...], str],
] = {}


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


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


@lru_cache(maxsize=16)
def _compiler_version(compiler: str) -> str:
    version = subprocess.run(
        [compiler, "--version"], capture_output=True, text=True, check=False
    )
    return version.stdout or version.stderr


def _header_digest(cpp_root: str, eigen_include: str) -> str:
    """Hash native headers once per unchanged filesystem fingerprint.

    GP compilation can launch many independent translation units. Re-reading all
    C++ headers and querying the compiler version for every formula serializes
    avoidable Python/filesystem work. The stat fingerprint keeps editable-source
    invalidation semantics: modifying any header changes the digest in-process.
    """

    root = Path(cpp_root)
    eigen_root = Path(eigen_include)
    headers = sorted(root.rglob("*.hpp"))
    eigen_macros = eigen_root / "Eigen" / "src" / "Core" / "util" / "Macros.h"
    fingerprint_items = [
        (
            header.relative_to(root).as_posix(),
            header.stat().st_mtime_ns,
            header.stat().st_size,
        )
        for header in headers
    ]
    if eigen_macros.is_file():
        stat = eigen_macros.stat()
        fingerprint_items.append(
            (f"Eigen::{eigen_macros}", stat.st_mtime_ns, stat.st_size)
        )
    fingerprint = tuple(fingerprint_items)
    cache_key = (cpp_root, eigen_include)
    with _HEADER_DIGEST_LOCK:
        cached = _HEADER_DIGEST_CACHE.get(cache_key)
        if cached is not None and cached[0] == fingerprint:
            return cached[1]

    digest = hashlib.sha256()
    for header in headers:
        digest.update(header.relative_to(root).as_posix().encode())
        digest.update(header.read_bytes())
    digest.update(str(eigen_root).encode())
    if eigen_macros.is_file():
        digest.update(eigen_macros.read_bytes())
    result = digest.hexdigest()
    with _HEADER_DIGEST_LOCK:
        _HEADER_DIGEST_CACHE[cache_key] = (fingerprint, result)
    return result


def _toolchain_digest(
    compiler: str,
    compile_flags: list[str],
    link_flags: list[str],
) -> str:
    cpp_root = _cpp_root()
    eigen_include = _eigen_include()
    digest = hashlib.sha256()
    digest.update(_header_digest(str(cpp_root), str(eigen_include)).encode())
    digest.update(_compiler_version(compiler).encode())
    digest.update("\0".join((*compile_flags, *link_flags)).encode())
    digest.update(
        (
            f"{platform.platform()}|{platform.machine()}|"
            f"{sys.implementation.cache_tag}"
        ).encode()
    )
    return digest.hexdigest()


def _pch_header_text() -> str:
    # Keep this aligned with the formula-independent includes in runner.cpp.j2.
    # Formula-specific stage/output types remain in formula.cpp and still receive
    # the normal -O3/LTO/native optimization passes.
    return """#pragma once
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <memory>
#include <string>
#include <thread>
#include <vector>
#include \"stackdsl/runtime.hpp\"
"""


def _pch_compile_args(
    compiler: str,
    compile_flags: list[str],
    toolchain_digest: str,
) -> tuple[str, ...]:
    """Build/reuse a native PCH, falling back harmlessly on failure."""

    if not _env_enabled("TRADING_DSL_ENGINE_CPP_PCH", True):
        return ()
    basename = Path(compiler).name.lower()
    is_clang = "clang" in basename
    is_gcc = (
        any(token in basename for token in ("g++", "gcc", "c++"))
        and not is_clang
    )
    if not (is_gcc or is_clang):
        return ()

    pch_dir = _cache_root() / "_pch" / toolchain_digest
    header = pch_dir / "cpp_stream_pch.hpp"
    artifact = pch_dir / (
        "cpp_stream_pch.pch" if is_clang else "cpp_stream_pch.hpp.gch"
    )

    def arguments() -> tuple[str, ...]:
        return (
            ("-include-pch", str(artifact))
            if is_clang
            else ("-include", str(header), "-Winvalid-pch")
        )

    if artifact.is_file():
        return arguments()

    with _PCH_BUILD_LOCK:
        if artifact.is_file():
            return arguments()
        pch_dir.mkdir(parents=True, exist_ok=True)
        header.write_text(_pch_header_text())
        unique = f"{os.getpid()}.{threading.get_ident()}"
        temporary = artifact.with_name(f"{artifact.name}.{unique}.tmp")
        pch_flags = [flag for flag in compile_flags if flag != "-shared"]
        command = [
            compiler,
            *pch_flags,
            f"-I{_cpp_root()}",
            f"-I{_eigen_include()}",
            "-x",
            "c++-header",
            str(header),
            "-o",
            str(temporary),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            temporary.unlink(missing_ok=True)
            failure = pch_dir / "pch-build-failure.txt"
            failure.write_text(
                " ".join(command) + "\n" + result.stdout + result.stderr
            )
            warnings.warn(
                "cpp_stream PCH build failed; falling back to ordinary native "
                f"compilation. See {failure}",
                RuntimeWarning,
                stacklevel=2,
            )
            return ()
        temporary.replace(artifact)
    return arguments()


def build_shared(source: str) -> tuple[Path, Path]:
    compiler = _compiler()
    compile_flags, link_flags = _flags()
    toolchain_digest = _toolchain_digest(compiler, compile_flags, link_flags)
    digest = hashlib.sha256(source.encode())
    digest.update(toolchain_digest.encode())
    build_dir = _cache_root() / digest.hexdigest()
    cpp_path = build_dir / "formula.cpp"
    so_path = build_dir / "formula.so"
    if so_path.is_file():
        return so_path, cpp_path

    pch_args = _pch_compile_args(compiler, compile_flags, toolchain_digest)
    build_dir.mkdir(parents=True, exist_ok=True)
    unique = f"{os.getpid()}.{threading.get_ident()}"
    temporary_cpp = build_dir / f"formula.{unique}.cpp"
    temporary_so = build_dir / f"formula.{unique}.so"
    temporary_cpp.write_text(source)
    command = [
        compiler,
        *compile_flags,
        "-pipe",
        f"-I{_cpp_root()}",
        f"-I{_eigen_include()}",
        *pch_args,
        str(temporary_cpp),
        *link_flags,
        "-o",
        str(temporary_so),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        temporary_cpp.unlink(missing_ok=True)
        temporary_so.unlink(missing_ok=True)
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


def input_value_type(spec: InputTypeSpec, n_instruments: int) -> ValueType:
    shape = tuple(spec.row_shape or ())
    if not shape:
        return SCALAR
    logical_shape = (
        (None,) + shape[1:] if shape[0] == n_instruments else shape
    )
    return tensor(logical_shape, dtype=spec.dtype)


class ReferencedSourceTypes(Mapping[str, ValueType]):
    """Inspect only source metadata actually referenced by the IR."""

    def __init__(
        self,
        data: Mapping[str, SourceValue],
        expected_types: Mapping[str, InputTypeSpec] | None,
        n_instruments: int | None,
    ) -> None:
        self._data = data
        self._expected_types = expected_types or {}
        self._n_instruments = n_instruments
        self._infos: dict[str, SourceInfo] = {}

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def _info(self, name: str) -> SourceInfo:
        if name not in self._data:
            raise KeyError(name)
        info = self._infos.get(name)
        if info is None:
            info = inspect_source(
                self._data[name], expected=self._expected_types.get(name)
            )
            self._infos[name] = info
        return info

    def __getitem__(self, name: str) -> ValueType:
        spec = self._info(name).input_type
        if self._n_instruments is not None:
            return input_value_type(spec, int(self._n_instruments))
        shape = tuple(spec.row_shape or ())
        if not shape:
            return SCALAR
        return tensor((None,) + shape[1:], dtype=spec.dtype)

    def infos_for(self, names: tuple[str, ...]) -> dict[str, SourceInfo]:
        return {name: self._info(name) for name in names}


def _broadcast_shapes(
    shapes: tuple[tuple[int | None, ...], ...]
) -> tuple[int | None, ...]:
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


def repair_value_types(program: Program) -> Program:
    """Propagate tensor broadcasting and shape-preserving temporal operators."""

    nodes: list[Node] = []
    for original in program.nodes:
        node = original
        op = node.op
        if isinstance(op, GroupByOp):
            inner = repair_value_types(op.inner_program)
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


def row_scalar_analysis(
    program: Program,
    input_types: tuple[InputTypeSpec, ...],
):
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


def apply_input_key_hints(
    program: Program,
    input_types: tuple[InputTypeSpec, ...],
) -> Program:
    is_row_scalar = row_scalar_analysis(program, input_types)
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


def infer_n(infos: Mapping[str, SourceInfo], requested: int | None) -> int:
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


def validate_names(
    program: Program,
    data: Mapping[str, object],
    *,
    what: str,
) -> None:
    missing = [name for name in program.input_names if name not in data]
    if missing:
        raise KeyError(f"{what} mismatch: missing={missing}")


__all__ = [
    "ReferencedSourceTypes",
    "apply_input_key_hints",
    "build_shared",
    "infer_n",
    "input_value_type",
    "repair_value_types",
    "row_scalar_analysis",
    "validate_names",
]
