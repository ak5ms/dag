from __future__ import annotations

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
from trading_dsl_engine.cpp_stream.python.runtime import CppStreamRuntime
from trading_dsl_engine.ir.frontend import compile_ir


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


def compile_formula(
    formula: str | Expr,
    *,
    n_instruments: int,
    dsl_registry: DSLFunctionRegistry | None = None,
    column_names: list[str] | tuple[str, ...] | None = None,
    default_group_capacity: int = 64,
    key_cardinalities: Mapping[str, int] | None = None,
    prefetch_rows: int = 16,
) -> CppStreamRuntime:
    """Compile an existing DSL formula to a formula-specialized mmap C++ runner.

    This backend starts from ``trading_dsl_engine.ir`` and does not depend on
    ``jax_flat``. ``key_cardinalities`` enables direct dense indexing for bounded
    categorical inputs such as ``{"minute_of_day": 1440}``.
    """
    if prefetch_rows < 0:
        raise ValueError("prefetch_rows must be >= 0")
    program = compile_ir(formula, dsl_registry=dsl_registry, column_names=column_names)
    if program.nodes[program.output_id].value_type.kind != "vector":
        raise ValueError("cpp_stream currently requires a vector root output")
    plan = lower_program(program, n_instruments=n_instruments, default_group_capacity=default_group_capacity, key_cardinalities=key_cardinalities)
    generated = render_translation_unit(plan, n_instruments=n_instruments, prefetch_rows=prefetch_rows)
    library_path, cpp_path = _build_shared(generated.text)
    return CppStreamRuntime(program=program, plan=plan, library_path=library_path, generated_cpp=cpp_path, n_instruments=n_instruments)
