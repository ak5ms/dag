from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import types
from typing import Any, Iterable, Mapping, Sequence


_DEFAULT_CLARABEL_CPP_COMMIT = "0de6259a3edfd5cc041ec42b2148599ce63e73cb"
_DEFAULT_CLARABEL_RS_TAG = "v0.11.1"
_SUPPORTED_CVXPYGEN_VERSION = "1.0.0"
_CANONICAL_BLOCKS = ("P", "q", "A", "b", "d")
_SOLVER_UPDATE_BLOCKS = ("P", "A", "q", "b")


@dataclass(frozen=True, slots=True)
class ClarabelNativePaths:
    """Headers and static library for the Clarabel C ABI."""

    include_dir: Path
    static_library: Path
    version: str = "0.11.1"

    def normalized(self) -> "ClarabelNativePaths":
        include_dir = Path(self.include_dir).expanduser().resolve()
        static_library = Path(self.static_library).expanduser().resolve()
        if not (include_dir / "clarabel.h").is_file():
            raise FileNotFoundError(
                f"Clarabel C header not found: {include_dir / 'clarabel.h'}"
            )
        if not static_library.is_file():
            raise FileNotFoundError(
                f"Clarabel static library not found: {static_library}"
            )
        return ClarabelNativePaths(include_dir, static_library, self.version)


@dataclass(frozen=True, slots=True)
class ParameterLayout:
    name: str
    shape: tuple[int, ...]
    size: int
    offset: int
    dirty_blocks: tuple[str, ...]
    column_major: bool = True


@dataclass(frozen=True, slots=True)
class PrimalLayout:
    name: str
    shape: tuple[int, ...]
    size: int


@dataclass(frozen=True, slots=True)
class GeneratedCvxpygenProgram:
    """Generated, persistent, reentrant CVXPYgen/Clarabel program."""

    root: Path
    instance_header: Path
    manifest_path: Path
    class_name: str
    prefix: str
    parameters: tuple[ParameterLayout, ...]
    primals: tuple[PrimalLayout, ...]
    clarabel: ClarabelNativePaths

    @property
    def include_dirs(self) -> tuple[Path, ...]:
        return (
            self.root / "cpp" / "include",
            self.root / "c" / "include",
            self.clarabel.include_dir,
        )

    @property
    def link_files(self) -> tuple[Path, ...]:
        return (self.clarabel.static_library,)

    @property
    def fingerprint_files(self) -> tuple[Path, ...]:
        generated_headers = tuple(
            sorted(
                path
                for directory in (self.root / "cpp" / "include", self.root / "c" / "include")
                for path in directory.rglob("*")
                if path.is_file()
            )
        )
        return (*generated_headers, self.manifest_path)

    def build_shared_kwargs(self) -> dict[str, tuple[Path, ...]]:
        """Arguments for ``cpp_stream.python.compiler_support.build_shared``."""

        return {
            "extra_include_dirs": self.include_dirs,
            "extra_link_files": self.link_files,
            "extra_fingerprint_files": self.fingerprint_files,
        }

    def compiler_arguments(self) -> tuple[str, ...]:
        return tuple(f"-I{path}" for path in self.include_dirs) + (
            str(self.clarabel.static_library),
            "-ldl",
            "-lpthread",
            "-lm",
        )


@dataclass(frozen=True, slots=True)
class _GeneratedParameter:
    name: str
    offset: int
    size: int
    dirty_blocks: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _GeneratedPrimal:
    name: str
    size: int


@contextmanager
def _without_pdaqp_import_side_effects():
    """Avoid CVXPYgen importing Julia when only Clarabel is requested.

    CVXPYgen 1.0 imports every solver backend eagerly, and PDAQP imports
    juliacall.  Clarabel generation does not use that backend.  A temporary
    stub keeps offline/native builds deterministic without changing the
    installed CVXPYgen package.
    """

    previous = sys.modules.get("pdaqp")
    if previous is None:
        stub = types.ModuleType("pdaqp")
        stub.MPQP = type("MPQP", (), {})
        sys.modules["pdaqp"] = stub
    try:
        yield
    finally:
        if previous is None:
            sys.modules.pop("pdaqp", None)
        else:
            sys.modules["pdaqp"] = previous


def _import_cpg():
    with _without_pdaqp_import_side_effects():
        from cvxpygen import cpg

    return cpg


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _safe_identifier(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"invalid {label} {value!r}")
    return value


def _find_function_span(text: str, signature: str) -> tuple[int, int]:
    start = text.find(signature)
    if start < 0:
        raise ValueError(f"generated function not found: {signature}")
    brace = text.find("{", start)
    if brace < 0:
        raise ValueError(f"generated function has no body: {signature}")
    depth = 0
    for index in range(brace, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return start, index + 1
    raise ValueError(f"unterminated generated function: {signature}")


def _strip_generated_preamble(text: str, include_name: str) -> str:
    marker = f'#include "{include_name}"'
    position = text.find(marker)
    if position < 0:
        raise ValueError(f"generated include {marker!r} not found")
    return text[position + len(marker) :].lstrip()


def _generated_prefix(solve_source: str) -> str:
    match = re.search(r"\bvoid\s+([A-Za-z_][A-Za-z0-9_]*)cpg_solve\s*\(\s*\)", solve_source)
    if match is None:
        raise ValueError("could not infer CVXPYgen symbol prefix")
    return match.group(1)


def _generated_parameters(
    solve_source: str,
    problem: Any,
    prefix: str,
) -> tuple[_GeneratedParameter, ...]:
    by_name = {parameter.name(): parameter for parameter in problem.parameters()}
    pattern = re.compile(
        rf"void\s+{re.escape(prefix)}cpg_update_([A-Za-z_][A-Za-z0-9_]*)"
        rf"\s*\((.*?)\)\s*\{{(.*?)\n\}}",
        flags=re.S,
    )
    result: list[_GeneratedParameter] = []
    for match in pattern.finditer(solve_source):
        name, arguments, body = match.groups()
        parameter = by_name.get(name)
        if parameter is None:
            continue
        offset_match = re.search(
            rf"{re.escape(prefix)}cpg_params_vec\s*\[\s*(?:idx\s*\+\s*)?(\d+)\s*\]",
            body,
        )
        if offset_match is None:
            raise ValueError(f"could not infer parameter offset for {name!r}")
        offset = int(offset_match.group(1))
        shape = tuple(int(extent) for extent in parameter.shape)
        size = 1
        for extent in shape:
            size *= extent
        dirty = tuple(
            block
            for block in _CANONICAL_BLOCKS
            if re.search(
                rf"{re.escape(prefix)}Canon_Outdated\.{block}\s*=\s*1\s*;",
                body,
            )
        )
        if not dirty:
            raise ValueError(f"generated update for {name!r} marks no canonical block")
        result.append(_GeneratedParameter(name, offset, size, dirty))
    missing = sorted(set(by_name) - {item.name for item in result})
    if missing:
        raise ValueError(f"CVXPYgen omitted parameter update functions: {missing}")
    return tuple(sorted(result, key=lambda item: item.offset))


def _generated_primals(
    workspace_header: str,
    problem: Any,
    prefix: str,
) -> tuple[_GeneratedPrimal, ...]:
    by_name = {variable.name(): variable for variable in problem.variables()}
    result: list[_GeneratedPrimal] = []
    for name, variable in by_name.items():
        array = re.search(
            rf"extern\s+cpg_float\s+{re.escape(prefix)}cpg_{re.escape(name)}\[(\d+)\]\s*;",
            workspace_header,
        )
        if array is not None:
            size = int(array.group(1))
        else:
            scalar = re.search(
                rf"extern\s+cpg_float\s+{re.escape(prefix)}cpg_{re.escape(name)}\s*;",
                workspace_header,
            )
            if scalar is None:
                continue
            size = 1
        expected = 1
        for extent in variable.shape:
            expected *= int(extent)
        if size != expected:
            raise ValueError(
                f"generated primal {name!r} has size {size}; expected {expected}"
            )
        result.append(_GeneratedPrimal(name, size))
    return tuple(result)


def _loop_count(solve_source: str, prefix: str, block: str) -> int:
    signature = f"void {prefix}cpg_canonicalize_{block}()"
    start, end = _find_function_span(solve_source, signature)
    body = solve_source[start:end]
    match = re.search(r"for\s*\(\s*i\s*=\s*0\s*;\s*i\s*<\s*(\d+)\s*;", body)
    if match is None:
        raise ValueError(f"could not infer canonical {block} size")
    return int(match.group(1))


def _canonical_blocks(solve_source: str, prefix: str) -> tuple[str, ...]:
    return tuple(
        block
        for block in _CANONICAL_BLOCKS
        if f"void {prefix}cpg_canonicalize_{block}()" in solve_source
    )


def _settings_assignments(solve_body: str, prefix: str) -> str:
    assignments = re.findall(
        rf"\s*{re.escape(prefix)}settings\.[A-Za-z0-9_]+\s*=\s*"
        rf"{re.escape(prefix)}Canon_Settings\.[A-Za-z0-9_]+\s*;",
        solve_body,
    )
    if not assignments:
        raise ValueError("generated Clarabel settings assignments not found")
    return "\n".join(line.strip() for line in assignments)


def _matrix_initializers(solve_body: str, prefix: str) -> tuple[str, ...]:
    initializers: list[str] = []
    for match in re.finditer(
        rf"clarabel_CscMatrix_init\(&{re.escape(prefix)}(?:P|A),.*?;",
        solve_body,
        flags=re.S,
    ):
        initializer = match.group(0).strip()
        if initializer not in initializers:
            initializers.append(initializer)
    return tuple(initializers)


def _solver_constructor(solve_body: str, prefix: str) -> str:
    match = re.search(
        rf"{re.escape(prefix)}solver\s*=\s*clarabel_DefaultSolver_new\(.*?\);",
        solve_body,
        flags=re.S,
    )
    if match is None:
        raise ValueError("generated Clarabel constructor not found")
    return match.group(0).strip()


def _patch_setting_setters(solve_source: str, prefix: str) -> str:
    cursor = 0
    pieces: list[str] = []
    marker = f"void {prefix}cpg_set_solver_"
    while True:
        start = solve_source.find(marker, cursor)
        if start < 0:
            pieces.append(solve_source[cursor:])
            break
        pieces.append(solve_source[cursor:start])
        brace_start = solve_source.find("{", start)
        if brace_start < 0:
            raise ValueError("generated solver setting function missing body")
        depth = 0
        end = None
        for index in range(brace_start, len(solve_source)):
            if solve_source[index] == "{":
                depth += 1
            elif solve_source[index] == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        if end is None:
            raise ValueError("unterminated generated solver setting function")
        function = solve_source[start:end]
        function = function[:-1] + "  solver_settings_dirty_ = true;\n}\n"
        pieces.append(function)
        cursor = end
    return "".join(pieces)


def _patch_persistent_solve(solve_source: str, prefix: str) -> str:
    signature = f"void {prefix}cpg_solve()"
    start, end = _find_function_span(solve_source, signature)
    original = solve_source[start:end]
    blocks = _canonical_blocks(solve_source, prefix)
    matrices = _matrix_initializers(original, prefix)
    constructor = _solver_constructor(original, prefix)
    settings = _settings_assignments(original, prefix)

    lines = [f"void {prefix}cpg_solve(){{"]
    for block in blocks:
        lines.append(
            f"  const bool {block}_changed = {prefix}Canon_Outdated.{block};"
        )
    for block in blocks:
        lines.append(
            f"  if ({block}_changed) {prefix}cpg_canonicalize_{block}();"
        )
    lines.extend(
        [
            f"  if (solver_settings_dirty_ && {prefix}solver) {{",
            f"    clarabel_DefaultSolver_free({prefix}solver);",
            f"    {prefix}solver = nullptr;",
            "  }",
            f"  if (!{prefix}solver) {{",
            f"    {prefix}cpg_copy_all();",
        ]
    )
    lines.extend(f"    {initializer}" for initializer in matrices)
    lines.extend(
        [
            f"    {prefix}settings = clarabel_DefaultSettings_default();",
            *(f"    {line}" for line in settings.splitlines()),
            f"    {constructor}",
            "    solver_settings_dirty_ = false;",
            "  } else {",
        ]
    )
    for block in blocks:
        if block not in _SOLVER_UPDATE_BLOCKS:
            continue
        count = _loop_count(solve_source, prefix, block)
        value = (
            f"{prefix}Canon_Params_conditioning.{block}->x"
            if block in {"P", "A"}
            else f"{prefix}Canon_Params_conditioning.{block}"
        )
        lines.extend(
            [
                f"    if ({block}_changed) {{",
                f"      {prefix}cpg_copy_{block}();",
                f"      clarabel_DefaultSolver_update_{block}({prefix}solver, {value}, {count});",
                "    }",
            ]
        )
    lines.extend(
        [
            "  }",
            f"  clarabel_DefaultSolver_solve({prefix}solver);",
            f"  {prefix}solution = clarabel_DefaultSolver_solution({prefix}solver);",
            f"  {prefix}cpg_retrieve_prim();",
            f"  {prefix}cpg_retrieve_dual();",
            f"  {prefix}cpg_retrieve_info();",
        ]
    )
    for block in blocks:
        lines.append(f"  {prefix}Canon_Outdated.{block} = 0;")
    lines.append("}")
    replacement = "\n".join(lines)
    patched = solve_source[:start] + replacement + solve_source[end:]
    return _patch_setting_setters(patched, prefix)


def _declaration_symbol(line: str) -> str | None:
    match = re.match(
        r"\s*(?:cpg_float|cpg_int|cpg_csc|Canon_Params_t|Canon_Outdated_t|"
        r"CPG_Prim_t|CPG_Dual_t|CPG_Info_t|CPG_Result_t|Canon_Settings_t|"
        r"ClarabelCscMatrix|ClarabelSupportedConeT|ClarabelDefaultSettings|"
        r"ClarabelDefaultSolver\s*\*|ClarabelDefaultSolution)\s+"
        r"([A-Za-z_][A-Za-z0-9_]*)",
        line,
    )
    return None if match is None else match.group(1)


def _shared_workspace_symbol(symbol: str, prefix: str) -> bool:
    local = symbol[len(prefix) :] if symbol.startswith(prefix) else symbol
    if "_map_" in local or re.fullmatch(r"canon_[A-Za-z0-9]+_map", local):
        return True
    if re.fullmatch(r"canon_(?:P|A)(?:_conditioning)?_(?:i|p)", local):
        return True
    if local == "cones":
        return True
    return False


def _instance_workspace_body(workspace_source: str, prefix: str) -> str:
    body = _strip_generated_preamble(workspace_source, "cpg_workspace.h")
    lines = body.splitlines()
    result: list[str] = []
    for line in lines:
        symbol = _declaration_symbol(line)
        if symbol is not None and _shared_workspace_symbol(symbol, prefix):
            leading = line[: len(line) - len(line.lstrip())]
            line = leading + "inline static " + line.lstrip()
        result.append(line)
    return "\n".join(result).rstrip() + "\n"


def _bulk_setters(parameters: Sequence[_GeneratedParameter], prefix: str) -> str:
    methods: list[str] = []
    for parameter in parameters:
        flags = " ".join(
            f"{prefix}Canon_Outdated.{block} = 1;"
            for block in parameter.dirty_blocks
        )
        methods.append(
            f"""  void set_{parameter.name}(std::span<const cpg_float> values) {{
    if (values.size() != {parameter.size}) throw std::invalid_argument(
        \"parameter {parameter.name} expects {parameter.size} values\");
    std::copy(values.begin(), values.end(), {prefix}cpg_params_vec + {parameter.offset});
    {flags}
  }}"""
        )
    return "\n\n".join(methods)


def _primal_accessors(primals: Sequence[_GeneratedPrimal], prefix: str) -> str:
    return "\n\n".join(
        f"""  [[nodiscard]] std::span<const cpg_float> primal_{item.name}() const noexcept {{
    return std::span<const cpg_float>({prefix}CPG_Result.prim->{item.name}, {item.size});
  }}"""
        for item in primals
    )


def _replace_compound_literals(source: str) -> tuple[str, str]:
    """Lift C compound-literal arrays into persistent C++ object members."""

    declarations: list[str] = []
    counter = 0
    pattern = re.compile(r"\((cpg_int|cpg_float)\[\]\)\s*\{([^{}]*)\}")

    def replace(match: re.Match[str]) -> str:
        nonlocal counter
        c_type = match.group(1)
        values = [value.strip() for value in match.group(2).split(",") if value.strip()]
        name = f"compound_literal_{counter}_"
        counter += 1
        declarations.append(
            f"  std::array<{c_type}, {len(values)}> {name}{{{{{', '.join(values)}}}}};"
        )
        return f"{name}.data()"

    return pattern.sub(replace, source), "\n".join(declarations)


def _emit_instance_header(
    generated_root: Path,
    class_name: str,
    problem: Any,
) -> tuple[Path, str, tuple[_GeneratedParameter, ...], tuple[_GeneratedPrimal, ...]]:
    solve_path = generated_root / "c" / "src" / "cpg_solve.c"
    workspace_source_path = generated_root / "c" / "src" / "cpg_workspace.c"
    workspace_header_path = generated_root / "c" / "include" / "cpg_workspace.h"
    solve_source = solve_path.read_text()
    workspace_source = workspace_source_path.read_text()
    workspace_header = workspace_header_path.read_text()
    prefix = _generated_prefix(solve_source)
    parameters = _generated_parameters(solve_source, problem, prefix)
    primals = _generated_primals(workspace_header, problem, prefix)
    solve_body = _strip_generated_preamble(solve_source, "cpg_workspace.h")
    solve_body, compound_members = _replace_compound_literals(solve_body)
    solve_body = re.sub(
        r"\bstatic\s+cpg_int\s+i\s*;\s*\bstatic\s+cpg_int\s+j\s*;",
        "",
        solve_body,
        count=1,
        flags=re.S,
    )
    solve_body = _patch_persistent_solve(solve_body, prefix)
    workspace_body = _instance_workspace_body(workspace_source, prefix)

    header_dir = generated_root / "cpp" / "include"
    header_dir.mkdir(parents=True, exist_ok=True)
    compatibility = header_dir / "Clarabel"
    compatibility.write_text(
        '#pragma once\nextern "C" {\n#include <clarabel.h>\n}\n'
    )
    header = header_dir / "cpg_instance.hpp"
    header.write_text(
        f"""#pragma once

#include \"cpg_workspace.h\"
#include <algorithm>
#include <array>
#include <cstddef>
#include <span>
#include <stdexcept>

class {class_name} final {{
  cpg_int i{{0}};
  cpg_int j{{0}};
  bool solver_settings_dirty_{{false}};
{compound_members}

 public:
{workspace_body}
{solve_body}
  {class_name}() {{
    {prefix}cpg_set_solver_default_settings();
    {prefix}cpg_set_solver_verbose(0);
    {prefix}cpg_set_solver_presolve_enable(0);
    solver_settings_dirty_ = false;
  }}

  ~{class_name}() {{ reset_solver(); }}
  {class_name}(const {class_name}&) = delete;
  {class_name}& operator=(const {class_name}&) = delete;
  {class_name}({class_name}&&) = delete;
  {class_name}& operator=({class_name}&&) = delete;

  void reset_solver() noexcept {{
    if ({prefix}solver) {{
      clarabel_DefaultSolver_free({prefix}solver);
      {prefix}solver = nullptr;
    }}
  }}

  void solve() {{ {prefix}cpg_solve(); }}

{_bulk_setters(parameters, prefix)}

{_primal_accessors(primals, prefix)}

  [[nodiscard]] const CPG_Result_t& result() const noexcept {{
    return {prefix}CPG_Result;
  }}
}};
"""
    )
    return header, prefix, parameters, primals


def generate_clarabel_program(
    problem: Any,
    *,
    code_dir: str | os.PathLike[str],
    clarabel: ClarabelNativePaths,
    class_name: str = "GeneratedCvxpyProgram",
    prefix: str = "cpg_",
    enable_settings: Iterable[str] = (
        "verbose",
        "max_iter",
        "tol_gap_abs",
        "tol_gap_rel",
        "tol_feas",
        "presolve_enable",
    ),
    force: bool = False,
) -> GeneratedCvxpygenProgram:
    """Generate a reentrant persistent Clarabel class from a CVXPY problem.

    CVXPYgen still owns DPP validation, all parameter-to-canonical maps, cone
    layout and inverse result maps.  This function changes only the generated
    object lifetime: globals become C++ instance members and Clarabel is updated
    in place after the first solve.
    """

    _safe_identifier(class_name, label="C++ class name")
    _safe_identifier(prefix, label="CVXPYgen prefix")
    cvxpygen_version = _package_version("cvxpygen")
    if cvxpygen_version != _SUPPORTED_CVXPYGEN_VERSION:
        raise RuntimeError(
            "generated-source adapter supports CVXPYgen "
            f"{_SUPPORTED_CVXPYGEN_VERSION}, found {cvxpygen_version}; "
            "update the adapter and its generated-code tests before changing versions"
        )
    if not problem.is_dcp(dpp=True):
        raise ValueError("CVXPY problem must be DPP-compliant")
    root = Path(code_dir).expanduser().resolve()
    if root.exists():
        if not force:
            raise FileExistsError(
                f"code directory already exists: {root}; pass force=True to replace it"
            )
        shutil.rmtree(root)
    root.parent.mkdir(parents=True, exist_ok=True)
    cpg = _import_cpg()
    cpg.generate_code(
        problem,
        code_dir=str(root),
        solver="CLARABEL",
        enable_settings=list(enable_settings),
        prefix=prefix,
        wrapper=False,
    )
    header, generated_prefix, parameters, primals = _emit_instance_header(
        root, class_name, problem
    )
    clarabel = clarabel.normalized()
    parameter_by_name = {parameter.name(): parameter for parameter in problem.parameters()}
    variable_by_name = {variable.name(): variable for variable in problem.variables()}
    public_parameters = tuple(
        ParameterLayout(
            item.name,
            tuple(int(extent) for extent in parameter_by_name[item.name].shape),
            item.size,
            item.offset,
            item.dirty_blocks,
        )
        for item in parameters
    )
    public_primals = tuple(
        PrimalLayout(
            item.name,
            tuple(int(extent) for extent in variable_by_name[item.name].shape),
            item.size,
        )
        for item in primals
    )
    manifest = {
        "class_name": class_name,
        "prefix": generated_prefix,
        "cvxpy_version": _package_version("cvxpy"),
        "cvxpygen_version": cvxpygen_version,
        "solver": "CLARABEL",
        "clarabel_version": clarabel.version,
        "instance_owned": True,
        "persistent_solver": True,
        "parameters": [
            {
                "name": item.name,
                "shape": list(item.shape),
                "size": item.size,
                "offset": item.offset,
                "dirty_blocks": list(item.dirty_blocks),
                "column_major": item.column_major,
            }
            for item in public_parameters
        ],
        "primals": [
            {
                "name": item.name,
                "shape": list(item.shape),
                "size": item.size,
            }
            for item in public_primals
        ],
    }
    manifest_path = root / "cpp" / "cpg_instance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return GeneratedCvxpygenProgram(
        root,
        header,
        manifest_path,
        class_name,
        generated_prefix,
        public_parameters,
        public_primals,
        clarabel,
    )


def build_current_clarabel(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    cpp_commit: str = _DEFAULT_CLARABEL_CPP_COMMIT,
    rs_tag: str = _DEFAULT_CLARABEL_RS_TAG,
    force: bool = False,
) -> ClarabelNativePaths:
    """Build and cache the current Clarabel C static library.

    The source revisions are pinned.  Set ``CLARABEL_CPP_SOURCE_DIR`` and
    ``CLARABEL_RS_SOURCE_DIR`` to pre-populated checkouts for offline builds.
    """

    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home() / ".cache" / "trading_dsl_engine" / "clarabel" / rs_tag
    ).resolve()
    include = root / "native" / "include"
    library = root / "native" / "lib" / "libclarabel_c.a"
    if include.is_dir() and library.is_file() and not force:
        return ClarabelNativePaths(include, library, rs_tag.removeprefix("v"))
    if force and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    cpp = root / "Clarabel.cpp"
    rs = root / "Clarabel.rs"
    cpp_source = os.environ.get("CLARABEL_CPP_SOURCE_DIR")
    rs_source = os.environ.get("CLARABEL_RS_SOURCE_DIR")
    if cpp_source:
        shutil.copytree(Path(cpp_source), cpp, dirs_exist_ok=True)
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "https://github.com/oxfordcontrol/Clarabel.cpp.git",
                str(cpp),
            ],
            check=True,
        )
        subprocess.run(["git", "checkout", cpp_commit], cwd=cpp, check=True)
    if rs_source:
        shutil.copytree(Path(rs_source), rs, dirs_exist_ok=True)
    else:
        subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                rs_tag,
                "https://github.com/oxfordcontrol/Clarabel.rs.git",
                str(rs),
            ],
            check=True,
        )
    target_rs = cpp / "Clarabel.rs"
    if target_rs.exists():
        shutil.rmtree(target_rs)
    shutil.copytree(rs, target_rs, ignore=shutil.ignore_patterns(".git"))
    subprocess.run(
        [
            "cargo",
            "build",
            "--release",
            "--manifest-path",
            str(cpp / "rust_wrapper" / "Cargo.toml"),
        ],
        check=True,
    )
    include.mkdir(parents=True, exist_ok=True)
    (root / "native" / "lib").mkdir(parents=True, exist_ok=True)
    shutil.copytree(cpp / "include", include, dirs_exist_ok=True)
    shutil.copy2(
        cpp / "rust_wrapper" / "target" / "release" / "libclarabel_c.a",
        library,
    )
    return ClarabelNativePaths(include, library, rs_tag.removeprefix("v"))


def artifact_fingerprint(artifact: GeneratedCvxpygenProgram) -> str:
    digest = hashlib.sha256()
    for path in sorted(artifact.root.rglob("*")):
        if path.is_file() and "solver_code" not in path.parts:
            digest.update(path.relative_to(artifact.root).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update(artifact.clarabel.static_library.read_bytes())
    return digest.hexdigest()


__all__ = [
    "ClarabelNativePaths",
    "GeneratedCvxpygenProgram",
    "ParameterLayout",
    "PrimalLayout",
    "artifact_fingerprint",
    "build_current_clarabel",
    "generate_clarabel_program",
]
