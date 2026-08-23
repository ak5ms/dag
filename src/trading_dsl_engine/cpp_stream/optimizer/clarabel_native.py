from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from math import prod
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Iterable, Mapping

from jinja2 import Environment, FileSystemLoader, StrictUndefined


_DEFAULT_CLARABEL_CPP_COMMIT = "0de6259a3edfd5cc041ec42b2148599ce63e73cb"
_DEFAULT_CLARABEL_RS_TAG = "v0.11.1"
_INFO_FIELDS = (
    ("objective", "obj_val"),
    ("iterations", "iterations"),
    ("status", "status"),
    ("primal_residual", "r_prim"),
    ("dual_residual", "r_dual"),
)


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
class DualLayout:
    name: str
    constraint_index: int
    label: str | None
    shape: tuple[int, ...]
    size: int


@dataclass(frozen=True, slots=True)
class FieldAlias:
    name: str
    primal_name: str


@dataclass(frozen=True, slots=True)
class FieldLayout:
    name: str
    kind: str
    source_name: str
    source_index: int
    offset: int
    count: int
    stride: int
    logical_shape: tuple[int, ...]

    @property
    def primal_name(self) -> str:
        return self.source_name

    @property
    def primal_index(self) -> int:
        return self.source_index


@dataclass(frozen=True, slots=True)
class GeneratedClarabelProgram:
    """Generated direct-Clarabel program with instance-owned mutable state."""

    root: Path
    instance_header: Path
    manifest_path: Path
    class_name: str
    prefix: str
    parameters: tuple[ParameterLayout, ...]
    primals: tuple[PrimalLayout, ...]
    duals: tuple[DualLayout, ...]
    aliases: tuple[FieldAlias, ...]
    clarabel: ClarabelNativePaths
    instrument_count: int | None = None

    @property
    def include_dirs(self) -> tuple[Path, ...]:
        return self.root / "cpp" / "include", self.clarabel.include_dir

    @property
    def link_files(self) -> tuple[Path, ...]:
        return (self.clarabel.static_library,)

    @property
    def fingerprint_files(self) -> tuple[Path, ...]:
        headers = tuple(
            sorted(
                path
                for path in (self.root / "cpp" / "include").rglob("*")
                if path.is_file()
            )
        )
        return *headers, self.manifest_path

    def build_shared_kwargs(self) -> dict[str, tuple[Path, ...]]:
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

    def parameter_index(self, name: str) -> int:
        for index, parameter in enumerate(self.parameters):
            if parameter.name == name:
                return index
        raise KeyError(f"unknown generated parameter {name!r}")

    def parameter_logical_shape(self, name: str) -> tuple[int, ...]:
        shape = self.parameters[self.parameter_index(name)].shape
        return tuple(reversed(shape)) if len(shape) > 1 else shape

    def resolve_field(self, name: str) -> FieldLayout:
        return _resolve_result_field(
            str(name),
            primals=self.primals,
            duals=self.duals,
            aliases=self.aliases,
        )


# Compatibility for the public name used before the direct backend.
GeneratedCvxpygenProgram = GeneratedClarabelProgram


_NO_FIELD_MATCH = object()


def _match_base_field(name: str, base: str) -> str | None | object:
    if name == base:
        return None
    match = re.fullmatch(re.escape(base) + r"\[(\d+)\]", name)
    return _NO_FIELD_MATCH if match is None else match.group(1)


def _indexed_result_layout(
    requested_name: str,
    *,
    kind: str,
    source_name: str,
    source_index: int,
    shape: tuple[int, ...],
    size: int,
    index_text: str | None,
) -> FieldLayout:
    if index_text is None:
        logical_shape = tuple(reversed(shape)) if len(shape) > 1 else shape
        return FieldLayout(
            requested_name,
            kind,
            source_name,
            source_index,
            0,
            size,
            1,
            logical_shape,
        )
    if not shape:
        raise KeyError(f"scalar field {requested_name!r} cannot be indexed")
    index = int(index_text)
    if index >= shape[0]:
        raise KeyError(
            f"field {requested_name!r} indexes axis 0 of size {shape[0]}"
        )
    count = prod(shape[1:]) if len(shape) > 1 else 1
    return FieldLayout(
        requested_name,
        kind,
        source_name,
        source_index,
        index,
        int(count),
        int(shape[0]),
        tuple(reversed(shape[1:])),
    )


def _resolve_result_field(
    name: str,
    *,
    primals: tuple[PrimalLayout, ...],
    duals: tuple[DualLayout, ...],
    aliases: tuple[FieldAlias, ...],
) -> FieldLayout:
    alias_by_name = {alias.name: alias.primal_name for alias in aliases}
    primal_by_name = {primal.name: primal for primal in primals}
    for public_name, primal_name in alias_by_name.items():
        index_text = _match_base_field(name, public_name)
        if index_text is _NO_FIELD_MATCH:
            continue
        primal = primal_by_name.get(primal_name)
        if primal is None:
            raise KeyError(
                f"generated field alias {public_name!r} targets missing primal "
                f"{primal_name!r}"
            )
        return _indexed_result_layout(
            name,
            kind="primal",
            source_name=primal.name,
            source_index=primals.index(primal),
            shape=primal.shape,
            size=primal.size,
            index_text=index_text,
        )
    for primal_index, primal in enumerate(primals):
        index_text = _match_base_field(name, primal.name)
        if index_text is _NO_FIELD_MATCH:
            continue
        return _indexed_result_layout(
            name,
            kind="primal",
            source_name=primal.name,
            source_index=primal_index,
            shape=primal.shape,
            size=primal.size,
            index_text=index_text,
        )
    for dual_index, dual in enumerate(duals):
        bases = [
            dual.name,
            f"dual[{dual.constraint_index}]",
            f"lagrangian[{dual.constraint_index}]",
            f"constraint[{dual.constraint_index}].dual",
            f"constraint[{dual.constraint_index}].lagrangian",
        ]
        if dual.label is not None:
            bases.extend((f"{dual.label}.dual", f"{dual.label}.lagrangian"))
        for base in bases:
            index_text = _match_base_field(name, base)
            if index_text is _NO_FIELD_MATCH:
                continue
            return _indexed_result_layout(
                name,
                kind="dual",
                source_name=dual.name,
                source_index=dual_index,
                shape=dual.shape,
                size=dual.size,
                index_text=index_text,
            )
    info_aliases = {
        "objective_value": "objective",
        "obj_val": "objective",
        "iter": "iterations",
        "pri_res": "primal_residual",
        "dua_res": "dual_residual",
    }
    normalized = info_aliases.get(name, name)
    if normalized.startswith("info."):
        raw = normalized.removeprefix("info.")
        normalized = info_aliases.get(raw, raw)
        for public_name, member_name in _INFO_FIELDS:
            if normalized == member_name:
                normalized = public_name
                break
    for info_index, (public_name, _member_name) in enumerate(_INFO_FIELDS):
        if normalized == public_name:
            return FieldLayout(name, "info", public_name, info_index, 0, 1, 1, ())
    available = [primal.name for primal in primals]
    available.extend(alias.name for alias in aliases)
    available.extend(("dual[index]", "constraint[index].dual"))
    available.extend(name for name, _ in _INFO_FIELDS)
    raise KeyError(
        f"unknown generated field {name!r}; available fields include {available}"
    )


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unknown"


def _safe_identifier(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"invalid {label} {value!r}")
    return value


def _template_environment() -> Environment:
    return Environment(
        loader=FileSystemLoader(Path(__file__).with_name("templates")),
        undefined=StrictUndefined,
        autoescape=False,
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def _constraint_dual_shape(constraint: Any, size: int) -> tuple[int, ...]:
    dual_variables = tuple(getattr(constraint, "dual_variables", ()))
    if len(dual_variables) == 1:
        shape = tuple(int(extent) for extent in dual_variables[0].shape)
        if (prod(shape) if shape else 1) == size:
            return shape
    return (size,)


def _constraint_label(constraint: Any) -> str | None:
    label = getattr(constraint, "label", None)
    if label is None:
        return None
    label = str(label)
    return label if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label) else None


def generate_clarabel_program(
    problem: Any,
    *,
    code_dir: str | os.PathLike[str],
    clarabel: ClarabelNativePaths,
    class_name: str = "GeneratedCvxpyProgram",
    prefix: str = "cpg_",
    instrument_count: int | None = None,
    enable_settings: Iterable[str] = (
        "verbose",
        "max_iter",
        "tol_gap_abs",
        "tol_gap_rel",
        "tol_feas",
        "presolve_enable",
    ),
    field_aliases: Mapping[str, str] | None = None,
    force: bool = False,
    parameter_shard_size: int = 512,
) -> GeneratedClarabelProgram:
    from .direct_clarabel import generate_direct_clarabel_program

    return generate_direct_clarabel_program(
        problem,
        code_dir=code_dir,
        clarabel=clarabel,
        class_name=class_name,
        prefix=prefix,
        instrument_count=instrument_count,
        enable_settings=enable_settings,
        field_aliases=field_aliases,
        force=force,
        parameter_shard_size=parameter_shard_size,
    )


def load_clarabel_program(
    code_dir: str | os.PathLike[str],
    *,
    clarabel: ClarabelNativePaths,
) -> GeneratedClarabelProgram:
    from .direct_clarabel import load_direct_clarabel_program

    return load_direct_clarabel_program(code_dir, clarabel=clarabel)


def _patch_clarabel_allocation_free_timers(source_root: Path) -> None:
    timer_path = source_root / "src" / "timers" / "timers.rs"
    text = timer_path.read_text()
    old = """impl SubTimersMap {
    fn reset_subtimer(&mut self, key: &'static str) {
"""
    new = """impl SubTimersMap {
    fn reset(&mut self) {
        for timer in self.values_mut() {
            timer.reset();
        }
    }

    fn reset_subtimer(&mut self, key: &'static str) {
"""
    if old not in text or "self.subtimers.clear();" not in text:
        if "self.subtimers.reset();" in text:
            return
        raise RuntimeError(
            "Clarabel timer source no longer matches the reviewed "
            "allocation-free reset patch"
        )
    timer_path.write_text(
        text.replace(old, new, 1).replace(
            "self.subtimers.clear();", "self.subtimers.reset();", 1
        )
    )


def build_current_clarabel(
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    cpp_commit: str = _DEFAULT_CLARABEL_CPP_COMMIT,
    rs_tag: str = _DEFAULT_CLARABEL_RS_TAG,
    force: bool = False,
) -> ClarabelNativePaths:
    """Build and cache the pinned allocation-free Clarabel C static library."""

    build_id = f"Clarabel.rs {rs_tag} + allocation-free timer reset v1\n"
    root = (
        Path(cache_dir).expanduser()
        if cache_dir is not None
        else Path.home()
        / ".cache"
        / "trading_dsl_engine"
        / "clarabel"
        / f"{rs_tag}-noalloc1"
    ).resolve()
    include = root / "native" / "include"
    library = root / "native" / "lib" / "libclarabel_c.a"
    marker = root / "native" / "BUILD_ID"
    if (
        include.is_dir()
        and library.is_file()
        and marker.is_file()
        and marker.read_text() == build_id
        and not force
    ):
        return ClarabelNativePaths(include, library, rs_tag.removeprefix("v"))
    if root.exists():
        if not force:
            raise RuntimeError(
                f"Clarabel cache {root} exists but is incomplete; pass force=True "
                "to replace that dedicated cache directory"
            )
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
    _patch_clarabel_allocation_free_timers(target_rs)
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
    library.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cpp / "include", include, dirs_exist_ok=True)
    shutil.copy2(
        cpp / "rust_wrapper" / "target" / "release" / "libclarabel_c.a",
        library,
    )
    marker.write_text(build_id)
    return ClarabelNativePaths(include, library, rs_tag.removeprefix("v"))


def artifact_fingerprint(artifact: GeneratedClarabelProgram) -> str:
    digest = hashlib.sha256()
    for path in sorted(artifact.root.rglob("*")):
        if path.is_file():
            digest.update(path.relative_to(artifact.root).as_posix().encode())
            digest.update(path.read_bytes())
    digest.update(artifact.clarabel.static_library.read_bytes())
    return digest.hexdigest()


__all__ = [
    "ClarabelNativePaths",
    "DualLayout",
    "FieldAlias",
    "FieldLayout",
    "GeneratedClarabelProgram",
    "GeneratedCvxpygenProgram",
    "ParameterLayout",
    "PrimalLayout",
    "artifact_fingerprint",
    "build_current_clarabel",
    "generate_clarabel_program",
    "load_clarabel_program",
]
