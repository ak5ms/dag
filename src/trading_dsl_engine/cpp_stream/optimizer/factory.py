from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import hashlib
import inspect
import json
import marshal
import os
from pathlib import Path
import re
from threading import RLock
from typing import Any

from trading_dsl_engine.base.dsl import ensure_expr
from trading_dsl_engine.cpp_stream.optimizer.cvxpygen_native import (
    ClarabelNativePaths,
    DualLayout,
    FieldAlias,
    GeneratedCvxpygenProgram,
    ParameterLayout,
    PrimalLayout,
    _constraint_dual_shape,
    _constraint_label,
    _resolve_result_field,
    build_current_clarabel,
    generate_clarabel_program,
    load_clarabel_program,
)
from trading_dsl_engine.cpp_stream.optimizer.dsl import CvxpygenProgramExpr
from trading_dsl_engine.ir.types import ValueType


_SYMBOLIC_INSTRUMENT_COUNT = 113
_FACTORY_CACHE_SCHEMA = 1
_PROGRAM_CACHE_LOCK = RLock()
_DEFAULT_ENABLE_SETTINGS = (
    "verbose",
    "max_iter",
    "tol_gap_abs",
    "tol_gap_rel",
    "tol_feas",
    "presolve_enable",
)


@dataclass(frozen=True, slots=True)
class CvxpygenProgramPrototype:
    """Shape-only program used during cpp_stream's input-discovery pass."""

    class_name: str
    prefix: str
    parameters: tuple[ParameterLayout, ...]
    primals: tuple[PrimalLayout, ...]
    duals: tuple[DualLayout, ...]
    aliases: tuple[FieldAlias, ...]
    instrument_count: int

    def parameter_index(self, name: str) -> int:
        for index, parameter in enumerate(self.parameters):
            if parameter.name == name:
                return index
        raise KeyError(f"unknown generated parameter {name!r}")

    def parameter_logical_shape(self, name: str) -> tuple[int, ...]:
        shape = self.parameters[self.parameter_index(name)].shape
        return tuple(reversed(shape)) if len(shape) > 1 else shape

    def resolve_field(self, name: str):
        return _resolve_result_field(
            str(name),
            primals=self.primals,
            duals=self.duals,
            aliases=self.aliases,
        )


def _default_cache_root() -> Path:
    configured = os.environ.get("TRADING_DSL_ENGINE_CVXPYGEN_CACHE")
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".cache" / "trading_dsl_engine" / "cvxpygen"
    )


def _default_clarabel() -> ClarabelNativePaths:
    include = os.environ.get("CLARABEL_INCLUDE_DIR")
    library = os.environ.get("CLARABEL_STATIC_LIBRARY")
    if include and library:
        return ClarabelNativePaths(Path(include), Path(library)).normalized()
    if bool(include) != bool(library):
        raise ValueError(
            "set both CLARABEL_INCLUDE_DIR and CLARABEL_STATIC_LIBRARY"
        )
    return build_current_clarabel()


@contextmanager
def _exclusive_cache_lock(path: Path):
    """Serialize one sub-program cache key across definitions and processes."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as handle:
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows has process-local fallback.
            yield
            return
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_]", "_", value)
    if not stem or stem[0].isdigit():
        stem = f"program_{stem}"
    return stem


def _class_stem(value: str) -> str:
    parts = [part for part in re.split(r"[^A-Za-z0-9]+", value) if part]
    result = "".join(part[:1].upper() + part[1:] for part in parts)
    return result or "CvxpyProgram"


def _factory_fingerprint(factory: Callable[..., Any]) -> str:
    digest = hashlib.sha256()
    digest.update(factory.__module__.encode())
    digest.update(factory.__qualname__.encode())
    digest.update(marshal.dumps(factory.__code__))
    digest.update(repr(factory.__defaults__).encode())
    digest.update(repr(factory.__kwdefaults__).encode())
    closure = factory.__closure__ or ()
    digest.update(
        repr(tuple(repr(cell.cell_contents) for cell in closure)).encode()
    )
    globals_used: list[tuple[str, str]] = []
    for name in factory.__code__.co_names:
        value = factory.__globals__.get(name)
        if isinstance(value, (str, int, float, bool, tuple, frozenset, type(None))):
            globals_used.append((name, repr(value)))
    digest.update(repr(globals_used).encode())
    return digest.hexdigest()


def _constraint_value_expression(cp, constraint):
    try:
        return constraint.expr
    except (AttributeError, ValueError):
        pass
    parts = tuple(
        cp.reshape(argument, (argument.size,), order="F")
        for argument in constraint.args
    )
    if not parts:
        raise ValueError(
            f"constraint {type(constraint).__name__} exposes no numeric arguments"
        )
    return parts[0] if len(parts) == 1 else cp.hstack(parts)


def _requested_constraint_values(
    problem,
    requested_fields: frozenset[str],
) -> tuple[int, ...]:
    labels = {
        label: index
        for index, constraint in enumerate(problem.constraints)
        if (label := _constraint_label(constraint)) is not None
    }
    result: set[int] = set()
    for field in requested_fields:
        indexed = re.fullmatch(
            r"constraint\[(\d+)\]\.value(?:\[\d+\])?", field
        )
        if indexed is not None:
            index = int(indexed.group(1))
            if index >= len(problem.constraints):
                raise KeyError(
                    f"constraint field {field!r} indexes {len(problem.constraints)} "
                    "constraints"
                )
            result.add(index)
            continue
        labeled = re.fullmatch(
            r"([A-Za-z_][A-Za-z0-9_]*)\.value(?:\[\d+\])?", field
        )
        if labeled is not None:
            label = labeled.group(1)
            if label not in labels:
                raise KeyError(f"unknown labeled constraint {label!r}")
            result.add(labels[label])
    return tuple(sorted(result))


def _augment_constraint_values(cp, problem, requested_fields):
    requested = _requested_constraint_values(problem, requested_fields)
    if not requested:
        return problem, {}
    constraints = list(problem.constraints)
    aliases: dict[str, str] = {}
    variable_names = {variable.name() for variable in problem.variables()}
    for index in requested:
        constraint = problem.constraints[index]
        expression = _constraint_value_expression(cp, constraint)
        base_name = f"cpp_stream_constraint_value_{index}"
        name = base_name
        suffix = 1
        while name in variable_names:
            suffix += 1
            name = f"{base_name}_{suffix}"
        variable_names.add(name)
        value = cp.Variable(expression.shape, name=name)
        constraints.append(value == expression)
        aliases[f"constraint[{index}].value"] = name
        label = _constraint_label(constraint)
        if label is not None:
            aliases[f"{label}.value"] = name
    return cp.Problem(problem.objective, constraints), aliases


def _call_with_named_values(factory, signature, values):
    positional = []
    keywords = {}
    for parameter in signature.parameters.values():
        value = values[parameter.name]
        if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
            positional.append(value)
        else:
            keywords[parameter.name] = value
    return factory(*positional, **keywords)


class CvxpygenProgramDefinition:
    """A cached CVXPY problem factory that is callable from the formula DSL."""

    def __init__(
        self,
        factory: Callable[..., Any],
        *,
        cache_dir: str | os.PathLike[str] | None = None,
        clarabel: ClarabelNativePaths | Callable[[], ClarabelNativePaths] | None = None,
        class_name: str | None = None,
        prefix: str | None = None,
        parameter_options: Mapping[str, Mapping[str, Any]] | None = None,
        enable_settings: tuple[str, ...] = _DEFAULT_ENABLE_SETTINGS,
    ) -> None:
        self.factory = factory
        self.signature = inspect.signature(factory)
        for parameter in self.signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                raise TypeError("CVXPY program factories cannot use *args or **kwargs")
        self.cache_dir = (
            Path(cache_dir).expanduser() if cache_dir is not None else None
        )
        self.clarabel = clarabel
        self.configured_class_name = class_name
        self.configured_prefix = prefix
        self.parameter_options = {
            str(name): dict(options)
            for name, options in (parameter_options or {}).items()
        }
        unknown_options = sorted(
            set(self.parameter_options) - set(self.signature.parameters)
        )
        if unknown_options:
            raise KeyError(
                f"parameter_options contains unknown arguments {unknown_options}"
            )
        self.enable_settings = tuple(enable_settings)
        self._factory_hash = _factory_fingerprint(factory)
        self._lock = RLock()
        self._resolved: dict[str, GeneratedCvxpygenProgram] = {}
        functools.update_wrapper(self, factory)

    @property
    def expression_key(self) -> tuple[str, str, str]:
        return (
            self.factory.__module__,
            self.factory.__qualname__,
            self._factory_hash,
        )

    def validate_field_request(self, field: str) -> None:
        if (
            not field
            or len(field) > 256
            or any(character.isspace() for character in field)
        ):
            raise KeyError(f"invalid generated field request {field!r}")

    def __call__(self, *args, **kwargs) -> CvxpygenProgramExpr:
        bound = self.signature.bind(*args, **kwargs)
        bound.apply_defaults()
        missing = [
            name
            for name in self.signature.parameters
            if name not in bound.arguments
        ]
        if missing:
            raise TypeError(f"missing CVXPY program arguments {missing}")
        return CvxpygenProgramExpr(
            self,
            tuple(
                (name, ensure_expr(bound.arguments[name]))
                for name in self.signature.parameters
            ),
        )

    def _instantiate_problem(
        self,
        parameter_types: Mapping[str, ValueType],
        *,
        requested_fields: frozenset[str],
        n_instruments: int,
    ):
        import cvxpy as cp

        parameters = {}
        for name in self.signature.parameters:
            value_type = parameter_types[name]
            logical_shape = tuple(
                n_instruments if extent is None else int(extent)
                for extent in value_type.logical_shape
            )
            cvxpy_shape = (
                tuple(reversed(logical_shape))
                if len(logical_shape) > 1
                else logical_shape
            )
            parameters[name] = cp.Parameter(
                cvxpy_shape,
                name=name,
                **self.parameter_options.get(name, {}),
            )
        problem = _call_with_named_values(
            self.factory, self.signature, parameters
        )
        if not isinstance(problem, cp.Problem):
            raise TypeError(
                f"{self.factory.__qualname__} must return cvxpy.Problem, "
                f"got {type(problem).__name__}"
            )
        problem, aliases = _augment_constraint_values(
            cp, problem, requested_fields
        )
        problem_parameter_names = {
            parameter.name() for parameter in problem.parameters()
        }
        expected_names = set(parameters)
        missing = sorted(expected_names - problem_parameter_names)
        extra = sorted(problem_parameter_names - expected_names)
        if missing or extra:
            raise KeyError(
                "CVXPY problem parameters must be exactly its factory arguments: "
                f"missing={missing}, extra={extra}"
            )
        return problem, aliases

    def _prototype(self, problem, aliases, n_instruments):
        offset = 0
        parameters = []
        for parameter in problem.parameters():
            shape = tuple(int(extent) for extent in parameter.shape)
            size = int(parameter.size)
            parameters.append(
                ParameterLayout(parameter.name(), shape, size, offset, ())
            )
            offset += size
        primals = tuple(
            PrimalLayout(
                variable.name(),
                tuple(int(extent) for extent in variable.shape),
                int(variable.size),
            )
            for variable in problem.variables()
        )
        duals = tuple(
            DualLayout(
                f"d{index}",
                index,
                _constraint_label(constraint),
                _constraint_dual_shape(
                    constraint,
                    sum(int(variable.size) for variable in constraint.dual_variables),
                ),
                sum(int(variable.size) for variable in constraint.dual_variables),
            )
            for index, constraint in enumerate(problem.constraints)
        )
        return CvxpygenProgramPrototype(
            "CvxpygenPrototype",
            "prototype_",
            tuple(parameters),
            primals,
            duals,
            tuple(
                FieldAlias(name, primal_name)
                for name, primal_name in sorted(aliases.items())
            ),
            n_instruments,
        )

    def _cache_key(self, problem, parameter_types, n_instruments: int) -> str:
        payload = {
            "cache_schema": _FACTORY_CACHE_SCHEMA,
            "factory": self._factory_hash,
            "instrument_count": int(n_instruments),
            "parameters": {
                name: {
                    "shape": list(value_type.logical_shape),
                    "cvxpy_shape": list(
                        next(
                            parameter.shape
                            for parameter in problem.parameters()
                            if parameter.name() == name
                        )
                    ),
                    "dtype": value_type.dtype,
                    "options": self.parameter_options.get(name, {}),
                }
                for name, value_type in parameter_types.items()
            },
            "problem": str(problem),
            "variables": [
                (variable.name(), tuple(variable.shape))
                for variable in problem.variables()
            ],
            "constraints": [
                (
                    type(constraint).__name__,
                    tuple(constraint.shape),
                    _constraint_label(constraint),
                )
                for constraint in problem.constraints
            ],
            "enable_settings": self.enable_settings,
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, default=repr).encode()
        ).hexdigest()

    def _clarabel_paths(self) -> ClarabelNativePaths:
        if self.clarabel is None:
            return _default_clarabel()
        if callable(self.clarabel):
            return self.clarabel().normalized()
        return self.clarabel.normalized()

    def resolve_for_types(
        self,
        parameter_types: Mapping[str, ValueType],
        *,
        requested_fields: frozenset[str],
        n_instruments: int | None,
    ) -> GeneratedCvxpygenProgram | CvxpygenProgramPrototype:
        expected = tuple(self.signature.parameters)
        missing = sorted(set(expected) - set(parameter_types))
        extra = sorted(set(parameter_types) - set(expected))
        if missing or extra:
            raise KeyError(
                f"CVXPY program argument mismatch: missing={missing}, extra={extra}"
            )
        resolved_n = (
            _SYMBOLIC_INSTRUMENT_COUNT
            if n_instruments is None
            else int(n_instruments)
        )
        problem, aliases = self._instantiate_problem(
            parameter_types,
            requested_fields=requested_fields,
            n_instruments=resolved_n,
        )
        if n_instruments is None:
            return self._prototype(problem, aliases, resolved_n)

        cache_key = self._cache_key(
            problem, parameter_types, int(n_instruments)
        )
        with self._lock, _PROGRAM_CACHE_LOCK:
            cached = self._resolved.get(cache_key)
            if cached is not None:
                return cached
            stem = _safe_stem(self.factory.__qualname__)
            root = (self.cache_dir or _default_cache_root()) / stem / cache_key
            lock_path = root.parent / f".{cache_key}.lock"
            with _exclusive_cache_lock(lock_path):
                clarabel = self._clarabel_paths()
                if root.is_dir():
                    try:
                        artifact = load_clarabel_program(root, clarabel=clarabel)
                        self._resolved[cache_key] = artifact
                        return artifact
                    except (FileNotFoundError, KeyError, TypeError, ValueError):
                        force = True
                else:
                    force = False
                short_hash = cache_key[:12]
                class_name = self.configured_class_name or (
                    f"{_class_stem(self.factory.__name__)}_{short_hash}"
                )
                prefix = self.configured_prefix or (
                    f"{_safe_stem(self.factory.__name__).lower()}_{short_hash}_"
                )
                artifact = generate_clarabel_program(
                    problem,
                    code_dir=root,
                    clarabel=clarabel,
                    class_name=class_name,
                    prefix=prefix,
                    instrument_count=int(n_instruments),
                    enable_settings=self.enable_settings,
                    field_aliases=aliases,
                    force=force,
                )
                self._resolved[cache_key] = artifact
                return artifact


def clarabel_program(
    factory: Callable[..., Any] | None = None,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    clarabel: ClarabelNativePaths | Callable[[], ClarabelNativePaths] | None = None,
    class_name: str | None = None,
    prefix: str | None = None,
    parameter_options: Mapping[str, Mapping[str, Any]] | None = None,
    enable_settings: tuple[str, ...] = _DEFAULT_ENABLE_SETTINGS,
):
    """Decorate ``(**cvxpy.Parameters) -> cvxpy.Problem`` for DSL use."""

    def decorate(function):
        return CvxpygenProgramDefinition(
            function,
            cache_dir=cache_dir,
            clarabel=clarabel,
            class_name=class_name,
            prefix=prefix,
            parameter_options=parameter_options,
            enable_settings=enable_settings,
        )

    return decorate if factory is None else decorate(factory)


__all__ = [
    "CvxpygenProgramDefinition",
    "CvxpygenProgramPrototype",
    "clarabel_program",
]
