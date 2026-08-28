from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import wraps
from typing import Any


class ResultMapping(OrderedDict):
    """Ordered nested runtime results with recursive mapping helpers."""

    def map(self, function: Callable[[Any], Any]) -> "ResultMapping":
        if not callable(function):
            raise TypeError("ResultMapping.map expects a callable")

        def apply(value: Any) -> Any:
            if isinstance(value, Mapping):
                return type(self)((key, apply(child)) for key, child in value.items())
            return function(value)

        return type(self)((key, apply(value)) for key, value in self.items())

    def flatten(self) -> "ResultMapping":
        flattened = type(self)()

        def visit(value: Mapping[Any, Any], prefix: tuple[Any, ...]) -> None:
            for key, child in value.items():
                path = prefix + (key,)
                if isinstance(child, Mapping):
                    visit(child, path)
                else:
                    flattened[path] = child

        visit(self, ())
        return flattened


FormulaResultMapping = ResultMapping


@dataclass(frozen=True)
class _Leaf:
    index: int


@dataclass(frozen=True)
class _MappingSpec:
    items: tuple[tuple[Any, Any], ...]


def _flatten_formulas(formulas: Mapping[Any, Any]) -> tuple[list[Any], _MappingSpec]:
    leaves: list[Any] = []

    def visit(value: Mapping[Any, Any], path: tuple[Any, ...]) -> _MappingSpec:
        if not value:
            location = "root" if not path else repr(path)
            raise ValueError(f"formula mapping at {location} cannot be empty")
        items = []
        for key, child in value.items():
            if isinstance(child, Mapping):
                node = visit(child, path + (key,))
            else:
                node = _Leaf(len(leaves))
                leaves.append(child)
            items.append((key, node))
        return _MappingSpec(tuple(items))

    return leaves, visit(formulas, ())


def _restore(spec: _MappingSpec, values: Sequence[Any]) -> ResultMapping:
    def visit(node: Any) -> Any:
        if isinstance(node, _Leaf):
            return values[node.index]
        return ResultMapping((key, visit(child)) for key, child in node.items)

    return visit(spec)


def _as_leaves(value: Any, expected: int) -> list[Any]:
    if expected == 1:
        if isinstance(value, Mapping) and len(value) == 1:
            return [next(iter(value.values()))]
        if isinstance(value, (tuple, list)) and len(value) == 1:
            return [value[0]]
        return [value]
    if isinstance(value, Mapping):
        values = list(value.values())
    elif isinstance(value, (tuple, list)):
        values = list(value)
    else:
        candidate = getattr(value, "values", None)
        if callable(candidate):
            candidate = candidate()
        if isinstance(candidate, (tuple, list)):
            values = list(candidate)
        elif not hasattr(value, "shape") and isinstance(value, Sequence):
            values = list(value)
        else:
            raise TypeError(
                "multi-formula result.load() must return an ordered mapping or sequence"
            )
    if len(values) != expected:
        raise ValueError(f"runtime returned {len(values)} outputs for {expected} formulas")
    return values


class _StructuredResult:
    def __init__(self, result: Any, spec: _MappingSpec, leaf_count: int) -> None:
        self._result = result
        self._spec = spec
        self._leaf_count = leaf_count

    def load(self, *args: Any, **kwargs: Any) -> ResultMapping:
        loader = getattr(self._result, "load", None)
        loaded = loader(*args, **kwargs) if callable(loader) else self._result
        return _restore(self._spec, _as_leaves(loaded, self._leaf_count))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._result, name)

    def __enter__(self) -> "_StructuredResult":
        enter = getattr(self._result, "__enter__", None)
        if callable(enter):
            entered = enter()
            if entered is not self._result:
                self._result = entered
        return self

    def __exit__(self, *args: Any) -> Any:
        exit_ = getattr(self._result, "__exit__", None)
        return exit_(*args) if callable(exit_) else None


class _StructuredRuntime:
    def __init__(self, runtime: Any, spec: _MappingSpec, leaf_count: int) -> None:
        self._runtime = runtime
        self._spec = spec
        self._leaf_count = leaf_count

    def run(self, *args: Any, **kwargs: Any) -> _StructuredResult:
        return _StructuredResult(
            self._runtime.run(*args, **kwargs), self._spec, self._leaf_count
        )

    def run_batch(self, *args: Any, **kwargs: Any) -> Any:
        result = self._runtime.run_batch(*args, **kwargs)
        if hasattr(result, "load"):
            return _StructuredResult(result, self._spec, self._leaf_count)
        return _restore(self._spec, _as_leaves(result, self._leaf_count))

    def __getattr__(self, name: str) -> Any:
        return getattr(self._runtime, name)

    def __enter__(self) -> "_StructuredRuntime":
        enter = getattr(self._runtime, "__enter__", None)
        if callable(enter):
            entered = enter()
            if entered is not self._runtime:
                self._runtime = entered
        return self

    def __exit__(self, *args: Any) -> Any:
        exit_ = getattr(self._runtime, "__exit__", None)
        return exit_(*args) if callable(exit_) else None


def support_formula_mappings(compile_function):
    """Add nested mapping support through the existing multi-output compiler."""
    if getattr(compile_function, "_supports_formula_mappings", False):
        return compile_function

    @wraps(compile_function)
    def wrapped(formula, *args, **kwargs):
        if not isinstance(formula, Mapping):
            return compile_function(formula, *args, **kwargs)
        leaves, spec = _flatten_formulas(formula)
        return _StructuredRuntime(
            compile_function(leaves, *args, **kwargs), spec, len(leaves)
        )

    wrapped._supports_formula_mappings = True
    return wrapped


__all__ = ["FormulaResultMapping", "ResultMapping", "support_formula_mappings"]
