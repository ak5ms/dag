from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TypeInfo:
    kind: str  # "vector", "matrix", "scalar"


@dataclass(frozen=True)
class CompiledNode:
    type_info: TypeInfo
    instance_type: object
    ctor: Callable[[], object]


@dataclass(frozen=True)
class OpSpec:
    name: str
    validator: Callable[[list[TypeInfo]], TypeInfo]
    builder: Callable[[list[CompiledNode], list[float]], CompiledNode]


class OpRegistry:
    def __init__(self) -> None:
        self._ops: dict[str, OpSpec] = {}

    def register(self, spec: OpSpec) -> None:
        if spec.name in self._ops:
            raise ValueError(f"Operator already registered: {spec.name}")
        self._ops[spec.name] = spec

    def get(self, name: str) -> OpSpec:
        try:
            return self._ops[name]
        except KeyError as exc:
            raise KeyError(f"Unknown operator '{name}'") from exc


REGISTRY = OpRegistry()
