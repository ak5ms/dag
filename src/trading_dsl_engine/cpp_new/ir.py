"""Immutable, formula-specialized native IR.

The records in this module deliberately contain no executable Python objects.  They
are a stable lowering boundary and can therefore be hashed, inspected and emitted
without importing the native extension or tracing JAX.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json


class ValueKind(str, Enum):
    SCALAR = "scalar"
    INSTRUMENT_VECTOR = "instrument_vector"
    FIXED_VECTOR = "fixed_vector"
    MATRIX = "matrix"
    MODEL = "model"


@dataclass(frozen=True)
class ValueType:
    kind: ValueKind
    width: int | None = None
    rows: int | None = None
    dtype: str = "float64"


@dataclass(frozen=True)
class InputView:
    name: str
    index: int
    value_type: ValueType
    contiguous: bool = True


@dataclass(frozen=True)
class StateSlot:
    node: int
    family: str
    offset: int
    size: int
    alignment: int = 64


@dataclass(frozen=True)
class ScratchSlot:
    node: int
    offset: int
    size: int
    live_from: int
    live_until: int
    color: int
    alignment: int = 64


@dataclass(frozen=True)
class KernelTraits:
    pure: bool
    deterministic_state: bool
    fusion_barrier: bool
    direct_root: bool
    parallel: str = "serial"


@dataclass(frozen=True)
class KernelNode:
    id: int
    source_ids: tuple[int, ...]
    opcode: str
    children: tuple[int, ...]
    value_type: ValueType
    state_slot: int | None
    scratch_slot: int | None
    parameters: tuple[tuple[str, str], ...]
    traits: KernelTraits


@dataclass(frozen=True)
class Projection:
    source: int
    member: str
    value_type: ValueType


@dataclass(frozen=True)
class GraphOutput:
    node: int
    projection: Projection | None
    direct_write: bool


@dataclass(frozen=True)
class Diagnostics:
    source_to_nodes: tuple[tuple[int, tuple[int, ...]], ...]
    removed_projections: int = 0
    constant_folds: int = 0
    stateless_cse: int = 0
    stateful_cse: int = 0
    dead_nodes: int = 0
    aliases_removed: int = 0
    fusion_groups: tuple[tuple[int, ...], ...] = ()
    lifted_lanes: tuple[tuple[int, ...], ...] = ()
    schedule: tuple[str, ...] = ()


@dataclass(frozen=True)
class FormulaIR:
    abi_version: int
    inputs: tuple[InputView, ...]
    nodes: tuple[KernelNode, ...]
    outputs: tuple[GraphOutput, ...]
    states: tuple[StateSlot, ...]
    scratch: tuple[ScratchSlot, ...]
    state_bytes: int
    scratch_bytes: int
    diagnostics: Diagnostics

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()

    def inspect(self) -> dict:
        return json.loads(self.canonical_json())
