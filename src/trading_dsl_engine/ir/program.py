from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from trading_dsl_engine.ir.types import ValueType

if TYPE_CHECKING:
    from trading_dsl_engine.ir.ops import OpSpec


@dataclass(frozen=True, slots=True)
class Node:
    op: "OpSpec"
    child_ids: tuple[int, ...]
    value_type: ValueType


@dataclass(frozen=True, slots=True)
class Program:
    nodes: tuple[Node, ...]
    outputs: tuple[int, ...]
    input_names: tuple[str, ...]

    @property
    def output_id(self) -> int:
        if len(self.outputs) != 1:
            raise ValueError("program does not have exactly one output")
        return self.outputs[0]
