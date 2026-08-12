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

    def plot(
        self,
        backend: str = "pydot",
        *,
        show: bool = True,
        rankdir: str = "LR",
        figsize: tuple[float, float] | None = None,
    ):
        """Plot this neutral IR DAG and return the backend graph object."""
        from trading_dsl_engine.visualization import plot

        return plot(
            self,
            backend=backend,
            show=show,
            rankdir=rankdir,
            figsize=figsize,
        )
