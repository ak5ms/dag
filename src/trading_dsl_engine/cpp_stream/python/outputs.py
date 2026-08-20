from __future__ import annotations

from dataclasses import dataclass

from trading_dsl_engine.ir.ops import EmitOp, ReductionOp
from trading_dsl_engine.ir.program import Program
from trading_dsl_engine.ir.types import resolve_shape, shape_size


@dataclass(frozen=True, slots=True)
class FormulaOutput:
    """Compile-time layout metadata for one public formula output.

    ``offset`` and ``size`` are measured in float64 values within the row or final
    region. ``lane_partitionable`` records whether the logical tensor has the
    instrument axis first, so generated C++ can prove lane-sharded writes safe per
    output instead of inferring safety from the aggregate packed width.
    """

    root_id: int
    shape: tuple[int, ...]
    size: int
    mode: str
    offset: int
    lane_partitionable: bool


@dataclass(frozen=True, slots=True)
class OutputLayout:
    outputs: tuple[FormulaOutput, ...]
    row_width: int
    final_width: int

    @property
    def mode(self) -> str:
        if self.row_width and self.final_width:
            return "mixed"
        return "final" if self.final_width else "rows"

    @property
    def row_lane_partitionable(self) -> bool:
        return all(
            output.mode == "final" or output.lane_partitionable
            for output in self.outputs
        )

    def storage_size(self, rows: int) -> int:
        return int(rows) * self.row_width + self.final_width


def _output_mode(program: Program, root_id: int) -> str:
    op = program.nodes[root_id].op
    if isinstance(op, EmitOp) or (
        isinstance(op, ReductionOp) and op.temporal
    ):
        return "final"
    return "rows"


def build_output_layout(program: Program, n_instruments: int) -> OutputLayout:
    """Resolve every public root into one exact compile-time packed layout."""

    if not program.outputs:
        raise ValueError("cpp_stream program requires at least one output")

    row_offset = 0
    final_offset = 0
    outputs: list[FormulaOutput] = []
    for root_id in program.outputs:
        value_type = program.nodes[root_id].value_type
        logical_shape = value_type.logical_shape
        shape = resolve_shape(value_type, n_instruments)
        size = shape_size(shape)
        mode = _output_mode(program, root_id)
        if mode == "rows":
            offset = row_offset
            row_offset += size
        else:
            offset = final_offset
            final_offset += size
        outputs.append(
            FormulaOutput(
                root_id,
                shape,
                size,
                mode,
                offset,
                bool(
                    logical_shape
                    and logical_shape[0] is None
                    and shape
                    and shape[0] == n_instruments
                    and size % n_instruments == 0
                ),
            )
        )

    return OutputLayout(tuple(outputs), row_offset, final_offset)


__all__ = ["FormulaOutput", "OutputLayout", "build_output_layout"]
