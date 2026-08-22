from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from trading_dsl_engine.base.dsl import ensure_expr
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.cpp_stream.optimizer.cvxpygen_native import (
    GeneratedCvxpygenProgram,
)


@dataclass(frozen=True, eq=False)
class CvxpygenProgramExpr(Expr):
    """Object-valued call to one generated CVXPYgen program."""

    program: GeneratedCvxpygenProgram
    bindings: tuple[tuple[str, Expr], ...]


@dataclass(frozen=True, eq=False)
class CvxpygenFieldExpr(Expr):
    """Named compile-time projection from a generated CVXPYgen program."""

    program_expr: CvxpygenProgramExpr
    field: str


def bind_program(
    program: GeneratedCvxpygenProgram,
    /,
    **parameters: Expr | int | float,
) -> CvxpygenProgramExpr:
    """Bind DAG expressions to every generated parameter by name.

    Values remain expressions in the cpp_stream graph. They are copied directly
    into the generated CVXPYgen parameter buffer inside the runner's single row
    loop; this function never evaluates or materializes them in Python.
    """

    expected = tuple(parameter.name for parameter in program.parameters)
    missing = sorted(set(expected) - set(parameters))
    extra = sorted(set(parameters) - set(expected))
    if missing or extra:
        raise KeyError(
            f"generated parameter mismatch: missing={missing}, extra={extra}"
        )
    return CvxpygenProgramExpr(
        program,
        tuple((name, ensure_expr(parameters[name])) for name in expected),
    )


def get_field(program_expr: CvxpygenProgramExpr, field: str) -> CvxpygenFieldExpr:
    """Project a named primal or ``primal[index]`` from one native solve."""

    if not isinstance(program_expr, CvxpygenProgramExpr):
        raise TypeError("get_field expects a CVXPYgen program expression")
    program_expr.program.resolve_field(field)
    return CvxpygenFieldExpr(program_expr, str(field))


__all__ = [
    "CvxpygenFieldExpr",
    "CvxpygenProgramExpr",
    "bind_program",
    "get_field",
]
