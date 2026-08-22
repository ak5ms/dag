from __future__ import annotations

from dataclasses import dataclass, field

from trading_dsl_engine.base.dsl import ensure_expr
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.cpp_stream.optimizer.cvxpygen_native import (
    GeneratedCvxpygenProgram,
)


@dataclass(frozen=True, eq=False)
class CvxpygenProgramExpr(Expr):
    """Object-valued call to one generated CVXPYgen program."""

    program: object
    bindings: tuple[tuple[str, Expr], ...]
    requested_fields: set[str] = field(default_factory=set)


@dataclass(frozen=True, eq=False)
class CvxpygenFieldExpr(Expr):
    """Named compile-time projection from a generated CVXPYgen program."""

    program_expr: CvxpygenProgramExpr
    field: str


@dataclass(frozen=True, eq=False)
class CvxpygenPreviousSolutionExpr(Expr):
    """A delayed edge from the preceding solve into the next parameter set."""

    field: str
    initial: Expr


def previous_solution(
    field: str,
    *,
    initial: Expr | int | float = 0.0,
) -> CvxpygenPreviousSolutionExpr:
    """Feed a prior primal field into the next row's optimizer parameters.

    ``initial`` supplies the first row. A scalar initial value broadcasts over
    the bound parameter; otherwise its logical shape must match exactly.
    """

    field = str(field)
    if not field or any(character.isspace() for character in field):
        raise KeyError(f"invalid previous solution field {field!r}")
    return CvxpygenPreviousSolutionExpr(field, ensure_expr(initial))


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
    """Project a named primal, constraint result, dual, or solver diagnostic."""

    if not isinstance(program_expr, CvxpygenProgramExpr):
        raise TypeError("get_field expects a CVXPYgen program expression")
    field = str(field)
    if isinstance(program_expr.program, GeneratedCvxpygenProgram):
        program_expr.program.resolve_field(field)
    else:
        validator = getattr(program_expr.program, "validate_field_request", None)
        if validator is None:
            raise TypeError("unsupported deferred CVXPYgen program definition")
        validator(field)
        program_expr.requested_fields.add(field)
    return CvxpygenFieldExpr(program_expr, field)


__all__ = [
    "CvxpygenFieldExpr",
    "CvxpygenPreviousSolutionExpr",
    "CvxpygenProgramExpr",
    "bind_program",
    "get_field",
    "previous_solution",
]
