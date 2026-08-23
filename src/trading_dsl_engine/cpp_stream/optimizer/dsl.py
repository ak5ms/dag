from __future__ import annotations

from dataclasses import dataclass, field

from trading_dsl_engine.base.dsl import ensure_expr
from trading_dsl_engine.base.parser import Expr


@dataclass(frozen=True, eq=False)
class CvxpyProgramExpr(Expr):
    """Object-valued call to one generated native CVXPY program."""

    program: object
    bindings: tuple[tuple[str, Expr], ...]
    requested_fields: set[str] = field(default_factory=set)


@dataclass(frozen=True, eq=False)
class CvxpyFieldExpr(Expr):
    """Named compile-time projection from a generated Clarabel program."""

    program_expr: CvxpyProgramExpr
    field: str


@dataclass(frozen=True, eq=False)
class CvxpyPreviousSolutionExpr(Expr):
    """A delayed edge from the preceding solve into the next parameter set."""

    field: str
    initial: Expr


def previous_solution(
    field: str,
    *,
    initial: Expr | int | float = 0.0,
) -> CvxpyPreviousSolutionExpr:
    """Feed a prior primal field into the next row's optimizer parameters.

    ``initial`` supplies the first row. A scalar initial value broadcasts over
    the bound parameter; otherwise its logical shape must match exactly.
    """

    field = str(field)
    if not field or any(character.isspace() for character in field):
        raise KeyError(f"invalid previous solution field {field!r}")
    return CvxpyPreviousSolutionExpr(field, ensure_expr(initial))


def get_field(program_expr: CvxpyProgramExpr, field: str) -> CvxpyFieldExpr:
    """Project a named primal, constraint result, dual, or solver diagnostic."""

    if not isinstance(program_expr, CvxpyProgramExpr):
        raise TypeError("get_field expects a CVXPY program expression")
    field = str(field)
    validator = getattr(program_expr.program, "validate_field_request", None)
    if validator is None:
        raise TypeError("unsupported CVXPY program definition")
    validator(field)
    program_expr.requested_fields.add(field)
    return CvxpyFieldExpr(program_expr, field)


__all__ = [
    "CvxpyFieldExpr",
    "CvxpyPreviousSolutionExpr",
    "CvxpyProgramExpr",
    "get_field",
    "previous_solution",
]
