from __future__ import annotations

from dataclasses import dataclass, field

from trading_dsl_engine.base.parser import Expr


@dataclass(frozen=True, eq=False)
class CvxpyProgramExpr(Expr):
    """Backend-neutral object-valued optimizer program expression."""

    program: object
    bindings: tuple[tuple[str, Expr], ...]
    requested_fields: set[str] = field(default_factory=set)


@dataclass(frozen=True, eq=False)
class CvxpyFieldExpr(Expr):
    """Named compile-time projection from an optimizer program."""

    program_expr: CvxpyProgramExpr
    field: str


@dataclass(frozen=True, eq=False)
class CvxpyPreviousSolutionExpr(Expr):
    """A delayed edge from the preceding solve into the next parameter set."""

    field: str
    initial: Expr


__all__ = [
    "CvxpyFieldExpr",
    "CvxpyPreviousSolutionExpr",
    "CvxpyProgramExpr",
]
