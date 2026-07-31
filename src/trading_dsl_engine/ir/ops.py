from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from trading_dsl_engine.ir.program import Program


@dataclass(frozen=True, slots=True)
class InputOp:
    input_index: int
    name: str


@dataclass(frozen=True, slots=True)
class LiteralOp:
    value: float


@dataclass(frozen=True, slots=True)
class NaryOp:
    name: str
    arity: int


@dataclass(frozen=True, slots=True)
class CumsumOp:
    pass


@dataclass(frozen=True, slots=True)
class EwmOp:
    span: float
    min_periods: int = 0
    ignore_na: bool = True
    adjust: bool = False


@dataclass(frozen=True, slots=True)
class XsRankOp:
    pass


@dataclass(frozen=True, slots=True)
class GroupByOp:
    """Backend-neutral grouped RHS graph.

    ``children`` on the owning Node are ordered as dynamic keys, lhs, then
    captures used by ``inner_program``. Inner input 0 is ``self_``; inner
    inputs 1..N correspond to captures in the same order.
    """

    n_dynamic_keys: int
    static_groups: tuple[tuple[int, ...], ...] | None
    inner_program: "Program"
    capacity: int | None = None
    hash_capacity: int | None = None


OpSpec: TypeAlias = InputOp | LiteralOp | NaryOp | CumsumOp | EwmOp | XsRankOp | GroupByOp
