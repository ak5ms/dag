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
    # Preserve whether the user wrote/provided an integer or floating literal.
    # Physical backends may therefore keep integer expressions integer instead
    # of eagerly converting every constant and input to float64.
    value: int | float


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
class GroupKeySpec:
    """Metadata aligned with one dynamic child of a ``GroupByOp``.

    ``num_keys`` describes a bounded consecutive integer domain. When it is
    present, valid non-NaN values are exactly
    ``[offset, offset + num_keys)``. Dense routing uses ``value - offset`` as
    that key's zero-based mixed-radix digit. For example, ``num_keys=12`` and
    ``offset=1`` describe months 1 through 12. NaN is one additional category
    for floating-point keys.

    ``row_scalar`` says one key value applies to every lane in an input row.
    ``None`` means infer it; ``True`` is an assertion that permits one evaluation
    and one group-slot lookup per row.

    ``dtype`` is the expected native scalar type of the completed key expression.
    It is validated, not used as permission to cast the expression.
    """

    num_keys: int | None = None
    offset: int = 0
    row_scalar: bool | None = None
    dtype: str | None = None


@dataclass(frozen=True, slots=True)
class GroupByOp:
    """Backend-neutral grouped RHS graph.

    ``children`` on the owning Node are ordered as dynamic keys, lhs, then
    captures used by ``inner_program``. ``key_specs`` is aligned with the dynamic
    key prefix. Inner input 0 is ``self_``; inner inputs 1..N correspond to
    captures in the same order.
    """

    key_specs: tuple[GroupKeySpec, ...]
    static_groups: tuple[tuple[int, ...], ...] | None
    inner_program: "Program"
    capacity: int | None = None
    hash_capacity: int | None = None

    @property
    def n_dynamic_keys(self) -> int:
        return len(self.key_specs)


OpSpec: TypeAlias = InputOp | LiteralOp | NaryOp | CumsumOp | EwmOp | XsRankOp | GroupByOp
