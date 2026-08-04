from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

from trading_dsl_engine.ir.einsum import EinsumSpec

if TYPE_CHECKING:
    from trading_dsl_engine.ir.program import Program


@dataclass(frozen=True, slots=True)
class InputOp:
    input_index: int
    name: str


@dataclass(frozen=True, slots=True)
class LiteralOp:
    value: int | float


@dataclass(frozen=True, slots=True)
class NaryOp:
    name: str
    arity: int


@dataclass(frozen=True, slots=True)
class CustomCallOp:
    """Named backend-neutral stateless call.

    Python/JAX backends may execute the original callable stored on the AST.
    Native backends select an implementation by this stable name.
    """

    name: str
    arity: int


@dataclass(frozen=True, slots=True)
class CatOp:
    child_widths: tuple[int, ...]

    @property
    def width(self) -> int:
        return sum(self.child_widths)


@dataclass(frozen=True, slots=True)
class CumsumOp:
    pass


@dataclass(frozen=True, slots=True)
class ReductionOp:
    kind: str
    axes: tuple[int, ...]
    ddof: int = 0
    ignore_na: bool = True

    def __post_init__(self) -> None:
        if self.kind not in {"sum", "mean", "std"}:
            raise ValueError(f"unsupported reduction kind {self.kind!r}")
        if self.ddof < 0:
            raise ValueError("reduction ddof must be >= 0")

    @property
    def temporal(self) -> bool:
        return 0 in self.axes


@dataclass(frozen=True, slots=True)
class EmitOp:
    mode: str = "last"

    def __post_init__(self) -> None:
        if self.mode != "last":
            raise ValueError(f"unsupported emit mode {self.mode!r}")


@dataclass(frozen=True, slots=True)
class FFillOp:
    limit: int | None = None


@dataclass(frozen=True, slots=True)
class ShiftOp:
    lag: int = 1
    max_lag: int = 1


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
class RbfBasisOp:
    n_basis: int


@dataclass(frozen=True, slots=True)
class FutureRbfBasisSumOp:
    n_basis: int
    n_steps: int


@dataclass(frozen=True, slots=True)
class EinsumOp:
    spec: EinsumSpec

    @property
    def subscripts(self) -> str:
        return self.spec.subscripts


@dataclass(frozen=True, slots=True)
class InstrumentBasisMeanOp:
    feature_width: int
    has_weights: bool


@dataclass(frozen=True, slots=True)
class InstrumentBasisProjectionOp:
    field: str  # beta or preds

    def __post_init__(self) -> None:
        if self.field not in {"beta", "preds"}:
            raise ValueError(f"unsupported InstrumentBasisMean projection {self.field!r}")


@dataclass(frozen=True, slots=True)
class RidgeOp:
    feature_widths: tuple[int, ...]
    has_weights: bool
    nonneg: bool = False
    is_stateful: bool = True

    @property
    def coefficient_width(self) -> int:
        return sum(self.feature_widths)


@dataclass(frozen=True, slots=True)
class RidgeProjectionOp:
    field: str  # beta or preds

    def __post_init__(self) -> None:
        if self.field not in {"beta", "preds"}:
            raise ValueError(f"unsupported Ridge projection {self.field!r}")


@dataclass(frozen=True, slots=True)
class GroupKeySpec:
    num_keys: int | None = None
    offset: int = 0
    row_scalar: bool | None = None
    dtype: str | None = None


@dataclass(frozen=True, slots=True)
class GroupByOp:
    key_specs: tuple[GroupKeySpec, ...]
    static_groups: tuple[tuple[int, ...], ...] | None
    inner_program: "Program"
    capacity: int | None = None
    hash_capacity: int | None = None

    @property
    def n_dynamic_keys(self) -> int:
        return len(self.key_specs)


OpSpec: TypeAlias = (
    InputOp
    | LiteralOp
    | NaryOp
    | CustomCallOp
    | CatOp
    | CumsumOp
    | ReductionOp
    | EmitOp
    | FFillOp
    | ShiftOp
    | EwmOp
    | XsRankOp
    | RbfBasisOp
    | FutureRbfBasisSumOp
    | EinsumOp
    | InstrumentBasisMeanOp
    | InstrumentBasisProjectionOp
    | RidgeOp
    | RidgeProjectionOp
    | GroupByOp
)


__all__ = [
    "InputOp",
    "LiteralOp",
    "NaryOp",
    "CustomCallOp",
    "CatOp",
    "CumsumOp",
    "ReductionOp",
    "EmitOp",
    "FFillOp",
    "ShiftOp",
    "EwmOp",
    "XsRankOp",
    "RbfBasisOp",
    "FutureRbfBasisSumOp",
    "EinsumOp",
    "InstrumentBasisMeanOp",
    "InstrumentBasisProjectionOp",
    "RidgeOp",
    "RidgeProjectionOp",
    "GroupKeySpec",
    "GroupByOp",
    "OpSpec",
]
