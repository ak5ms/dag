from __future__ import annotations

from dataclasses import dataclass
import math
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
        if self.kind not in {"sum", "mean", "std", "min", "max"}:
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
class XsPctRankOp:
    pass


@dataclass(frozen=True, slots=True)
class XsAggregateOp:
    kind: str
    quantile: float = 0.5

    def __post_init__(self) -> None:
        if self.kind not in {
            "count", "sum", "mean", "std", "min", "max", "quantile"
        }:
            raise ValueError(f"unsupported cross-sectional aggregate {self.kind!r}")
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError("cross-sectional quantile must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class XsWeightedMeanOp:
    pass


@dataclass(frozen=True, slots=True)
class XsProjectionOp:
    intercept: bool = False


@dataclass(frozen=True, slots=True)
class XsGeneralizedRankOp:
    power: float = 1.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.power) or self.power < 0.0:
            raise ValueError("generalized-rank power must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class XsDensifyOp:
    pass


@dataclass(frozen=True, slots=True)
class VectorQuantileOp:
    quantile: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError("vector quantile must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class ColumnOp:
    index: int

    def __post_init__(self) -> None:
        if self.index < 0:
            raise ValueError("column index must be >= 0")


@dataclass(frozen=True, slots=True)
class RollingOp:
    """Fixed-period rolling statistic over observations/rows."""

    kind: str
    periods: int
    min_periods: int
    ddof: int = 0
    quantile: float = 0.5

    def __post_init__(self) -> None:
        supported = {
            "sum",
            "mean",
            "std",
            "min",
            "max",
            "median",
            "quantile",
            "pct_rank",
            "argmin",
            "argmax",
        }
        if self.kind not in supported:
            raise ValueError(f"unsupported rolling statistic {self.kind!r}")
        if self.periods < 1:
            raise ValueError("rolling periods must be >= 1")
        if not 0 <= self.min_periods <= self.periods:
            raise ValueError("rolling min_periods must be in [0, periods]")
        if self.ddof < 0:
            raise ValueError("rolling ddof must be >= 0")
        if not 0.0 <= self.quantile <= 1.0:
            raise ValueError("rolling quantile must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class TheilSenOp:
    periods: int
    min_periods: int

    def __post_init__(self) -> None:
        if self.periods < 2:
            raise ValueError("Theil-Sen periods must be >= 2")
        if not 2 <= self.min_periods <= self.periods:
            raise ValueError("Theil-Sen min_periods must be in [2, periods]")


@dataclass(frozen=True, slots=True)
class PeriodsSinceChangeOp:
    pass


@dataclass(frozen=True, slots=True)
class HumpOp:
    threshold: float
    relative: bool = False
    move_by_threshold: bool = False

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or self.threshold < 0.0:
            raise ValueError("hump threshold must be finite and >= 0")


@dataclass(frozen=True, slots=True)
class TradeWhenOp:
    pass


@dataclass(frozen=True, slots=True)
class LinearFilterOp:
    feedforward: tuple[float, ...]
    recursive: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if not self.feedforward:
            raise ValueError("filter requires at least one feed-forward weight")
        if not all(math.isfinite(value) for value in self.feedforward + self.recursive):
            raise ValueError("filter weights must be finite")


@dataclass(frozen=True, slots=True)
class RollingProductOp:
    periods: int
    min_periods: int

    def __post_init__(self) -> None:
        if self.periods < 1 or not 0 <= self.min_periods <= self.periods:
            raise ValueError("invalid rolling-product periods/min_periods")


@dataclass(frozen=True, slots=True)
class RollingKthOp:
    periods: int
    min_periods: int
    k: int = 1
    ignore_zero: bool = True

    def __post_init__(self) -> None:
        if self.periods < 1 or not 0 <= self.min_periods <= self.periods:
            raise ValueError("invalid rolling-kth periods/min_periods")
        if not 1 <= self.k <= self.periods:
            raise ValueError("rolling-kth k must be in [1, periods]")


@dataclass(frozen=True, slots=True)
class RollingPrevDiffOp:
    periods: int

    def __post_init__(self) -> None:
        if self.periods < 2:
            raise ValueError("rolling previous-different periods must be >= 2")


@dataclass(frozen=True, slots=True)
class RollingDecayOp:
    periods: int
    min_periods: int

    def __post_init__(self) -> None:
        if self.periods < 1 or not 0 <= self.min_periods <= self.periods:
            raise ValueError("invalid rolling-decay periods/min_periods")


@dataclass(frozen=True, slots=True)
class RollingEntropyOp:
    periods: int
    min_periods: int
    buckets: int = 10

    def __post_init__(self) -> None:
        if self.periods < 1 or not 0 <= self.min_periods <= self.periods:
            raise ValueError("invalid rolling-entropy periods/min_periods")
        if self.buckets < 1:
            raise ValueError("rolling-entropy buckets must be >= 1")


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
    recompute_every: int = 1

    def __post_init__(self) -> None:
        if self.recompute_every < 1:
            raise ValueError("Ridge recompute_every must be >= 1")

    @property
    def coefficient_width(self) -> int:
        return sum(self.feature_widths)


@dataclass(frozen=True, slots=True)
class RidgeProjectionOp:
    field: str
    component: int | None = None

    def __post_init__(self) -> None:
        supported = {
            "beta",
            "preds",
            "residuals",
            "coefficient",
            "standard_errors",
            "standard_error",
            "tstats",
            "tstat",
            "sse",
            "sst",
            "r2",
            "residual_variance",
            "effective_df",
            "effective_n",
        }
        if self.field not in supported:
            raise ValueError(f"unsupported Ridge projection {self.field!r}")
        component_fields = {"coefficient", "standard_error", "tstat"}
        if self.field in component_fields:
            if self.component is None or self.component < 0:
                raise ValueError(f"Ridge {self.field} requires a nonnegative component")
        elif self.component is not None:
            raise ValueError(f"Ridge {self.field} does not accept a component")


@dataclass(frozen=True, slots=True)
class GroupKeySpec:
    num_keys: int | None = None
    offset: int = 0
    row_scalar: bool | None = None
    dtype: str | None = None
    monotonic: bool = False


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
    | XsPctRankOp
    | XsAggregateOp
    | XsWeightedMeanOp
    | XsProjectionOp
    | XsGeneralizedRankOp
    | XsDensifyOp
    | VectorQuantileOp
    | ColumnOp
    | RollingOp
    | TheilSenOp
    | PeriodsSinceChangeOp
    | HumpOp
    | TradeWhenOp
    | LinearFilterOp
    | RollingProductOp
    | RollingKthOp
    | RollingPrevDiffOp
    | RollingDecayOp
    | RollingEntropyOp
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
    "XsPctRankOp",
    "XsAggregateOp",
    "XsWeightedMeanOp",
    "XsProjectionOp",
    "XsGeneralizedRankOp",
    "XsDensifyOp",
    "VectorQuantileOp",
    "ColumnOp",
    "RollingOp",
    "TheilSenOp",
    "PeriodsSinceChangeOp",
    "HumpOp",
    "TradeWhenOp",
    "LinearFilterOp",
    "RollingProductOp",
    "RollingKthOp",
    "RollingPrevDiffOp",
    "RollingDecayOp",
    "RollingEntropyOp",
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
