from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from collections.abc import Iterable, Mapping


class SearchShape(str, Enum):
    SCALAR = "scalar"
    ROW = "row"
    BOOLEAN_ROW = "boolean_row"
    MATRIX = "matrix"
    OBJECT = "object"


GENERIC_TYPES = frozenset(
    {
        "numeric",
        "finite",
        "bounded",
        "nonnegative",
        "positive",
        "integer",
        "parameter",
        "compile_time",
        "runtime",
    }
)

# These tags describe provenance, book side, aggregation, or field location. They
# are useful for operator constraints and priors, but they are not value domains
# that make addition/min/max meaningful. For example, ``bv0`` and ``ap0`` are
# both level-0 best quotes, but one is a quantity and the other is a price.
DESCRIPTOR_TYPES = frozenset(
    {
        "activity",
        "ask",
        "best_quote",
        "bid",
        "level_0",
        "liquidity",
        "ohlc",
        "quote_side",
        "volume_weighted",
    }
)

NON_VALUE_TYPES = GENERIC_TYPES | DESCRIPTOR_TYPES


@dataclass(frozen=True)
class SemanticInfo:
    types: frozenset[str]
    shape: SearchShape
    lower: float = -math.inf
    upper: float = math.inf
    integer: bool = False
    static: bool = False
    role: str = "value"

    @property
    def positive(self) -> bool:
        return self.lower > 0.0

    @property
    def nonnegative(self) -> bool:
        return self.lower >= 0.0

    @property
    def boolean(self) -> bool:
        return "boolean" in self.types or self.shape is SearchShape.BOOLEAN_ROW

    def with_types(self, types: Iterable[str], **updates) -> "SemanticInfo":
        return replace(self, types=frozenset(types), **updates)


@dataclass(frozen=True)
class TypeGraph:
    edges: frozenset[tuple[str, str]]

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[str, str]]) -> "TypeGraph":
        return cls(frozenset((str(a), str(b)) for a, b in edges))

    def closure(self, types: Iterable[str]) -> frozenset[str]:
        seen = set(types)
        changed = True
        while changed:
            changed = False
            for child, parent in self.edges:
                if child in seen and parent not in seen:
                    seen.add(parent)
                    changed = True
        return frozenset(seen)

    def meaningful_common(
        self, left: SemanticInfo, right: SemanticInfo
    ) -> frozenset[str]:
        return (
            self.closure(left.types)
            & self.closure(right.types)
        ) - NON_VALUE_TYPES


DEFAULT_TYPE_GRAPH = TypeGraph.from_edges(
    (
        ("ask_price", "quote_price"),
        ("bid_price", "quote_price"),
        ("quote_price", "price"),
        ("trade_vwap", "trade_price"),
        ("open_price", "trade_price"),
        ("high_price", "trade_price"),
        ("low_price", "trade_price"),
        ("close_price", "trade_price"),
        ("trade_price", "price"),
        ("ask", "quote_side"),
        ("bid", "quote_side"),
        ("quoted_size", "quantity"),
        ("contract_quantity", "quantity"),
        ("trade_volume", "quantity"),
        ("signed_trade_side", "dimensionless"),
        ("boolean", "dimensionless"),
        ("boolean_0_1", "boolean"),
    )
)


def broadcast_shape(left: SearchShape, right: SearchShape) -> SearchShape | None:
    if left is right:
        return left
    if left is SearchShape.SCALAR:
        return right
    if right is SearchShape.SCALAR:
        return left
    if {left, right} <= {SearchShape.ROW, SearchShape.BOOLEAN_ROW}:
        return SearchShape.ROW
    return None


def compatible(
    left: SemanticInfo,
    right: SemanticInfo,
    graph: TypeGraph = DEFAULT_TYPE_GRAPH,
) -> bool:
    return (
        broadcast_shape(left.shape, right.shape) is not None
        and bool(graph.meaningful_common(left, right))
    )


def common_output(
    left: SemanticInfo,
    right: SemanticInfo,
    graph: TypeGraph = DEFAULT_TYPE_GRAPH,
) -> SemanticInfo:
    shape = broadcast_shape(left.shape, right.shape)
    common = graph.meaningful_common(left, right)
    if shape is None or not common:
        raise ValueError("operands have no compatible semantic value type")
    return SemanticInfo(
        types=frozenset({"numeric"}) | common,
        shape=shape,
        lower=-math.inf,
        upper=math.inf,
    )


def unary_preserve(value: SemanticInfo) -> SemanticInfo:
    return value.with_types(value.types)


def dimensionless_output(
    value: SemanticInfo,
    *,
    lower: float = -math.inf,
    upper: float = math.inf,
) -> SemanticInfo:
    shape = (
        SearchShape.ROW
        if value.shape is SearchShape.BOOLEAN_ROW
        else value.shape
    )
    return SemanticInfo(
        frozenset({"numeric", "dimensionless"}),
        shape,
        lower,
        upper,
    )


def boolean_output(left: SemanticInfo, right: SemanticInfo) -> SemanticInfo:
    shape = broadcast_shape(left.shape, right.shape)
    if shape is None:
        raise ValueError("comparison operands cannot broadcast")
    return SemanticInfo(
        frozenset({"numeric", "boolean", "dimensionless"}),
        SearchShape.BOOLEAN_ROW if shape is not SearchShape.SCALAR else SearchShape.SCALAR,
        0.0,
        1.0,
        integer=True,
    )


def multiplication_output(
    left: SemanticInfo,
    right: SemanticInfo,
    graph: TypeGraph = DEFAULT_TYPE_GRAPH,
) -> SemanticInfo:
    shape = broadcast_shape(left.shape, right.shape)
    if shape is None:
        raise ValueError("multiplication operands cannot broadcast")
    ltypes = graph.closure(left.types)
    rtypes = graph.closure(right.types)
    if "dimensionless" in ltypes and "dimensionless" not in rtypes:
        out_types = right.types | frozenset({"numeric"})
    elif "dimensionless" in rtypes and "dimensionless" not in ltypes:
        out_types = left.types | frozenset({"numeric"})
    elif "dimensionless" in ltypes and "dimensionless" in rtypes:
        out_types = frozenset({"numeric", "dimensionless"})
    else:
        principal_left = sorted(ltypes - NON_VALUE_TYPES)
        principal_right = sorted(rtypes - NON_VALUE_TYPES)
        label = "product:" + "*".join(
            (principal_left[-1:] or ["unknown"])
            + (principal_right[-1:] or ["unknown"])
        )
        out_types = frozenset({"numeric", label})
    return SemanticInfo(out_types, shape)


def division_output(
    left: SemanticInfo,
    right: SemanticInfo,
    graph: TypeGraph = DEFAULT_TYPE_GRAPH,
) -> SemanticInfo:
    shape = broadcast_shape(left.shape, right.shape)
    if shape is None:
        raise ValueError("division operands cannot broadcast")
    ltypes = graph.closure(left.types)
    rtypes = graph.closure(right.types)
    common = (ltypes & rtypes) - NON_VALUE_TYPES
    if common:
        return SemanticInfo(
            frozenset({"numeric", "dimensionless", "ratio"}),
            shape,
        )
    if "dimensionless" in rtypes:
        return SemanticInfo(left.types | frozenset({"numeric"}), shape)
    principal_left = sorted(ltypes - NON_VALUE_TYPES)
    principal_right = sorted(rtypes - NON_VALUE_TYPES)
    label = "ratio:" + "/".join(
        (principal_left[-1:] or ["unknown"])
        + (principal_right[-1:] or ["unknown"])
    )
    return SemanticInfo(frozenset({"numeric", label}), shape)


def literal_semantics(value: float, *, role: str = "parameter") -> SemanticInfo:
    return SemanticInfo(
        frozenset(
            {
                "numeric",
                "dimensionless",
                "parameter",
                "compile_time",
            }
        ),
        SearchShape.SCALAR,
        float(value),
        float(value),
        integer=float(value).is_integer(),
        static=True,
        role=role,
    )


def alpha_terminal_metadata() -> dict[str, SemanticInfo]:
    row = SearchShape.ROW
    return {
        "ap0": SemanticInfo(
            frozenset(
                {
                    "numeric",
                    "price",
                    "quote_price",
                    "ask_price",
                    "ask",
                    "best_quote",
                    "level_0",
                }
            ),
            row,
            math.nextafter(0.0, 1.0),
        ),
        "bp0": SemanticInfo(
            frozenset(
                {
                    "numeric",
                    "price",
                    "quote_price",
                    "bid_price",
                    "bid",
                    "best_quote",
                    "level_0",
                }
            ),
            row,
            math.nextafter(0.0, 1.0),
        ),
        "av0": SemanticInfo(
            frozenset(
                {
                    "numeric",
                    "quantity",
                    "contract_quantity",
                    "quoted_size",
                    "liquidity",
                    "ask",
                    "best_quote",
                    "level_0",
                }
            ),
            row,
            0.0,
        ),
        "bv0": SemanticInfo(
            frozenset(
                {
                    "numeric",
                    "quantity",
                    "contract_quantity",
                    "quoted_size",
                    "liquidity",
                    "bid",
                    "best_quote",
                    "level_0",
                }
            ),
            row,
            0.0,
        ),
        "volume": SemanticInfo(
            frozenset(
                {
                    "numeric",
                    "quantity",
                    "contract_quantity",
                    "trade_volume",
                    "activity",
                }
            ),
            row,
            0.0,
        ),
        "vwap": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "trade_vwap"}),
            row,
            math.nextafter(0.0, 1.0),
        ),
        "open": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "ohlc", "open_price"}),
            row,
            math.nextafter(0.0, 1.0),
        ),
        "high": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "ohlc", "high_price"}),
            row,
            math.nextafter(0.0, 1.0),
        ),
        "low": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "ohlc", "low_price"}),
            row,
            math.nextafter(0.0, 1.0),
        ),
        "close": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "ohlc", "close_price"}),
            row,
            math.nextafter(0.0, 1.0),
        ),
        "soft_side_wavg": SemanticInfo(
            frozenset(
                {
                    "numeric",
                    "dimensionless",
                    "signed_trade_side",
                    "order_flow",
                    "volume_weighted",
                }
            ),
            row,
            -1.0,
            1.0,
        ),
    }


def target_type_satisfied(
    info: SemanticInfo,
    required_types: Iterable[str],
    graph: TypeGraph = DEFAULT_TYPE_GRAPH,
) -> bool:
    required = frozenset(required_types)
    return not required or bool(graph.closure(info.types) & required)


def metadata_as_dict(
    values: Mapping[str, SemanticInfo] | None = None,
) -> dict[str, SemanticInfo]:
    return dict(alpha_terminal_metadata() if values is None else values)


__all__ = [
    "DEFAULT_TYPE_GRAPH",
    "DESCRIPTOR_TYPES",
    "GENERIC_TYPES",
    "NON_VALUE_TYPES",
    "SearchShape",
    "SemanticInfo",
    "TypeGraph",
    "alpha_terminal_metadata",
    "boolean_output",
    "broadcast_shape",
    "common_output",
    "compatible",
    "dimensionless_output",
    "division_output",
    "literal_semantics",
    "metadata_as_dict",
    "multiplication_output",
    "target_type_satisfied",
    "unary_preserve",
]
