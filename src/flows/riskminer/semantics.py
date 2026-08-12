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

# Provenance/location tags are deliberately excluded from value compatibility.
# E.g. ap0_out0 and volume_a0_out0 are both ask/level-0 fields but must not
# become add/sub compatible merely because they share those descriptors.
DESCRIPTOR_TYPES = frozenset(
    {
        "activity",
        "ask",
        "best_quote",
        "bid",
        "book",
        "book_level",
        "close",
        "event",
        "first",
        "high",
        "level_0",
        "level_1",
        "level_2",
        "level_3",
        "level_4",
        "level_5",
        "level_6",
        "level_7",
        "level_8",
        "level_9",
        "last",
        "liquidity",
        "low",
        "max",
        "mid",
        "min",
        "next_session",
        "ohlc",
        "open",
        "quote_side",
        "sampled",
        "session",
        "sum",
        "trade",
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
        # Price relations.
        ("ask_level_price", "ask_price"),
        ("ask_ohlc_price", "ask_price"),
        ("ask_price", "quote_price"),
        ("bid_level_price", "bid_price"),
        ("bid_ohlc_price", "bid_price"),
        ("bid_price", "quote_price"),
        ("quote_price", "price"),
        ("mid_ohlc_price", "mid_price"),
        ("mid_vwap", "mid_price"),
        ("mid_price", "reference_price"),
        ("reference_price", "price"),
        ("trade_vwap", "trade_price"),
        ("trade_price", "price"),
        ("open_price", "trade_price"),
        ("high_price", "trade_price"),
        ("low_price", "trade_price"),
        ("close_price", "trade_price"),

        # Quantity/activity relations.
        ("book_volume", "quoted_size"),
        ("quoted_size", "quantity"),
        ("level_volume", "quantity"),
        ("contract_quantity", "quantity"),
        ("trade_volume", "contract_quantity"),
        ("cross_weighted_trade_quantity", "quantity"),

        # Dimensionless values.
        ("signed_trade_side", "dimensionless"),
        ("half_spread_fraction", "dimensionless"),
        ("trade_count", "count"),
        ("count", "dimensionless"),
        ("boolean", "dimensionless"),
        ("boolean_0_1", "boolean"),

        # Time relations. Microsecond timestamps can be subtracted into
        # duration_us, while wdte remains a separate trading-day horizon type.
        ("event_timestamp", "timestamp"),
        ("session_timestamp", "timestamp"),
        ("duration_us", "duration"),
        ("weekdays_to_expiry", "trading_day_horizon"),

        # Descriptor ancestry.
        ("ask", "quote_side"),
        ("bid", "quote_side"),
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


def subtraction_output(
    left: SemanticInfo,
    right: SemanticInfo,
    graph: TypeGraph = DEFAULT_TYPE_GRAPH,
) -> SemanticInfo:
    shape = broadcast_shape(left.shape, right.shape)
    if shape is None:
        raise ValueError("subtraction operands cannot broadcast")
    ltypes = graph.closure(left.types)
    rtypes = graph.closure(right.types)
    if "timestamp" in ltypes and "timestamp" in rtypes:
        return SemanticInfo(
            frozenset({"numeric", "duration", "duration_us"}),
            shape,
        )
    return common_output(left, right, graph)


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
    """Small backwards-compatible terminal set used by synthetic benchmarks."""

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


INPUTDATA_ALPHA_KEYS = (
    "_ev_ts",
    "ap0_out0",
    "ap1_out0",
    "ap2_out0",
    "ap3_out0",
    "ap4_out0",
    "ap5_out0",
    "ap6_out0",
    "ap7_out0",
    "ap8_out0",
    "ap9_out0",
    "ap_out0.close",
    "ap_out0.high",
    "ap_out0.low",
    "ap_out0.open",
    "bp0_out0",
    "bp1_out0",
    "bp2_out0",
    "bp3_out0",
    "bp4_out0",
    "bp5_out0",
    "bp6_out0",
    "bp7_out0",
    "bp8_out0",
    "bp9_out0",
    "bp_out0.close",
    "bp_out0.high",
    "bp_out0.low",
    "bp_out0.open",
    "is_tradable_out0",
    "mp_out0.close",
    "mp_out0.high",
    "mp_out0.low",
    "mp_out0.open",
    "next_session_end0",
    "next_session_start0",
    "session_end0",
    "session_start0",
    "trade_cross_pct_out0.count",
    "trade_cross_pct_out0.first",
    "trade_cross_pct_out0.last",
    "trade_cross_pct_out0.max",
    "trade_cross_pct_out0.min",
    "trade_cross_pct_out0.sum",
    "volume_a0_out0",
    "volume_a1_out0",
    "volume_a2_out0",
    "volume_a3_out0",
    "volume_a4_out0",
    "volume_a5_out0",
    "volume_a6_out0",
    "volume_a7_out0",
    "volume_a8_out0",
    "volume_a9_out0",
    "volume_b0_out0",
    "volume_b1_out0",
    "volume_b2_out0",
    "volume_b3_out0",
    "volume_b4_out0",
    "volume_b5_out0",
    "volume_b6_out0",
    "volume_b7_out0",
    "volume_b8_out0",
    "volume_b9_out0",
    "volume_out0",
    "vw_halfspread_out0",
    "vwap_mp_out0",
    "vwap_out0",
    "wdte_out0",
)


def _price_info(*types: str, descriptors: Iterable[str] = ()) -> SemanticInfo:
    return SemanticInfo(
        frozenset({"numeric", "price", *types, *descriptors}),
        SearchShape.ROW,
        math.nextafter(0.0, 1.0),
    )


def _quantity_info(
    *types: str,
    descriptors: Iterable[str] = (),
    nonnegative: bool = True,
) -> SemanticInfo:
    return SemanticInfo(
        frozenset({"numeric", "quantity", *types, *descriptors}),
        SearchShape.ROW,
        0.0 if nonnegative else -math.inf,
    )


def _level_descriptors(side: str, level: int) -> tuple[str, ...]:
    return (side, "book", "book_level", f"level_{level}", "sampled")


def inputdata_alpha_terminal_metadata() -> dict[str, SemanticInfo]:
    """Semantic metadata for every user-approved InputData alpha field.

    The assignments follow dg_v1.py/qvalues.py:
    - ap*/bp* are sampled book prices.
    - volume_a*/volume_b* are sampled/reset VolumeAtPx level quantities.
    - ap_out0/bp_out0/mp_out0 are quote/mid OHLC candles.
    - volume_out0 is reset TradeQty sum.
    - vwap_out0 is trade-price VWAP; vwap_mp_out0 is trade-clock mid VWAP.
    - vw_halfspread_out0 is volume-weighted (ask-bid)/(ask+bid).
    - trade_cross_pct is TradeQty times a dimensionless price-in-spread location,
      so first/last/min/max/sum are quantity-like; count is count-like.
    - session fields and _ev_ts are microsecond timestamps.
    - wdte_out0 is a weekdays-to-expiry horizon, deliberately not equated with
      microsecond duration.
    """

    row = SearchShape.ROW
    values: dict[str, SemanticInfo] = {}

    for side, prefix, leaf_type in (
        ("ask", "ap", "ask_level_price"),
        ("bid", "bp", "bid_level_price"),
    ):
        for level in range(10):
            name = f"{prefix}{level}_out0"
            values[name] = _price_info(
                leaf_type,
                f"{side}_price",
                "quote_price",
                descriptors=_level_descriptors(side, level),
            )

    for side, prefix, leaf_type in (
        ("ask", "ap", "ask_ohlc_price"),
        ("bid", "bp", "bid_ohlc_price"),
        ("mid", "mp", "mid_ohlc_price"),
    ):
        for candle_part in ("open", "high", "low", "close"):
            values[f"{prefix}_out0.{candle_part}"] = _price_info(
                leaf_type,
                (
                    f"{side}_price"
                    if side in {"ask", "bid"}
                    else "mid_price"
                ),
                "quote_price" if side in {"ask", "bid"} else "reference_price",
                descriptors=(side, "ohlc", candle_part, "sampled"),
            )

    for side, prefix in (("ask", "a"), ("bid", "b")):
        for level in range(10):
            values[f"volume_{prefix}{level}_out0"] = _quantity_info(
                "level_volume",
                descriptors=(
                    "liquidity",
                    *_level_descriptors(side, level),
                ),
            )

    values["volume_out0"] = _quantity_info(
        "trade_volume",
        "contract_quantity",
        descriptors=("trade", "activity", "sampled"),
    )
    values["vwap_out0"] = _price_info(
        "trade_vwap",
        "trade_price",
        descriptors=("trade", "volume_weighted", "sampled"),
    )
    values["vwap_mp_out0"] = _price_info(
        "mid_vwap",
        "mid_price",
        "reference_price",
        descriptors=("mid", "volume_weighted", "sampled"),
    )
    values["vw_halfspread_out0"] = SemanticInfo(
        frozenset(
            {
                "numeric",
                "dimensionless",
                "half_spread_fraction",
                "volume_weighted",
                "sampled",
            }
        ),
        row,
    )

    values["trade_cross_pct_out0.count"] = SemanticInfo(
        frozenset({"numeric", "dimensionless", "count", "trade_count", "trade", "sampled"}),
        row,
        0.0,
        integer=True,
    )
    for aggregate in ("first", "last", "max", "min", "sum"):
        values[f"trade_cross_pct_out0.{aggregate}"] = _quantity_info(
            "cross_weighted_trade_quantity",
            descriptors=("trade", "sampled", aggregate),
            nonnegative=False,
        )

    values["is_tradable_out0"] = SemanticInfo(
        frozenset({"numeric", "boolean", "boolean_0_1"}),
        SearchShape.BOOLEAN_ROW,
        0.0,
        1.0,
        integer=True,
    )
    values["_ev_ts"] = SemanticInfo(
        frozenset({"numeric", "timestamp", "event_timestamp", "event"}),
        row,
        0.0,
    )
    for name in (
        "session_start0",
        "session_end0",
        "next_session_start0",
        "next_session_end0",
    ):
        values[name] = SemanticInfo(
            frozenset(
                {
                    "numeric",
                    "timestamp",
                    "session_timestamp",
                    "session",
                    *(
                        ("next_session",)
                        if name.startswith("next_")
                        else ()
                    ),
                }
            ),
            row,
            0.0,
        )
    values["wdte_out0"] = SemanticInfo(
        frozenset(
            {
                "numeric",
                "weekdays_to_expiry",
                "trading_day_horizon",
            }
        ),
        row,
        0.0,
    )

    missing = set(INPUTDATA_ALPHA_KEYS) - set(values)
    extras = set(values) - set(INPUTDATA_ALPHA_KEYS)
    if missing or extras:
        raise AssertionError(
            f"InputData alpha metadata mismatch: missing={sorted(missing)}, extras={sorted(extras)}"
        )
    return {name: values[name] for name in INPUTDATA_ALPHA_KEYS}


def inputdata_alpha_keys() -> tuple[str, ...]:
    return INPUTDATA_ALPHA_KEYS


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
    "INPUTDATA_ALPHA_KEYS",
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
    "inputdata_alpha_keys",
    "inputdata_alpha_terminal_metadata",
    "literal_semantics",
    "metadata_as_dict",
    "multiplication_output",
    "subtraction_output",
    "target_type_satisfied",
    "unary_preserve",
]
