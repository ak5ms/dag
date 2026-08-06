from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterable, Literal, Mapping


SearchShape = Literal["scalar", "row", "matrix", "boolean", "object", "unknown"]
ValueMode = Literal["runtime_row", "runtime_scalar", "compile_time_literal", "object"]


_GENERIC_TYPES = frozenset({
    "numeric", "finite", "bounded", "signed", "nonnegative", "positive",
    "scalar", "row", "matrix", "boolean", "unknown", "parameter",
})


@dataclass(frozen=True)
class SemanticInfo:
    types: frozenset[str]
    shape: SearchShape = "row"
    lower: float = -math.inf
    upper: float = math.inf
    integer: bool = False
    mode: ValueMode = "runtime_row"
    role: str = "value"
    literal_value: float | None = None

    @property
    def is_boolean(self) -> bool:
        return "boolean" in self.types or self.shape == "boolean"

    @property
    def is_dimensionless(self) -> bool:
        return "dimensionless" in self.types

    @property
    def is_static_literal(self) -> bool:
        return self.mode == "compile_time_literal" and self.literal_value is not None

    def with_types(self, types: Iterable[str], **changes) -> "SemanticInfo":
        return replace(self, types=frozenset(types), **changes)


@dataclass(frozen=True)
class TypeRelations:
    edges: frozenset[tuple[str, str]]

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[str, str]] = ()) -> "TypeRelations":
        return cls(frozenset((str(source), str(target)) for source, target in edges))

    def closure(self, types: Iterable[str]) -> frozenset[str]:
        seen = set(types)
        changed = True
        while changed:
            changed = False
            for source, target in self.edges:
                if source in seen and target not in seen:
                    seen.add(target)
                    changed = True
        return frozenset(seen)

    def useful(self, types: Iterable[str]) -> frozenset[str]:
        return self.closure(types) - _GENERIC_TYPES

    def common(self, left: SemanticInfo, right: SemanticInfo) -> frozenset[str]:
        return self.useful(left.types) & self.useful(right.types)


DEFAULT_TYPE_RELATIONS = TypeRelations.from_edges((
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
))


def default_market_semantics() -> dict[str, SemanticInfo]:
    positive_price = dict(shape="row", lower=math.nextafter(0.0, 1.0), upper=math.inf)
    nonnegative_quantity = dict(shape="row", lower=0.0, upper=math.inf)
    return {
        "ap0": SemanticInfo(
            frozenset({"numeric", "price", "quote_price", "ask_price", "ask", "best_quote", "level_0"}),
            **positive_price,
        ),
        "bp0": SemanticInfo(
            frozenset({"numeric", "price", "quote_price", "bid_price", "bid", "best_quote", "level_0"}),
            **positive_price,
        ),
        "av0": SemanticInfo(
            frozenset({"numeric", "quantity", "contract_quantity", "quoted_size", "liquidity", "ask", "best_quote", "level_0", "nonnegative"}),
            **nonnegative_quantity,
        ),
        "bv0": SemanticInfo(
            frozenset({"numeric", "quantity", "contract_quantity", "quoted_size", "liquidity", "bid", "best_quote", "level_0", "nonnegative"}),
            **nonnegative_quantity,
        ),
        "volume": SemanticInfo(
            frozenset({"numeric", "quantity", "contract_quantity", "trade_volume", "activity", "nonnegative"}),
            **nonnegative_quantity,
        ),
        "vwap": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "trade_vwap"}),
            **positive_price,
        ),
        "open": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "ohlc", "open_price"}),
            **positive_price,
        ),
        "high": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "ohlc", "high_price"}),
            **positive_price,
        ),
        "low": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "ohlc", "low_price"}),
            **positive_price,
        ),
        "close": SemanticInfo(
            frozenset({"numeric", "price", "trade_price", "ohlc", "close_price"}),
            **positive_price,
        ),
        "soft_side_wavg": SemanticInfo(
            frozenset({"numeric", "dimensionless", "signed_trade_side", "order_flow", "volume_weighted", "bounded"}),
            shape="row",
            lower=-1.0,
            upper=1.0,
        ),
    }


def literal_semantics(value: float) -> SemanticInfo:
    numeric = float(value)
    return SemanticInfo(
        frozenset({"numeric", "dimensionless", "parameter", "scalar", "finite"}),
        shape="scalar",
        lower=numeric,
        upper=numeric,
        integer=numeric.is_integer(),
        mode="compile_time_literal",
        role="parameter",
        literal_value=numeric,
    )


def broadcast_shape(*values: SemanticInfo) -> SearchShape | None:
    shapes = {value.shape for value in values if value.shape not in {"unknown", "scalar"}}
    if not shapes:
        return "scalar"
    if shapes <= {"row", "boolean"}:
        return "row"
    if len(shapes) == 1:
        return next(iter(shapes))
    return None


def compatible_additive(
    left: SemanticInfo,
    right: SemanticInfo,
    relations: TypeRelations = DEFAULT_TYPE_RELATIONS,
) -> SemanticInfo | None:
    shape = broadcast_shape(left, right)
    common = relations.common(left, right)
    if shape is None or not common:
        return None
    return SemanticInfo(
        frozenset({"numeric"}) | common,
        shape=shape,
        lower=-math.inf,
        upper=math.inf,
    )


def numeric_product(
    left: SemanticInfo,
    right: SemanticInfo,
    relations: TypeRelations = DEFAULT_TYPE_RELATIONS,
) -> SemanticInfo | None:
    shape = broadcast_shape(left, right)
    if shape is None or "numeric" not in left.types or "numeric" not in right.types:
        return None
    left_types = relations.closure(left.types)
    right_types = relations.closure(right.types)
    if "dimensionless" in left_types and "dimensionless" not in right_types:
        output = right_types
    elif "dimensionless" in right_types and "dimensionless" not in left_types:
        output = left_types
    elif "dimensionless" in left_types and "dimensionless" in right_types:
        output = frozenset({"numeric", "dimensionless"})
    else:
        left_label = sorted(relations.useful(left.types))[:1] or ["numeric"]
        right_label = sorted(relations.useful(right.types))[:1] or ["numeric"]
        output = frozenset({"numeric", f"product:{left_label[0]}:{right_label[0]}"})
    return SemanticInfo(output | frozenset({"numeric"}), shape=shape)


def numeric_ratio(
    numerator: SemanticInfo,
    denominator: SemanticInfo,
    relations: TypeRelations = DEFAULT_TYPE_RELATIONS,
) -> SemanticInfo | None:
    shape = broadcast_shape(numerator, denominator)
    if shape is None or "numeric" not in numerator.types or "numeric" not in denominator.types:
        return None
    common = relations.common(numerator, denominator)
    denominator_types = relations.closure(denominator.types)
    if common:
        output = frozenset({"numeric", "dimensionless", "ratio"})
    elif "dimensionless" in denominator_types:
        output = relations.closure(numerator.types) | frozenset({"numeric"})
    else:
        left_label = sorted(relations.useful(numerator.types))[:1] or ["numeric"]
        right_label = sorted(relations.useful(denominator.types))[:1] or ["numeric"]
        output = frozenset({"numeric", f"ratio:{left_label[0]}:{right_label[0]}"})
    return SemanticInfo(output, shape=shape)


def comparison_result(
    left: SemanticInfo,
    right: SemanticInfo,
    relations: TypeRelations = DEFAULT_TYPE_RELATIONS,
) -> SemanticInfo | None:
    shape = broadcast_shape(left, right)
    if shape is None or not relations.common(left, right):
        return None
    return SemanticInfo(
        frozenset({"numeric", "boolean", "dimensionless"}),
        shape="boolean" if shape == "scalar" else "row",
        lower=0.0,
        upper=1.0,
        integer=True,
    )


def branch_result(
    condition: SemanticInfo,
    true_value: SemanticInfo,
    false_value: SemanticInfo,
    relations: TypeRelations = DEFAULT_TYPE_RELATIONS,
) -> SemanticInfo | None:
    if not condition.is_boolean:
        return None
    return compatible_additive(true_value, false_value, relations)


def dimensionless_result(value: SemanticInfo) -> SemanticInfo | None:
    if "numeric" not in value.types:
        return None
    return SemanticInfo(
        frozenset({"numeric", "dimensionless"}),
        shape="row" if value.shape in {"row", "boolean"} else value.shape,
    )


def target_satisfied(
    value: SemanticInfo,
    target_types: Iterable[str],
    relations: TypeRelations = DEFAULT_TYPE_RELATIONS,
) -> bool:
    required = frozenset(target_types)
    return not required or bool(relations.closure(value.types) & required)


def metadata_to_semantics(
    fields: Mapping[str, Mapping[str, object]],
) -> dict[str, SemanticInfo]:
    out: dict[str, SemanticInfo] = {}
    for name, spec in fields.items():
        types = frozenset(str(value) for value in spec.get("types", ())) | frozenset({"numeric"})
        bounds = spec.get("range", "real")
        if isinstance(bounds, tuple):
            lower, upper = map(float, bounds)
        elif bounds == ">0":
            lower, upper = math.nextafter(0.0, 1.0), math.inf
        elif bounds == ">=0":
            lower, upper = 0.0, math.inf
        elif bounds == "boolean":
            lower, upper = 0.0, 1.0
            types |= frozenset({"boolean", "dimensionless"})
        else:
            lower, upper = -math.inf, math.inf
        out[name] = SemanticInfo(
            types,
            shape="row",
            lower=lower,
            upper=upper,
            integer="integer" in types,
        )
    return out


__all__ = [
    "DEFAULT_TYPE_RELATIONS",
    "SearchShape",
    "SemanticInfo",
    "TypeRelations",
    "branch_result",
    "broadcast_shape",
    "comparison_result",
    "compatible_additive",
    "default_market_semantics",
    "dimensionless_result",
    "literal_semantics",
    "metadata_to_semantics",
    "numeric_product",
    "numeric_ratio",
    "target_satisfied",
]
