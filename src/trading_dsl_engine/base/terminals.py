from __future__ import annotations

from collections.abc import Iterable, Mapping


def futures_field_metadata(levels: Iterable[int] = range(10)) -> dict[str, dict[str, object]]:
    fields: dict[str, dict[str, object]] = {
        "_ev_ts": _field(("event_timestamp", "calendar_time", "integer"), ">=0"),
        "volume_out0": _field(("contract_quantity", "trade_volume", "front_month_contract"), ">=0"),
        "volume_out1": _field(("contract_quantity", "trade_volume", "following_month_contract"), ">=0"),
        "is_tradable_out0": _field(("dimensionless", "boolean_0_1", "front_month_contract"), "boolean"),
        "is_tradable_out1": _field(("dimensionless", "boolean_0_1", "following_month_contract"), "boolean"),
        "wdte_out0": _field(("calendar_time", "day_count", "weekdays_to_expiry", "front_month_contract", "integer"), ">=0"),
        "wdte_out1": _field(("calendar_time", "day_count", "weekdays_to_expiry", "following_month_contract", "integer"), ">=0"),
    }
    for suffix, contract in (("0", "front_month_contract"), ("1", "following_month_contract")):
        fields[f"vwap_out{suffix}"] = _field(("price", "trade_vwap", contract), ">0")
        fields[f"vwap_mp_out{suffix}"] = _field(("price", "mid_price_vwap", contract), ">0")
        fields[f"vw_halfspread_out{suffix}"] = _field(("dimensionless", "spread_fraction", "volume_weighted", contract), (0.0, 1.0))
        fields[f"trade_cross_pct_out{suffix}.count"] = _field(("count", "dimensionless", contract, "integer"), ">=0")
        for agg in ("first", "last", "max", "min", "sum"):
            fields[f"trade_cross_pct_out{suffix}.{agg}"] = _field(("contract_quantity_weighted_dimensionless", "trade_cross_pct", contract), "real")
        for side, side_tag in (("a", "ask"), ("b", "bid")):
            for level in levels:
                fields[f"{side}p{level}_out{suffix}"] = _field(("price", side_tag, f"level_{level}", contract), ">0")
                fields[f"volume_{side}{level}_out{suffix}"] = _field(("contract_quantity", side_tag, f"level_{level}", contract), ">=0")
        for prefix, side_tag in (("ap", "ask"), ("bp", "bid"), ("mp", "mid")):
            for part in ("open", "high", "low", "close"):
                fields[f"{prefix}_out{suffix}.{part}"] = _field(("price", side_tag, "level_0", "ohlc_bar", contract), ">0")
        calendar = f"calendar_{suffix}"
        for name in ("session_start", "session_end", "next_session_start", "next_session_end"):
            fields[f"{name}{suffix}"] = _field((calendar, name, "calendar_time", contract, "integer"), ">=0")
    return fields


def feature_names_with_tags(
    fields: Mapping[str, Mapping[str, object]],
    *,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
) -> tuple[str, ...]:
    include_set = set(include)
    exclude_set = set(exclude)
    out = []
    for name, spec in fields.items():
        tags = set(spec.get("types", ()))
        if include_set.issubset(tags) and not tags.intersection(exclude_set):
            out.append(name)
    return tuple(out)


def futures_type_relations(levels: Iterable[int] = range(10)) -> tuple[tuple[str, str], ...]:
    edges = [
        ("ask", "price"),
        ("bid", "price"),
        ("mid", "price"),
        ("trade_vwap", "price"),
        ("mid_price_vwap", "price"),
        ("spread_fraction", "dimensionless"),
        ("boolean_0_1", "dimensionless"),
        ("trade_cross_pct", "contract_quantity_weighted_dimensionless"),
        ("day_count", "calendar_time"),
        ("event_timestamp", "calendar_time"),
        ("session_start", "calendar_time"),
        ("session_end", "calendar_time"),
        ("next_session_start", "calendar_time"),
        ("next_session_end", "calendar_time"),
    ]
    edges.extend((f"level_{level}", "book_level") for level in levels)
    return tuple(edges)


def _field(types: Iterable[str], value_range: str | tuple[float, float]) -> dict[str, object]:
    type_tuple = tuple(types)
    out: dict[str, object] = {"types": type_tuple, "range": value_range}
    unit = _unit_for_types(type_tuple)
    if unit is not None:
        out["units"] = unit
    return out


def _unit_for_types(types: tuple[str, ...]) -> str | None:
    if "price" in types:
        return "price"
    if "calendar_time" in types:
        return "calendar_time"
    if "contract_quantity" in types or "contract_quantity_weighted_dimensionless" in types:
        return "contract_quantity"
    if "count" in types:
        return "count"
    return None


__all__ = ["feature_names_with_tags", "futures_field_metadata", "futures_type_relations"]
