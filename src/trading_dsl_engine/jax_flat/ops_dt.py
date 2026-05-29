from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat.ops import Op

jax.config.update("jax_enable_x64", True)


@dataclass(frozen=True)
class ToDtOp(Op):
    unit: str
    output_kind: str = "datetime"
    output_width: int | None = 1

    def tick(self, state, *child_values: jax.Array):
        del state
        return None, child_values[0]

    def scan_batch(self, state, *child_sequences: jax.Array):
        del state
        return None, child_sequences[0]


_UNIT_ALIASES = {
    "ns": 1,
    "nanosecond": 1,
    "nanoseconds": 1,
    "us": 1_000,
    "microsecond": 1_000,
    "microseconds": 1_000,
    "ms": 1_000_000,
    "millisecond": 1_000_000,
    "milliseconds": 1_000_000,
    "s": 1_000_000_000,
    "sec": 1_000_000_000,
    "second": 1_000_000_000,
    "seconds": 1_000_000_000,
    "T": 60 * 1_000_000_000,
    "min": 60 * 1_000_000_000,
    "minute": 60 * 1_000_000_000,
    "minutes": 60 * 1_000_000_000,
    "H": 60 * 60 * 1_000_000_000,
    "h": 60 * 60 * 1_000_000_000,
    "hour": 60 * 60 * 1_000_000_000,
    "hours": 60 * 60 * 1_000_000_000,
    "D": 24 * 60 * 60 * 1_000_000_000,
    "d": 24 * 60 * 60 * 1_000_000_000,
    "day": 24 * 60 * 60 * 1_000_000_000,
    "days": 24 * 60 * 60 * 1_000_000_000,
}


def unit_ns(unit: str) -> int:
    try:
        return _UNIT_ALIASES[unit]
    except KeyError as exc:
        raise ValueError(f"Unsupported datetime unit {unit!r}") from exc


def duration_ns(value: str | int | float, default_unit: str = "ns") -> int:
    if isinstance(value, (int, float)):
        return int(value) * unit_ns(default_unit)
    text = value.strip()
    if not text:
        raise ValueError("Duration cannot be empty")
    idx = 0
    while idx < len(text) and (text[idx].isdigit() or text[idx] in "+-."):
        idx += 1
    number = float(text[:idx]) if idx else 1.0
    unit = text[idx:] or default_unit
    return int(round(number * unit_ns(unit)))


def _valid_datetime_result(x, value):
    return jnp.where(jnp.isfinite(jnp.asarray(x)), value.astype(jnp.float64), jnp.nan)


def _safe_datetime_int(x):
    x = jnp.asarray(x)
    return jnp.where(jnp.isfinite(x), x, 0).astype(jnp.int64)


def _days_from_timestamp(x, unit: str):
    units_per_day = unit_ns("D") // unit_ns(unit)
    return jnp.floor_divide(_safe_datetime_int(x), units_per_day)


def _civil_from_days(days):
    z = jnp.asarray(days, dtype=jnp.int64) + 719468
    era = jnp.floor_divide(jnp.where(z >= 0, z, z - 146096), 146097)
    doe = z - era * 146097
    yoe = jnp.floor_divide(
        doe - jnp.floor_divide(doe, 1460) + jnp.floor_divide(doe, 36524) - jnp.floor_divide(doe, 146096),
        365,
    )
    y = yoe + era * 400
    doy = doe - (365 * yoe + jnp.floor_divide(yoe, 4) - jnp.floor_divide(yoe, 100))
    mp = jnp.floor_divide(5 * doy + 2, 153)
    d = doy - jnp.floor_divide(153 * mp + 2, 5) + 1
    m = mp + jnp.where(mp < 10, 3, -9)
    y = y + (m <= 2)
    return y, m, d


def _is_leap_year(year):
    return ((year % 4) == 0) & (((year % 100) != 0) | ((year % 400) == 0))


def dayofyear_value(x, unit: str):
    days = _days_from_timestamp(x, unit)
    year, month, day = _civil_from_days(days)
    before = jnp.array([0, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334], dtype=jnp.int64)
    leap_adjust = (month > 2) & _is_leap_year(year)
    return _valid_datetime_result(x, before[month] + day + leap_adjust.astype(jnp.int64))


def date_part_value(x, unit: str, part: str):
    days = _days_from_timestamp(x, unit)
    year, month, day = _civil_from_days(days)
    if part == "year":
        return _valid_datetime_result(x, year)
    if part == "month":
        return _valid_datetime_result(x, month)
    if part == "day":
        return _valid_datetime_result(x, day)
    if part == "dayofweek":
        return _valid_datetime_result(x, jnp.mod(days + 3, 7))
    raise ValueError(f"Unsupported date part {part!r}")


def _timeofday_units(x, unit: str):
    units_per_day = unit_ns("D") // unit_ns(unit)
    return jnp.mod(_safe_datetime_int(x), units_per_day)


def timeofday_value(x, unit: str):
    value = _timeofday_units(x, unit).astype(jnp.float64) * (unit_ns(unit) / unit_ns("s"))
    return jnp.where(jnp.isfinite(jnp.asarray(x)), value, jnp.nan)


def time_part_value(x, unit: str, part: str):
    seconds = jnp.floor_divide(_timeofday_units(x, unit) * unit_ns(unit), unit_ns("s"))
    if part == "hour":
        return _valid_datetime_result(x, jnp.floor_divide(seconds, 3600))
    if part == "minute":
        return _valid_datetime_result(x, jnp.mod(jnp.floor_divide(seconds, 60), 60))
    if part == "second":
        return _valid_datetime_result(x, jnp.mod(seconds, 60))
    raise ValueError(f"Unsupported time part {part!r}")


def datetime_round_value(x, unit: str, freq: str, mode: str):
    stride = duration_ns(freq, unit) // unit_ns(unit)
    if stride <= 0:
        raise ValueError("datetime rounding frequency must be positive")
    values = jnp.asarray(x, dtype=jnp.float64)
    stride_value = jnp.asarray(stride, dtype=jnp.float64)
    scaled = values / stride_value
    if mode == "floor":
        rounded = jnp.floor(scaled)
    elif mode == "ceil":
        rounded = jnp.ceil(scaled)
    elif mode == "round":
        rounded = jnp.floor(scaled + 0.5)
    else:
        raise ValueError(f"Unsupported datetime rounding mode {mode!r}")
    return jnp.where(jnp.isfinite(jnp.asarray(x)), rounded * stride_value, jnp.nan)


__all__ = [
    "ToDtOp",
    "date_part_value",
    "datetime_round_value",
    "dayofyear_value",
    "duration_ns",
    "time_part_value",
    "timeofday_value",
    "unit_ns",
]
