from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from math import inf, isclose, isfinite
from typing import Any, Callable, Iterable, Mapping, Sequence

import jax.numpy as jnp
import numpy as np
import unxt
from astropy import units as apyu

from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, String, Universe


class MetadataError(ValueError):
    pass


@dataclass(frozen=True)
class _Interval:
    lower: Any
    upper: Any


class _IntervalInclusion:
    """Small interval-inclusion adapter with the same shape as the future interval backend."""

    def interval(self, lower, upper=None) -> _Interval:
        if upper is None:
            upper = lower
        return _Interval(jnp.asarray(lower, dtype=float), jnp.asarray(upper, dtype=float))

    def natif(self, fn: Callable[..., Any]) -> Callable[..., _Interval]:
        def traced(*intervals: _Interval) -> _Interval:
            outputs = []
            for point in _interval_samples([item.lower for item in intervals], [item.upper for item in intervals]):
                outputs.append(jnp.asarray(fn(*point), dtype=float))
            stacked = jnp.stack([jnp.ravel(output) for output in outputs])
            return self.interval(jnp.nanmin(stacked), jnp.nanmax(stacked))

        return traced

    def interval_union(self, intervals: Iterable[_Interval]) -> _Interval:
        intervals = tuple(intervals)
        if not intervals:
            return self.interval(-inf, inf)
        lower = jnp.nanmin(jnp.stack([jnp.ravel(item.lower) for item in intervals]))
        upper = jnp.nanmax(jnp.stack([jnp.ravel(item.upper) for item in intervals]))
        return self.interval(lower, upper)


_INTERVAL_INCLUSION = _IntervalInclusion()


def _load_interval_inclusion() -> _IntervalInclusion:
    return _INTERVAL_INCLUSION


def _load_unxt():
    return unxt


_REGISTERED_UNXT_UNITS: set[str] = set()


def _ensure_unxt_domain_units(labels: Iterable[str]) -> None:
    new_units = []
    for label in labels:
        if label in _REGISTERED_UNXT_UNITS or not label:
            continue
        try:
            apyu.Unit(label)
        except Exception:
            new_units.append(apyu.def_unit(label))
        _REGISTERED_UNXT_UNITS.add(label)
    if new_units:
        apyu.add_enabled_units(new_units)


def _interval_samples(lows: Sequence[Any], ups: Sequence[Any]) -> Iterable[tuple[Any, ...]]:
    if not lows:
        return ()
    points: list[tuple[Any, ...]] = [()]
    for low, high in zip(lows, ups):
        candidates = [low, high]
        low_arr = np.asarray(low, dtype=float)
        high_arr = np.asarray(high, dtype=float)
        if np.all(np.isfinite(low_arr)) and np.all(np.isfinite(high_arr)):
            candidates.append((low + high) / 2.0)
        if np.all(low_arr <= 0.0) and np.all(0.0 <= high_arr):
            candidates.append(jnp.zeros_like(low))
        unique = []
        for candidate in candidates:
            if not any(np.array_equal(np.asarray(candidate), np.asarray(existing)) for existing in unique):
                unique.append(candidate)
        points = [prefix + (value,) for prefix in points for value in unique]
    return tuple(points)


def _finite_scalar_bounds(value) -> tuple[float, float]:
    lower = np.asarray(value.lower, dtype=float) if hasattr(value, "lower") else np.asarray(value, dtype=float)
    upper = np.asarray(value.upper, dtype=float) if hasattr(value, "upper") else lower
    if lower.size == 0 or upper.size == 0 or np.all(np.isnan(lower)) or np.all(np.isnan(upper)):
        return -inf, inf
    both = np.concatenate([np.ravel(lower), np.ravel(upper)])
    return float(np.nanmin(both)), float(np.nanmax(both))


@dataclass(frozen=True)
class UnitInfo:
    """Formula unit metadata backed by unxt quantities plus a sparse label view."""

    powers: Mapping[str, float] = dataclass_field(default_factory=dict)
    unknown: bool = False

    def __post_init__(self) -> None:
        cleaned = {
            str(name): float(power)
            for name, power in self.powers.items()
            if not isclose(float(power), 0.0, abs_tol=1e-12)
        }
        object.__setattr__(self, "powers", cleaned)

    @classmethod
    def dimensionless(cls) -> UnitInfo:
        return cls({})

    @classmethod
    def unknown_units(cls) -> UnitInfo:
        return cls({}, unknown=True)

    @classmethod
    def from_value(cls, value: UnitInfo | Mapping[str, float] | str | None) -> UnitInfo:
        if value is None:
            return cls.dimensionless()
        if isinstance(value, UnitInfo):
            return value
        if isinstance(value, str):
            return cls({value: 1.0})
        return cls(value)

    def as_dict(self) -> dict[str, float]:
        return dict(self.powers)

    def is_unknown(self) -> bool:
        return self.unknown

    def _unit_expr(self) -> str:
        numerator = []
        denominator = []
        for name, power in sorted(self.powers.items()):
            target = numerator if power > 0 else denominator
            magnitude = abs(power)
            target.append(name if isclose(magnitude, 1.0) else f"{name}**{magnitude:g}")
        expr = " * ".join(numerator) or "1"
        if denominator:
            expr += " / " + " / ".join(denominator)
        return "" if expr == "1" else expr

    def to_unxt_quantity(self, value: float = 1.0):
        if self.unknown:
            raise MetadataError("Cannot convert unknown formula units to unxt")
        _ensure_unxt_domain_units(self.powers)
        return _load_unxt().Quantity(jnp.asarray(value, dtype=float), self._unit_expr())

    def _combine(self, other: UnitInfo, op: Callable[[Any, Any], Any], powers_op: Callable[[float, float], float]) -> UnitInfo:
        if self.unknown or other.unknown:
            return UnitInfo.unknown_units()
        op(self.to_unxt_quantity(), other.to_unxt_quantity())
        labels = set(self.powers) | set(other.powers)
        return UnitInfo({label: powers_op(self.powers.get(label, 0.0), other.powers.get(label, 0.0)) for label in labels})

    def __mul__(self, other: UnitInfo) -> UnitInfo:
        return self._combine(other, lambda a, b: a * b, lambda a, b: a + b)

    def __truediv__(self, other: UnitInfo) -> UnitInfo:
        return self._combine(other, lambda a, b: a / b, lambda a, b: a - b)

    def __pow__(self, exponent: float) -> UnitInfo:
        if self.unknown:
            return UnitInfo.unknown_units()
        self.to_unxt_quantity() ** exponent
        return UnitInfo({name: power * exponent for name, power in self.powers.items()})

    def compatible_or_unknown(self, other: UnitInfo) -> UnitInfo:
        if self.unknown or other.unknown:
            return UnitInfo.unknown_units()
        try:
            self.to_unxt_quantity() + other.to_unxt_quantity()
        except Exception:
            return UnitInfo.unknown_units()
        return self if self == other else UnitInfo.unknown_units()


@dataclass(frozen=True)
class ValueRange:
    """Scalar bounds represented externally as lower/upper and internally as intervals."""

    lower: float = -inf
    upper: float = inf

    @classmethod
    def unknown(cls) -> ValueRange:
        return cls(-inf, inf)

    @classmethod
    def real(cls) -> ValueRange:
        return cls(-inf, inf)

    @classmethod
    def nonnegative(cls) -> ValueRange:
        return cls(0.0, inf)

    @classmethod
    def positive(cls) -> ValueRange:
        return cls(0.0, inf)

    @classmethod
    def boolean(cls) -> ValueRange:
        return cls(0.0, 1.0)

    @classmethod
    def from_value(cls, value: ValueRange | Sequence[float] | str | None) -> ValueRange:
        if value is None:
            return cls.unknown()
        if isinstance(value, ValueRange):
            return value
        if isinstance(value, str):
            normalized = value.lower().replace("_", "-")
            aliases = {
                "real": cls.real(),
                "unknown": cls.unknown(),
                "nonnegative": cls.nonnegative(),
                "non-negative": cls.nonnegative(),
                ">=0": cls.nonnegative(),
                "positive": cls.positive(),
                ">0": cls.positive(),
                "bool": cls.boolean(),
                "boolean": cls.boolean(),
            }
            if normalized not in aliases:
                raise MetadataError(f"Unknown range alias {value!r}")
            return aliases[normalized]
        lower, upper = value
        return cls(float(lower), float(upper))

    def as_tuple(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    def to_interval(self):
        return _load_interval_inclusion().interval(jnp.asarray([self.lower], dtype=float), jnp.asarray([self.upper], dtype=float))

    @classmethod
    def from_interval(cls, interval) -> ValueRange:
        lower, upper = _finite_scalar_bounds(interval)
        return cls(lower, upper)

    @classmethod
    def via_interval(cls, fn: Callable[..., Any], *ranges: ValueRange) -> ValueRange:
        if any(not isfinite(rng.lower) or not isfinite(rng.upper) for rng in ranges):
            return cls.unknown()
        inclusion = _load_interval_inclusion()
        intervals = [rng.to_interval() for rng in ranges]
        try:
            out = cls.from_interval(inclusion.natif(fn)(*intervals))
        except Exception:
            out = cls.unknown()
        if _is_well_ordered_finite(out):
            return out
        return cls.from_interval(_interval_sample_union(fn, intervals))




def _is_well_ordered_finite(value_range: ValueRange) -> bool:
    return (
        isfinite(value_range.lower)
        and isfinite(value_range.upper)
        and not np.isnan(value_range.lower)
        and not np.isnan(value_range.upper)
        and value_range.lower <= value_range.upper
    )


def _interval_sample_union(fn: Callable[..., Any], intervals: Sequence[Any]):
    inclusion = _load_interval_inclusion()
    lows = [interval.lower for interval in intervals]
    ups = [interval.upper for interval in intervals]
    result = None
    traced = inclusion.natif(fn)
    for point in _interval_samples(lows, ups):
        degenerate = [inclusion.interval(value, value) for value in point]
        try:
            current = traced(*degenerate)
        except Exception:
            value = fn(*point)
            current = inclusion.interval(value, value)
        result = current if result is None else inclusion.interval_union([result, current])
    if result is None:
        result = inclusion.interval(-inf, inf)
    return result

@dataclass(frozen=True)
class FieldSpec:
    units: UnitInfo = dataclass_field(default_factory=UnitInfo.dimensionless)
    range: ValueRange = dataclass_field(default_factory=ValueRange.unknown)
    types: frozenset[str] = dataclass_field(default_factory=frozenset)
    width: int | None = 1

    @classmethod
    def from_value(cls, value: FieldSpec | Mapping | None) -> FieldSpec:
        if value is None:
            return cls()
        if isinstance(value, FieldSpec):
            return value
        return cls(
            units=UnitInfo.from_value(value.get("units")),
            range=ValueRange.from_value(value.get("range")),
            types=frozenset(value.get("types", ())),
            width=value.get("width", 1),
        )


def field(
    *,
    units: UnitInfo | Mapping[str, float] | str | None = None,
    range: ValueRange | Sequence[float] | str | None = None,
    types: Iterable[str] = (),
    width: int | None = 1,
) -> FieldSpec:
    return FieldSpec(UnitInfo.from_value(units), ValueRange.from_value(range), frozenset(types), width)


@dataclass(frozen=True)
class TypeRelationGraph:
    relations: frozenset[tuple[str, str]] = dataclass_field(default_factory=frozenset)

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[str, str]] | None) -> TypeRelationGraph:
        return cls(frozenset((str(a), str(b)) for a, b in (edges or ())))

    @property
    def types(self) -> tuple[str, ...]:
        return tuple(sorted({item for edge in self.relations for item in edge}))

    def closure(self, types: Iterable[str]) -> frozenset[str]:
        seen = set(types)
        changed = True
        while changed:
            changed = False
            for src, dst in self.relations:
                if src in seen and dst not in seen:
                    seen.add(dst)
                    changed = True
        return frozenset(seen)

    def as_matrix(self) -> tuple[tuple[bool, ...], ...]:
        labels = self.types
        closed = {src: self.closure((src,)) for src in labels}
        return tuple(tuple(dst in closed[src] for dst in labels) for src in labels)


@dataclass(frozen=True)
class MetadataConfig:
    fields: Mapping[str, FieldSpec] = dataclass_field(default_factory=dict)
    type_graph: TypeRelationGraph = dataclass_field(default_factory=TypeRelationGraph)

    @classmethod
    def from_value(cls, value: MetadataConfig | Mapping[str, Any] | None, *, type_relations=None) -> MetadataConfig:
        if isinstance(value, MetadataConfig):
            if not type_relations:
                return value
            return cls(value.fields, TypeRelationGraph.from_edges(type_relations))
        fields = {name: FieldSpec.from_value(spec) for name, spec in (value or {}).items()}
        return cls(fields, TypeRelationGraph.from_edges(type_relations))


def metadata(fields: Mapping[str, FieldSpec | Mapping], *, type_relations=None) -> MetadataConfig:
    return MetadataConfig.from_value(fields, type_relations=type_relations)


@dataclass(frozen=True)
class NodeMetadata:
    label: str
    expr_key: tuple
    metadata: FieldSpec


@dataclass(frozen=True)
class FormulaMetadata:
    units: UnitInfo
    range: ValueRange
    types: frozenset[str]
    fields: Mapping[str, FieldSpec]
    type_graph: TypeRelationGraph
    nodes: tuple[NodeMetadata, ...] = ()

    @property
    def input_fields(self) -> Mapping[str, FieldSpec]:
        return self.fields

    def get_units(self) -> UnitInfo:
        return self.units

    def get_range(self) -> ValueRange:
        return self.range

    def get_types(self) -> frozenset[str]:
        return self.types

    def get_node_metadata(self, label: str | None = None) -> tuple[NodeMetadata, ...]:
        return self.nodes if label is None else tuple(node for node in self.nodes if node.label == label)

    def get_node_types(self, label: str) -> tuple[frozenset[str], ...]:
        return tuple(node.metadata.types for node in self.get_node_metadata(label))


def _expr_key(node: Expr) -> tuple:
    if hasattr(node, "fn") and hasattr(node, "output_width") and hasattr(node, "args"):
        return ("stateless", getattr(node, "name", None), tuple(_expr_key(arg) for arg in node.args))
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", node.value)
    if isinstance(node, String):
        return ("str", node.value)
    if isinstance(node, Universe):
        return ("univ", node.groups)
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_expr_key(item) for item in node.items))
    if isinstance(node, Call):
        return ("call", node.fn, tuple(_expr_key(arg) for arg in node.args), tuple((k, _expr_key(v)) for k, v in node.kwargs))
    return (type(node).__name__, id(node))


def _same_expr(a: Expr, b: Expr) -> bool:
    return _expr_key(a) == _expr_key(b)


def _label(node: Expr) -> str:
    if hasattr(node, "fn") and hasattr(node, "output_width") and hasattr(node, "args"):
        return getattr(node, "name", None) or getattr(getattr(node, "fn", None), "__name__", "stateless")
    if isinstance(node, Identifier):
        return node.name
    if isinstance(node, Number):
        return "literal"
    if isinstance(node, String):
        return "string"
    if isinstance(node, Universe):
        return "univ"
    if isinstance(node, KeyTuple):
        return "tuple"
    if isinstance(node, Call):
        return node.fn
    return type(node).__name__


def _number(node: Expr) -> float | None:
    return float(node.value) if isinstance(node, Number) else None


def _is_literal(node: Expr, value: float) -> bool:
    num = _number(node)
    return num is not None and isclose(num, value)


def _is_nan_literal(node: Expr) -> bool:
    num = _number(node)
    return num is not None and np.isnan(num)


def _constant(value: float, types: Iterable[str] = ()) -> FieldSpec:
    return FieldSpec(UnitInfo.dimensionless(), ValueRange(value, value), frozenset(types))


def _truthy(spec: FieldSpec) -> FieldSpec:
    return FieldSpec(UnitInfo.dimensionless(), ValueRange.boolean(), frozenset({"boolean"}), spec.width)


def _literal_width(node: Expr) -> int | None:
    num = _number(node)
    return int(num) if num is not None else None


def _types_for_numeric(fn: str, args: Sequence[FieldSpec], units: UnitInfo) -> frozenset[str]:
    if fn in {"eq", "ne", "lt", "gt", "and", "and_", "or", "or_", "xor", "isnan"}:
        return frozenset({"boolean"})
    if units.as_dict() == {} and not units.is_unknown():
        if fn in {"div", "floordiv"} and len(args) == 2 and args[0].units == args[1].units:
            return frozenset({"ratio"})
        if fn == "sub" and args and "ratio" in args[0].types and any(isclose(v.range.lower, 1.0) and isclose(v.range.upper, 1.0) for v in args[1:]):
            return frozenset({"return"})
    return args[0].types if args and all(arg.types == args[0].types for arg in args) else frozenset()


def _auto_op(node: Call):
    from trading_dsl_engine.jax_flat.ops import ANY_ARITY, NaryOp, OP_FACTORIES

    factory = OP_FACTORIES.get((node.fn, len(node.args))) or OP_FACTORIES.get((node.fn, ANY_ARITY))
    if factory is None:
        return None
    literal_args = [_number(arg) for arg in node.args]
    nonliteral = [i for i, value in enumerate(literal_args) if value is None]
    probes = [0.0 for _ in node.args]
    for i, value in enumerate(literal_args):
        if value is not None:
            probes[i] = value
    try:
        op = factory(*probes) if factory is OP_FACTORIES.get((node.fn, ANY_ARITY)) else factory()
    except Exception:
        try:
            op = factory()
        except Exception:
            return None
    if not isinstance(op, NaryOp):
        return None
    return op.fn, nonliteral


def _call_range(fn: Callable[..., Any], ranges: Sequence[ValueRange]) -> ValueRange:
    return ValueRange.via_interval(lambda *xs: fn(*xs), *ranges)


def _range_union(ranges: Sequence[ValueRange]) -> ValueRange:
    finite_lowers = [rng.lower for rng in ranges]
    finite_uppers = [rng.upper for rng in ranges]
    return ValueRange(min(finite_lowers), max(finite_uppers)) if ranges else ValueRange.unknown()


def _div_value_range(numerator: ValueRange, denominator: ValueRange) -> ValueRange:
    if denominator.lower <= 0.0 <= denominator.upper:
        if numerator.lower >= 0.0 and denominator.upper > 0.0 and denominator.lower >= 0.0:
            return ValueRange(0.0, inf)
        return ValueRange.unknown()
    if numerator.lower >= 0.0 and denominator.lower >= 0.0:
        lower = numerator.lower / denominator.upper if isfinite(denominator.upper) else 0.0
        upper = numerator.upper / denominator.lower if isfinite(numerator.upper) and denominator.lower > 0.0 else inf
        return ValueRange(lower, upper)
    return _call_range(lambda a, b: a / b, [numerator, denominator])


def _core_call_spec(fn_name: str, node: Call, args: list[FieldSpec]) -> FieldSpec | None:
    if fn_name == "add" and len(args) == 2:
        if _is_literal(node.args[0], 0.0):
            return args[1]
        if _is_literal(node.args[1], 0.0):
            return args[0]
        if _is_literal(node.args[0], 0.0):
            return FieldSpec(args[1].units, _call_range(lambda b: -b, [args[1].range]), args[1].types)
        units = args[0].units.compatible_or_unknown(args[1].units)
        rng = _call_range(lambda a, b: a + b, [args[0].range, args[1].range]) if not units.is_unknown() else ValueRange.unknown()
        return FieldSpec(units, rng, _types_for_numeric(fn_name, args, units))
    if fn_name == "sub" and len(args) == 2:
        if _same_expr(node.args[0], node.args[1]):
            return FieldSpec(args[0].units, ValueRange(0.0, 0.0), args[0].types)
        if _is_literal(node.args[1], 0.0):
            return args[0]
        if _is_literal(node.args[0], 0.0):
            return FieldSpec(args[1].units, _call_range(lambda b: -b, [args[1].range]), args[1].types)
        units = args[0].units.compatible_or_unknown(args[1].units)
        rng = _call_range(lambda a, b: a - b, [args[0].range, args[1].range]) if not units.is_unknown() else ValueRange.unknown()
        return FieldSpec(units, rng, _types_for_numeric(fn_name, args, units))
    if fn_name == "mul" and len(args) == 2:
        if _is_literal(node.args[0], 0.0):
            return FieldSpec(args[1].units, ValueRange(0.0, 0.0), args[1].types)
        if _is_literal(node.args[1], 0.0):
            return FieldSpec(args[0].units, ValueRange(0.0, 0.0), args[0].types)
        if _is_literal(node.args[0], 1.0):
            return args[1]
        if _is_literal(node.args[1], 1.0):
            return args[0]
        units = args[0].units * args[1].units
        return FieldSpec(units, _call_range(lambda a, b: a * b, [args[0].range, args[1].range]), _types_for_numeric(fn_name, args, units))
    if fn_name in {"div", "floordiv"} and len(args) == 2:
        if _same_expr(node.args[0], node.args[1]):
            return FieldSpec(UnitInfo.dimensionless(), ValueRange(1.0, 1.0), frozenset({"ratio"}))
        if _is_literal(node.args[0], 0.0):
            return _constant(0.0)
        if _is_literal(node.args[1], 1.0):
            return args[0]
        units = args[0].units / args[1].units
        rng = _call_range(lambda a, b: jnp.floor_divide(a, b), [args[0].range, args[1].range]) if fn_name == "floordiv" else _div_value_range(args[0].range, args[1].range)
        return FieldSpec(units, rng, _types_for_numeric(fn_name, args, units))
    if fn_name == "pow" and len(args) == 2:
        if _is_literal(node.args[0], 1.0):
            return _constant(1.0)
        exponent = _number(node.args[1])
        if exponent is None:
            return FieldSpec(UnitInfo.dimensionless(), ValueRange.unknown())
        if isclose(exponent, 0.0):
            return _constant(1.0)
        if isclose(exponent, 1.0):
            return args[0]
        units = args[0].units ** exponent
        return FieldSpec(units, _call_range(lambda a: a**exponent, [args[0].range]), _types_for_numeric(fn_name, args, units))
    if fn_name == "abs" and len(args) == 1:
        return FieldSpec(args[0].units, _call_range(jnp.abs, [args[0].range]), args[0].types)
    if fn_name == "mod" and len(args) == 2:
        if _same_expr(node.args[0], node.args[1]) or _is_literal(node.args[0], 0.0) or _is_literal(node.args[1], 1.0):
            return FieldSpec(args[0].units, ValueRange(0.0, 0.0), args[0].types)
        return FieldSpec(args[0].units, _call_range(jnp.mod, [args[0].range, args[1].range]), args[0].types)
    return None


def _auto_trace(node: Call, args: list[FieldSpec]) -> FieldSpec | None:
    built = _auto_op(node)
    if built is None:
        return None
    fn, nonliteral = built
    ranges = [args[i].range for i in nonliteral]

    def wrapped(*xs):
        values = []
        iterator = iter(xs)
        for index, expr in enumerate(node.args):
            literal = _number(expr)
            values.append(next(iterator) if index in nonliteral else literal)
        return fn(*values)

    rng = _call_range(wrapped, ranges)
    units = args[nonliteral[0]].units if nonliteral and all(args[i].units == args[nonliteral[0]].units for i in nonliteral) else UnitInfo.dimensionless()
    out_type = frozenset({"boolean"}) if rng == ValueRange.boolean() else _types_for_numeric(node.fn, [args[i] for i in nonliteral], units)
    width = args[nonliteral[0]].width if nonliteral else 1
    return FieldSpec(units, rng, out_type, width)


def analyze_formula_metadata(expr: Expr, config: MetadataConfig | Mapping[str, Any] | None = None) -> FormulaMetadata:
    cfg = MetadataConfig.from_value(config)
    nodes: list[NodeMetadata] = []
    self_stack: list[FieldSpec] = []

    def close(spec: FieldSpec) -> FieldSpec:
        return FieldSpec(spec.units, spec.range, cfg.type_graph.closure(spec.types), spec.width)

    def record(node: Expr, spec: FieldSpec) -> FieldSpec:
        spec = close(spec)
        nodes.append(NodeMetadata(_label(node), _expr_key(node), spec))
        return spec

    def analyze(node: Expr) -> FieldSpec:
        if isinstance(node, Identifier):
            if node.name == "self_" and self_stack:
                return record(node, self_stack[-1])
            return record(node, cfg.fields.get(node.name, FieldSpec()))
        if isinstance(node, Number):
            return record(node, _constant(float(node.value)))
        if isinstance(node, (String, Universe, KeyTuple)):
            return record(node, FieldSpec())
        if hasattr(node, "fn") and hasattr(node, "output_width") and hasattr(node, "args"):
            args = [analyze(arg) for arg in node.args]
            lowered_name = (getattr(node, "name", None) or "").lower()
            units = args[0].units if args else UnitInfo.dimensionless()
            types = args[0].types if args and all(arg.types == args[0].types for arg in args) else frozenset()
            if "pct" in lowered_name or "ratio" in lowered_name or "nonnegative" in lowered_name:
                rng = ValueRange(0.0, 1.0)
                units = UnitInfo.dimensionless()
                types = frozenset({"ratio"})
            elif "volume" in lowered_name:
                rng = args[0].range if args else ValueRange.unknown()
                units = UnitInfo({"shares": 1.0})
                types = frozenset({"volume"})
            else:
                rng = _call_range(node.fn, [arg.range for arg in args])
            width = getattr(node, "output_width", None) or (args[0].width if args else 1)
            return record(node, FieldSpec(units, rng, types, width))
        if not isinstance(node, Call):
            return record(node, FieldSpec())

        if node.fn == "groupby" and len(node.args) == 3:
            _key = analyze(node.args[0])
            lhs = analyze(node.args[1])
            self_stack.append(lhs)
            try:
                op_meta = analyze(node.args[2])
            finally:
                self_stack.pop()
            return record(node, op_meta if op_meta.range != ValueRange.unknown() or op_meta.types or op_meta.units.as_dict() else lhs)

        args = [analyze(arg) for arg in node.args]
        fn = node.fn
        spec = _core_call_spec(fn, node, args)
        if spec is None and fn == "where" and len(args) == 3:
            if _same_expr(node.args[1], node.args[2]):
                spec = args[1]
            elif _is_literal(node.args[0], 1.0):
                spec = args[1]
            elif _is_literal(node.args[0], 0.0):
                spec = args[2]
            elif _is_nan_literal(node.args[1]):
                spec = args[2]
            elif _is_nan_literal(node.args[2]):
                spec = args[1]
            else:
                units = args[1].units.compatible_or_unknown(args[2].units)
                types = args[1].types if args[1].types == args[2].types else frozenset()
                spec = FieldSpec(units, _range_union([args[1].range, args[2].range]), types, args[1].width)
        if spec is None and fn in {"eq", "ne", "lt", "gt"} and len(args) == 2:
            if _same_expr(node.args[0], node.args[1]):
                spec = _constant(1.0 if fn in {"eq"} else 0.0, {"boolean"})
            else:
                ops = {"eq": lambda a, b: a == b, "ne": lambda a, b: a != b, "lt": lambda a, b: a < b, "gt": lambda a, b: a > b}
                spec = FieldSpec(UnitInfo.dimensionless(), _call_range(ops[fn], [args[0].range, args[1].range]), frozenset({"boolean"}))
        if spec is None and fn in {"and", "and_", "or", "or_", "xor"} and len(args) == 2:
            if fn == "xor" and _same_expr(node.args[0], node.args[1]):
                spec = _constant(0.0, {"boolean"})
            elif fn in {"and", "and_"} and (_is_literal(node.args[0], 0.0) or _is_literal(node.args[1], 0.0)):
                spec = _constant(0.0, {"boolean"})
            elif fn in {"or", "or_"} and (_is_literal(node.args[0], 1.0) or _is_literal(node.args[1], 1.0)):
                spec = _constant(1.0, {"boolean"})
            else:
                ops = {
                "and": lambda a, b: jnp.logical_and(a, b),
                "and_": lambda a, b: jnp.logical_and(a, b),
                "or": lambda a, b: jnp.logical_or(a, b),
                "or_": lambda a, b: jnp.logical_or(a, b),
                "xor": lambda a, b: jnp.logical_xor(a, b),
            }
                spec = FieldSpec(UnitInfo.dimensionless(), _call_range(ops[fn], [args[0].range, args[1].range]), frozenset({"boolean"}))
        if spec is None and fn == "isnan" and len(args) == 1:
            spec = _constant(0.0, {"boolean"}) if isinstance(node.args[0], Number) else FieldSpec(UnitInfo.dimensionless(), ValueRange.boolean(), frozenset({"boolean"}), args[0].width)
        if spec is None and fn == "xs_rank" and len(args) == 1:
            spec = FieldSpec(UnitInfo.dimensionless(), ValueRange(0.0, 1.0), frozenset({"dimensionless"}), args[0].width)
        if spec is None and fn == "clip" and len(args) == 3:
            lower = _number(node.args[1])
            upper = _number(node.args[2])
            if lower is not None and upper is not None:
                lo, hi = (lower, upper) if lower <= upper else (upper, lower)
                in_range = args[0].range
                bounded_lower = max(in_range.lower, lo) if isfinite(in_range.lower) else lo
                bounded_upper = min(in_range.upper, hi) if isfinite(in_range.upper) else hi
                spec = FieldSpec(args[0].units, ValueRange(bounded_lower, bounded_upper), args[0].types, args[0].width)
        if spec is None and fn in {"rbf_basis", "future_rbf_basis_sum", "bspline"}:
            width = _literal_width(node.args[3]) if len(node.args) > 3 else 1
            spec = FieldSpec(UnitInfo.dimensionless(), ValueRange(0.0, 1.0), frozenset(), width)
        if spec is None and fn in {"InstrumentBasisMean", "get_beta"}:
            spec = FieldSpec(UnitInfo.dimensionless(), ValueRange(0.0, 1.0), frozenset(), args[0].width if args else 1)
        if spec is None and fn in {"shift", "delay", "lag", "col", "buffer", "ffill", "cumsum"} and args:
            spec = args[0]
        if spec is None and fn == "fillna" and args:
            if len(args) == 2:
                units = args[0].units.compatible_or_unknown(args[1].units)
                types = args[0].types if args[0].types == args[1].types else frozenset()
                spec = FieldSpec(units, _range_union([args[0].range, args[1].range]), types, args[0].width)
            else:
                spec = args[0]
        if spec is None and fn == "groupby" and len(args) == 3:
            spec = args[2]
        if spec is None and fn == "cat" and args:
            units = args[0].units
            for arg in args[1:]:
                units = units.compatible_or_unknown(arg.units)
            width = None if any(arg.width is None for arg in args) else sum(int(arg.width) for arg in args)
            types = args[0].types if all(arg.types == args[0].types for arg in args) else frozenset()
            spec = FieldSpec(units, _range_union([arg.range for arg in args]), types, width)
        if spec is None and fn == "einsum" and args:
            numeric_args = args[:-1] if node.args and isinstance(node.args[-1], String) else args
            units = UnitInfo.dimensionless()
            for arg in numeric_args:
                units = units * arg.units
            if len(numeric_args) >= 2:
                product_range = _call_range(lambda a, b: a * b, [numeric_args[0].range, numeric_args[1].range])
                width = min((arg.width or 1) for arg in numeric_args[:2])
                rng = ValueRange(product_range.lower * width, product_range.upper * width)
                types = numeric_args[1].types
            elif numeric_args:
                rng = numeric_args[0].range
                types = numeric_args[0].types
            else:
                rng = ValueRange.unknown()
                types = frozenset()
            spec = FieldSpec(units, rng, types)
        if spec is None:
            spec = _auto_trace(node, args) or FieldSpec()
        return record(node, spec)

    root = analyze(expr)
    return FormulaMetadata(root.units, root.range, root.types, cfg.fields, cfg.type_graph, tuple(nodes))
