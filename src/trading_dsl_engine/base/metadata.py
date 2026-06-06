from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from importlib import import_module, util
from math import exp, inf, isclose, isfinite
from typing import Iterable, Mapping, Sequence

import numpy as np

from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, String, Universe


@dataclass(frozen=True)
class UnitInfo:
    """Lightweight trading-unit exponent vector with optional unxt conversion.

    Units are kept as semantic base labels (for example ``dollar`` and ``shares``)
    because trading schemas often need domain units that are not physical SI units.
    When ``unxt`` is installed, :meth:`to_unxt_quantity` exposes the same product as
    a ``unxt.Quantity`` for unit-aware downstream code that has compatible unit
    names registered with unxt/astropy.
    """

    powers: Mapping[str, float] = dataclass_field(default_factory=dict)

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

    def __mul__(self, other: UnitInfo) -> UnitInfo:
        powers = dict(self.powers)
        for name, power in other.powers.items():
            powers[name] = powers.get(name, 0.0) + power
        return UnitInfo(powers)

    def __truediv__(self, other: UnitInfo) -> UnitInfo:
        powers = dict(self.powers)
        for name, power in other.powers.items():
            powers[name] = powers.get(name, 0.0) - power
        return UnitInfo(powers)

    def __pow__(self, exponent: float) -> UnitInfo:
        return UnitInfo({name: power * exponent for name, power in self.powers.items()})

    def assert_compatible(self, other: UnitInfo, op_name: str) -> UnitInfo:
        if self != other:
            raise MetadataError(f"{op_name} requires compatible units, got {self.as_dict()} and {other.as_dict()}")
        return self

    def to_unxt_quantity(self, value: float = 1.0):
        if util.find_spec("unxt") is None:
            raise MetadataError("unxt is not installed; install trading_dsl_engine with unxt support")
        unxt = import_module("unxt")
        unit_expr = " * ".join(
            f"{name}**{power:g}" if not isclose(power, 1.0) else name
            for name, power in sorted(self.powers.items())
        ) or ""
        return unxt.Quantity(value, unit_expr)


@dataclass(frozen=True)
class ValueRange:
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
            if normalized in {"real", "unknown"}:
                return cls.real()
            if normalized in {"nonnegative", "non-negative", ">=0"}:
                return cls.nonnegative()
            if normalized in {"positive", ">0"}:
                return cls.positive()
            if normalized in {"bool", "boolean"}:
                return cls.boolean()
            raise MetadataError(f"Unknown range alias {value!r}")
        lower, upper = value
        return cls(float(lower), float(upper))

    def as_tuple(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    def to_immrax_interval(self):
        if util.find_spec("immrax") is None:
            raise MetadataError("immrax is not installed; install trading_dsl_engine with immrax support")
        immrax = import_module("immrax")
        return immrax.interval(np.asarray([self.lower], dtype=float), np.asarray([self.upper], dtype=float))


@dataclass(frozen=True)
class FieldSpec:
    units: UnitInfo = dataclass_field(default_factory=UnitInfo.dimensionless)
    range: ValueRange = dataclass_field(default_factory=ValueRange.unknown)
    types: frozenset[str] = dataclass_field(default_factory=frozenset)

    @classmethod
    def from_value(cls, value: FieldSpec | Mapping | None) -> FieldSpec:
        if value is None:
            return cls()
        if isinstance(value, FieldSpec):
            return value
        return cls(
            units=UnitInfo.from_value(value.get("units") or value.get("unit")),
            range=ValueRange.from_value(value.get("range")),
            types=frozenset(str(t) for t in value.get("types", ())),
        )


def field(
    units: UnitInfo | Mapping[str, float] | str | None = None,
    range: ValueRange | Sequence[float] | str | None = None,
    types: Iterable[str] = (),
) -> FieldSpec:
    return FieldSpec(UnitInfo.from_value(units), ValueRange.from_value(range), frozenset(str(t) for t in types))


@dataclass(frozen=True)
class TypeRelationGraph:
    types: tuple[str, ...] = ()
    implies: tuple[tuple[bool, ...], ...] = ()

    @classmethod
    def from_relations(
        cls,
        types: Iterable[str] = (),
        relations: Iterable[tuple[str, str]] = (),
    ) -> TypeRelationGraph:
        names = list(dict.fromkeys(str(t) for t in types))
        edges = [(str(a), str(b)) for a, b in relations]
        for a, b in edges:
            if a not in names:
                names.append(a)
            if b not in names:
                names.append(b)
        index = {name: i for i, name in enumerate(names)}
        matrix = [[i == j for j in range(len(names))] for i in range(len(names))]
        for a, b in edges:
            matrix[index[a]][index[b]] = True
        for k in range(len(names)):
            for i in range(len(names)):
                if matrix[i][k]:
                    for j in range(len(names)):
                        matrix[i][j] = matrix[i][j] or matrix[k][j]
        return cls(tuple(names), tuple(tuple(row) for row in matrix))

    def closure(self, direct_types: Iterable[str]) -> frozenset[str]:
        known = set(str(t) for t in direct_types)
        index = {name: i for i, name in enumerate(self.types)}
        for name in tuple(known):
            i = index.get(name)
            if i is not None:
                known.update(self.types[j] for j, related in enumerate(self.implies[i]) if related)
        return frozenset(known)

    def as_matrix(self) -> list[list[bool]]:
        return [list(row) for row in self.implies]


@dataclass(frozen=True)
class MetadataConfig:
    fields: Mapping[str, FieldSpec] = dataclass_field(default_factory=dict)
    type_graph: TypeRelationGraph = dataclass_field(default_factory=TypeRelationGraph)

    @classmethod
    def from_value(
        cls,
        value: MetadataConfig | Mapping[str, FieldSpec | Mapping] | None = None,
        *,
        type_relations: Iterable[tuple[str, str]] = (),
        types: Iterable[str] = (),
    ) -> MetadataConfig:
        if isinstance(value, MetadataConfig):
            if type_relations or types:
                raise MetadataError("Pass type relations either inside MetadataConfig or as compile_formula arguments, not both")
            return value
        fields = {str(name): FieldSpec.from_value(spec) for name, spec in (value or {}).items()}
        graph_types = list(types)
        for spec in fields.values():
            graph_types.extend(spec.types)
        return cls(fields, TypeRelationGraph.from_relations(graph_types, type_relations))


def metadata(
    fields: Mapping[str, FieldSpec | Mapping] | None = None,
    *,
    type_relations: Iterable[tuple[str, str]] = (),
    types: Iterable[str] = (),
) -> MetadataConfig:
    return MetadataConfig.from_value(fields, type_relations=type_relations, types=types)


@dataclass(frozen=True)
class FormulaMetadata:
    units: UnitInfo
    range: ValueRange
    types: frozenset[str]
    input_fields: Mapping[str, FieldSpec]
    type_graph: TypeRelationGraph

    def get_units(self) -> UnitInfo:
        return self.units

    def get_range(self) -> ValueRange:
        return self.range

    def get_types(self) -> frozenset[str]:
        return self.types


class MetadataError(ValueError):
    pass


def analyze_formula_metadata(expr: Expr, config: MetadataConfig | Mapping[str, FieldSpec | Mapping] | None) -> FormulaMetadata:
    cfg = MetadataConfig.from_value(config)

    def analyze(node: Expr) -> FieldSpec:
        if isinstance(node, Identifier):
            spec = cfg.fields.get(node.name, FieldSpec())
            return FieldSpec(spec.units, spec.range, cfg.type_graph.closure(spec.types))
        if isinstance(node, Number):
            return FieldSpec(UnitInfo.dimensionless(), ValueRange(float(node.value), float(node.value)), frozenset())
        if isinstance(node, String | Universe | KeyTuple):
            return FieldSpec()
        if not isinstance(node, Call):
            return FieldSpec()
        args = [analyze(arg) for arg in node.args]
        return _analyze_call(node, args)

    def _analyze_call(node: Call, args: list[FieldSpec]) -> FieldSpec:
        fn = node.fn
        if fn in {"add", "sub", "fillna"} and len(args) == 2:
            units = args[0].units.assert_compatible(args[1].units, fn)
            rng = _range_add(args[0].range, args[1].range) if fn == "add" else _range_sub(args[0].range, args[1].range)
            if fn == "fillna":
                rng = _range_union(args[0].range, args[1].range)
            return FieldSpec(units, rng, args[0].types & args[1].types)
        if fn == "mul" and len(args) == 2:
            return FieldSpec(args[0].units * args[1].units, _range_mul(args[0].range, args[1].range), frozenset())
        if fn in {"div", "floordiv"} and len(args) == 2:
            return FieldSpec(args[0].units / args[1].units, _range_div(args[0].range, args[1].range), frozenset())
        if fn == "pow" and len(args) == 2:
            exponent = _literal_number(node.args[1])
            units = UnitInfo.dimensionless() if exponent is None else args[0].units ** exponent
            rng = ValueRange.unknown() if exponent is None else _range_pow(args[0].range, exponent)
            return FieldSpec(units, rng, args[0].types if exponent == 1.0 else frozenset())
        if fn in {"abs", "ffill", "ewm", "shift", "cumsum", "purify"} and args:
            rng = _range_abs(args[0].range) if fn == "abs" else args[0].range
            return FieldSpec(args[0].units, rng, args[0].types)
        if fn == "mean" and args:
            return FieldSpec(args[0].units, args[0].range, args[0].types)
        if fn == "where" and len(args) == 3:
            units = args[1].units.assert_compatible(args[2].units, fn)
            return FieldSpec(units, _range_union(args[1].range, args[2].range), args[1].types & args[2].types)
        if fn in {"eq", "ne", "lt", "gt", "and", "and_", "or", "or_", "xor", "isnan"}:
            return FieldSpec(UnitInfo.dimensionless(), ValueRange.boolean(), frozenset({"boolean"}))
        if fn in {"ln", "exp", "sign", "fraction", "xs_rank", "xs_sort", "xstd", "bspline", "rbf_basis", "future_rbf_basis_sum"}:
            return FieldSpec(UnitInfo.dimensionless(), _known_dimensionless_range(fn, args), frozenset())
        if fn in {"ceil", "floor", "round", "mod"} and args:
            return FieldSpec(args[0].units, ValueRange.unknown(), args[0].types)
        if fn == "cat" and args:
            units = args[0].units
            for arg in args[1:]:
                units = units.assert_compatible(arg.units, fn)
            return FieldSpec(units, _range_union(*(arg.range for arg in args)), frozenset())
        if fn == "col" and args:
            return args[0]
        if fn == "buffer" and args:
            return args[0]
        if fn == "groupby" and len(args) == 3:
            return args[2]
        return FieldSpec()

    result = analyze(expr)
    return FormulaMetadata(result.units, result.range, result.types, cfg.fields, cfg.type_graph)


def _literal_number(expr: Expr) -> float | None:
    return float(expr.value) if isinstance(expr, Number) else None


def _range_union(*ranges: ValueRange) -> ValueRange:
    return ValueRange(min(r.lower for r in ranges), max(r.upper for r in ranges)) if ranges else ValueRange.unknown()


def _range_add(a: ValueRange, b: ValueRange) -> ValueRange:
    return ValueRange(a.lower + b.lower, a.upper + b.upper)


def _range_sub(a: ValueRange, b: ValueRange) -> ValueRange:
    return ValueRange(a.lower - b.upper, a.upper - b.lower)


def _finite_products(a: ValueRange, b: ValueRange) -> list[float]:
    vals = []
    for left in (a.lower, a.upper):
        for right in (b.lower, b.upper):
            vals.append(left * right)
    return vals


def _range_mul(a: ValueRange, b: ValueRange) -> ValueRange:
    vals = _finite_products(a, b)
    if any(np.isnan(v) for v in vals):
        return ValueRange.unknown()
    return ValueRange(min(vals), max(vals))


def _range_div(a: ValueRange, b: ValueRange) -> ValueRange:
    if b.lower <= 0.0 <= b.upper:
        return ValueRange.unknown()
    return _range_mul(a, ValueRange(1.0 / b.upper, 1.0 / b.lower))


def _range_pow(a: ValueRange, exponent: float) -> ValueRange:
    if exponent < 0.0 and a.lower <= 0.0 <= a.upper:
        return ValueRange.unknown()
    if not exponent.is_integer():
        return ValueRange.unknown() if a.lower < 0.0 else ValueRange(a.lower**exponent, a.upper**exponent)
    n = int(exponent)
    vals = [a.lower**n, a.upper**n]
    if n % 2 == 0 and a.lower <= 0.0 <= a.upper:
        vals.append(0.0)
    return ValueRange(min(vals), max(vals))


def _range_abs(a: ValueRange) -> ValueRange:
    upper = max(abs(a.lower), abs(a.upper))
    lower = 0.0 if a.lower <= 0.0 <= a.upper else min(abs(a.lower), abs(a.upper))
    return ValueRange(lower, upper)


def _known_dimensionless_range(fn: str, args: list[FieldSpec]) -> ValueRange:
    if fn == "exp" and args:
        lo = exp(args[0].range.lower) if isfinite(args[0].range.lower) else 0.0
        hi = exp(args[0].range.upper) if isfinite(args[0].range.upper) else inf
        return ValueRange(lo, hi)
    if fn in {"sign", "fraction", "bspline", "rbf_basis", "future_rbf_basis_sum"}:
        return ValueRange(-1.0, 1.0) if fn == "sign" else ValueRange(0.0, 1.0)
    return ValueRange.unknown()
