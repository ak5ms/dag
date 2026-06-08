from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from importlib import import_module, util
from itertools import product
from math import exp, inf, isclose, isfinite
from typing import Any, Iterable, Mapping, Sequence

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

    def __mul__(self, other: UnitInfo) -> UnitInfo:
        if self.unknown or other.unknown:
            return UnitInfo.unknown_units()
        powers = dict(self.powers)
        for name, power in other.powers.items():
            powers[name] = powers.get(name, 0.0) + power
        return UnitInfo(powers)

    def __truediv__(self, other: UnitInfo) -> UnitInfo:
        if self.unknown or other.unknown:
            return UnitInfo.unknown_units()
        powers = dict(self.powers)
        for name, power in other.powers.items():
            powers[name] = powers.get(name, 0.0) - power
        return UnitInfo(powers)

    def __pow__(self, exponent: float) -> UnitInfo:
        if self.unknown:
            return UnitInfo.unknown_units()
        return UnitInfo({name: power * exponent for name, power in self.powers.items()})

    def compatible_or_unknown(self, other: UnitInfo) -> UnitInfo:
        if self == other:
            return self
        return UnitInfo.unknown_units()

    def assert_compatible(self, other: UnitInfo, op_name: str) -> UnitInfo:
        if self != other:
            raise MetadataError(f"{op_name} requires compatible units, got {self.as_dict()} and {other.as_dict()}")
        return self

    def to_unxt_quantity(self, value: float = 1.0):
        if self.unknown:
            raise MetadataError("Cannot convert unknown formula units to unxt")
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
    width: int | None = 1

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
            width=value.get("width", 1),
        )


def field(
    units: UnitInfo | Mapping[str, float] | str | None = None,
    range: ValueRange | Sequence[float] | str | None = None,
    types: Iterable[str] = (),
    width: int | None = 1,
) -> FieldSpec:
    return FieldSpec(UnitInfo.from_value(units), ValueRange.from_value(range), frozenset(str(t) for t in types), width)


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
class NodeMetadata:
    label: str
    key: tuple
    metadata: FieldSpec


@dataclass(frozen=True)
class FormulaMetadata:
    units: UnitInfo
    range: ValueRange
    types: frozenset[str]
    input_fields: Mapping[str, FieldSpec]
    type_graph: TypeRelationGraph
    nodes: tuple[NodeMetadata, ...] = ()

    def get_units(self) -> UnitInfo:
        return self.units

    def get_range(self) -> ValueRange:
        return self.range

    def get_types(self) -> frozenset[str]:
        return self.types

    def get_node_metadata(self, label: str | None = None) -> tuple[NodeMetadata, ...]:
        if label is None:
            return self.nodes
        return tuple(node for node in self.nodes if node.label == label)

    def get_node_types(self, label: str) -> tuple[frozenset[str], ...]:
        return tuple(node.metadata.types for node in self.get_node_metadata(label))


class MetadataError(ValueError):
    pass


def _metadata_expr_key(node: Expr) -> tuple:
    if isinstance(node, Identifier):
        return ("id", node.name)
    if isinstance(node, Number):
        return ("num", node.value)
    if isinstance(node, String):
        return ("str", node.value)
    if isinstance(node, Call):
        return (
            "call",
            node.fn,
            tuple(_metadata_expr_key(arg) for arg in node.args),
            tuple((key, _metadata_expr_key(value)) for key, value in node.kwargs),
        )
    if isinstance(node, Universe):
        return ("univ", node.groups)
    if isinstance(node, KeyTuple):
        return ("tuple", tuple(_metadata_expr_key(item) for item in node.items))
    return ("unknown", type(node).__name__, id(node))


def _same_expr(left: Expr, right: Expr) -> bool:
    return _metadata_expr_key(left) == _metadata_expr_key(right)


def _metadata_node_label(node: Expr) -> str:
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
    if type(node).__name__ == "StatelessJaxCall":
        return getattr(node, "name", None) or getattr(getattr(node, "fn", None), "__name__", "stateless")
    return type(node).__name__


def _literal_number(expr: Expr) -> float | None:
    return float(expr.value) if isinstance(expr, Number) else None


def _is_literal(expr: Expr, value: float) -> bool:
    literal = _literal_number(expr)
    return literal is not None and isclose(literal, value, rel_tol=0.0, abs_tol=1e-12)


def _is_nan_literal(expr: Expr) -> bool:
    literal = _literal_number(expr)
    return literal is not None and np.isnan(literal)


def _constant_field(value: float, types: Iterable[str] = ()) -> FieldSpec:
    return FieldSpec(UnitInfo.dimensionless(), ValueRange(float(value), float(value)), frozenset(types))


def _scale_field(spec: FieldSpec, factor: float) -> FieldSpec:
    return FieldSpec(spec.units, _range_mul(spec.range, ValueRange(factor, factor)), spec.types, spec.width)


def _truthy_field(spec: FieldSpec) -> FieldSpec:
    if spec.range.lower == 0.0 and spec.range.upper == 0.0:
        return _constant_field(0.0, {"boolean"})
    if spec.range.lower > 0.0 or spec.range.upper < 0.0:
        return _constant_field(1.0, {"boolean"})
    return FieldSpec(UnitInfo.dimensionless(), ValueRange.boolean(), frozenset({"boolean"}))


@dataclass(frozen=True)
class _TraceResult:
    value_range: ValueRange
    values: tuple[float, ...]


def _sample_range_values(value_range: ValueRange) -> tuple[float, ...]:
    candidates = [value_range.lower, value_range.upper]
    if value_range.lower <= 0.0 <= value_range.upper:
        candidates.append(0.0)
    if isfinite(value_range.lower) and isfinite(value_range.upper):
        candidates.append((value_range.lower + value_range.upper) / 2.0)
    finite = []
    for value in candidates:
        if isfinite(value) and value not in finite:
            finite.append(float(value))
    return tuple(finite)


def _as_finite_values(value: Any) -> tuple[float, ...]:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return ()
    return tuple(float(v) for v in arr.reshape(-1))


def _trace_numeric_range(fn, child_ranges: Sequence[ValueRange]) -> _TraceResult | None:
    samples = [_sample_range_values(rng) for rng in child_ranges]
    if not samples or any(len(values) == 0 for values in samples):
        return None
    outputs = []
    for point in product(*samples):
        try:
            outputs.extend(_as_finite_values(fn(*point)))
        except Exception:
            return None
    if not outputs:
        return None
    return _TraceResult(ValueRange(min(outputs), max(outputs)), tuple(outputs))


def _build_trace_op(node: Call):
    from trading_dsl_engine.jax_flat.ops import ANY_ARITY, NaryOp, OP_FACTORIES

    factory = OP_FACTORIES.get((node.fn, len(node.args)))
    if factory is None:
        factory = OP_FACTORIES.get((node.fn, ANY_ARITY))
        if factory is None:
            return None
        static_args = tuple(arg.value for arg in node.args if isinstance(arg, String))
        op = factory(*static_args)
    else:
        op = factory()
    return op if isinstance(op, NaryOp) else None


def _auto_trace_field(node: Call, args: Sequence[FieldSpec]) -> FieldSpec | None:
    op = _build_trace_op(node)
    if op is None:
        return None
    traced = _trace_numeric_range(op.fn, [arg.range for arg in args])
    if traced is None:
        return None
    unit_preserving_ops = {"ceil", "floor", "round", "fraction", "purify"}
    output_types = frozenset({"boolean"}) if all(value in (0.0, 1.0) for value in traced.values) else frozenset()
    if node.fn in unit_preserving_ops and args:
        return FieldSpec(args[0].units, traced.value_range, args[0].types if not output_types else output_types, args[0].width)
    if node.fn in {"abs"} and args:
        return FieldSpec(args[0].units, traced.value_range, args[0].types, args[0].width)
    return FieldSpec(UnitInfo.dimensionless(), traced.value_range, output_types, getattr(op, "output_width", 1))


def _auto_trace_stateless(node, args: Sequence[FieldSpec]) -> FieldSpec:
    traced = _trace_numeric_range(node.fn, [arg.range for arg in args])
    name = node.name or getattr(node.fn, "__name__", "")
    width = node.output_width if node.output_width is not None else (args[0].width if args else 1)
    if traced is not None:
        output_types = frozenset({"boolean"}) if all(value in (0.0, 1.0) for value in traced.values) else frozenset()
        units = args[0].units if name in {"nonnegative", "volume_for_fit_session", "volume_for_seen_session"} and args else UnitInfo.dimensionless()
        if name in {"volume_for_fit_session", "volume_for_seen_session"}:
            output_types = frozenset({"volume"})
        if name == "pct_seen_session_volume":
            output_types = frozenset({"ratio"})
        return FieldSpec(units, traced.value_range, output_types, width)
    if name == "nonnegative" and args:
        upper = args[0].range.upper if isfinite(args[0].range.upper) and args[0].range.upper > 0.0 else inf
        return FieldSpec(args[0].units, ValueRange(0.0, upper), args[0].types, width)
    if name in {"volume_for_fit_session", "volume_for_seen_session"} and args:
        upper = args[0].range.upper if isfinite(args[0].range.upper) else inf
        return FieldSpec(args[0].units, ValueRange(0.0, max(0.0, upper)), frozenset({"volume"}), width)
    if name == "pct_seen_session_volume":
        return FieldSpec(UnitInfo.dimensionless(), ValueRange(0.0, 1.0), frozenset({"ratio"}), width)
    return FieldSpec(width=width)


def analyze_formula_metadata(expr: Expr, config: MetadataConfig | Mapping[str, FieldSpec | Mapping] | None) -> FormulaMetadata:
    cfg = MetadataConfig.from_value(config)
    node_metadata: list[NodeMetadata] = []

    def analyze(node: Expr, local_specs: Mapping[str, FieldSpec] | None = None) -> FieldSpec:
        spec = _analyze_node(node, local_specs or {})
        node_metadata.append(NodeMetadata(_metadata_node_label(node), _metadata_expr_key(node), spec))
        return spec

    def _analyze_node(node: Expr, local_specs: Mapping[str, FieldSpec]) -> FieldSpec:
        if isinstance(node, Identifier):
            spec = local_specs.get(node.name) or cfg.fields.get(node.name, FieldSpec())
            return FieldSpec(spec.units, spec.range, cfg.type_graph.closure(spec.types), spec.width)
        if isinstance(node, Number):
            return FieldSpec(UnitInfo.dimensionless(), ValueRange(float(node.value), float(node.value)), frozenset())
        if isinstance(node, String | Universe | KeyTuple):
            return FieldSpec()
        if type(node).__name__ == "StatelessJaxCall":
            args = [analyze(arg, local_specs) for arg in node.args]
            return _auto_trace_stateless(node, args)
        if not isinstance(node, Call):
            return FieldSpec()
        if node.fn == "groupby" and len(node.args) == 3:
            key_spec = analyze(node.args[0], local_specs)
            lhs_spec = analyze(node.args[1], local_specs)
            rhs_spec = analyze(node.args[2], {**local_specs, "self_": lhs_spec})
            return _analyze_call(node, [key_spec, lhs_spec, rhs_spec])
        args = [analyze(arg, local_specs) for arg in node.args]
        return _analyze_call(node, args)

    def _analyze_call(node: Call, args: list[FieldSpec]) -> FieldSpec:
        fn = node.fn
        if fn == "add" and len(args) == 2:
            if _is_literal(node.args[0], 0.0):
                return args[1]
            if _is_literal(node.args[1], 0.0):
                return args[0]
            units = args[0].units.compatible_or_unknown(args[1].units)
            return FieldSpec(units, _range_add(args[0].range, args[1].range), args[0].types & args[1].types)
        if fn == "sub" and len(args) == 2:
            if _same_expr(node.args[0], node.args[1]):
                return FieldSpec(args[0].units, ValueRange(0.0, 0.0), args[0].types)
            if _is_literal(node.args[1], 0.0):
                return args[0]
            if _is_literal(node.args[0], 0.0):
                return _scale_field(args[1], -1.0)
            units = args[0].units.compatible_or_unknown(args[1].units)
            output_range = _range_sub(args[0].range, args[1].range)
            if _is_literal(node.args[1], 1.0) and "ratio" in args[0].types and not units.is_unknown():
                return FieldSpec(units, output_range, frozenset({"return"}))
            return FieldSpec(units, output_range, args[0].types & args[1].types)
        if fn == "fillna" and len(args) == 2:
            if _same_expr(node.args[0], node.args[1]):
                return args[0]
            units = args[0].units.compatible_or_unknown(args[1].units)
            return FieldSpec(units, _range_union(args[0].range, args[1].range), args[0].types & args[1].types)
        if fn == "mul" and len(args) == 2:
            if _is_literal(node.args[0], 0.0):
                return FieldSpec(args[1].units, ValueRange(0.0, 0.0), args[1].types)
            if _is_literal(node.args[1], 0.0):
                return FieldSpec(args[0].units, ValueRange(0.0, 0.0), args[0].types)
            if _is_literal(node.args[0], 1.0):
                return args[1]
            if _is_literal(node.args[1], 1.0):
                return args[0]
            if _same_expr(node.args[0], node.args[1]):
                return FieldSpec(args[0].units ** 2.0, _range_pow(args[0].range, 2.0), frozenset())
            return FieldSpec(args[0].units * args[1].units, _range_mul(args[0].range, args[1].range), frozenset())
        if fn in {"div", "floordiv"} and len(args) == 2:
            if _same_expr(node.args[0], node.args[1]):
                return FieldSpec(UnitInfo.dimensionless(), ValueRange(1.0, 1.0), frozenset())
            units = args[0].units / args[1].units
            output_types = (
                frozenset({"ratio"})
                if not units.is_unknown() and units == UnitInfo.dimensionless() and args[0].units == args[1].units and bool(args[0].types | args[1].types)
                else frozenset()
            )
            if _is_literal(node.args[1], 1.0):
                return args[0]
            if _is_literal(node.args[0], 0.0):
                return FieldSpec(units, ValueRange(0.0, 0.0), output_types)
            return FieldSpec(units, _range_div(args[0].range, args[1].range), output_types)
        if fn == "pow" and len(args) == 2:
            exponent = _literal_number(node.args[1])
            if exponent is not None:
                if isclose(exponent, 0.0, rel_tol=0.0, abs_tol=1e-12):
                    return FieldSpec(UnitInfo.dimensionless(), ValueRange(1.0, 1.0), frozenset())
                if isclose(exponent, 1.0, rel_tol=0.0, abs_tol=1e-12):
                    return args[0]
            if _is_literal(node.args[0], 1.0):
                return FieldSpec(UnitInfo.dimensionless(), ValueRange(1.0, 1.0), frozenset())
            units = UnitInfo.dimensionless() if exponent is None else args[0].units ** exponent
            rng = ValueRange.unknown() if exponent is None else _range_pow(args[0].range, exponent)
            return FieldSpec(units, rng, args[0].types if exponent == 1.0 else frozenset())
        if fn == "abs" and args:
            if args[0].range.lower >= 0.0:
                return args[0]
            return FieldSpec(args[0].units, _range_abs(args[0].range), args[0].types)
        if fn in {"ffill", "ewm", "shift", "cumsum", "purify"} and args:
            return FieldSpec(args[0].units, args[0].range, args[0].types, args[0].width)
        if fn == "mean" and args:
            return FieldSpec(args[0].units, args[0].range, args[0].types, args[0].width)
        if fn == "where" and len(args) == 3:
            condition = _literal_number(node.args[0])
            if condition is not None:
                return args[1] if condition != 0.0 else args[2]
            if _same_expr(node.args[1], node.args[2]):
                return args[1]
            if _is_nan_literal(node.args[1]):
                return args[2]
            if _is_nan_literal(node.args[2]):
                return args[1]
            units = args[1].units.compatible_or_unknown(args[2].units)
            return FieldSpec(units, _range_union(args[1].range, args[2].range), args[1].types & args[2].types, args[1].width if args[1].width == args[2].width else None)
        if fn == "einsum":
            subscripts = next((arg.value for arg in node.args if isinstance(arg, String)), None)
            value_args = [arg for expr, arg in zip(node.args, args) if not isinstance(expr, String)]
            if subscripts == "nf,nf->n" and len(value_args) == 2:
                product_range = _range_mul(value_args[0].range, value_args[1].range)
                width = value_args[0].width if value_args[0].width == value_args[1].width else None
                if width is not None:
                    product_range = _range_mul(product_range, ValueRange(float(width), float(width)))
                units = value_args[0].units * value_args[1].units
                output_types = frozenset({"return"}) if "return" in (value_args[0].types | value_args[1].types) and units == UnitInfo.dimensionless() else frozenset()
                return FieldSpec(units, product_range, output_types)
        if fn == "eq" and len(args) == 2 and _same_expr(node.args[0], node.args[1]):
            return _constant_field(1.0, {"boolean"})
        if fn in {"ne", "lt", "gt"} and len(args) == 2 and _same_expr(node.args[0], node.args[1]):
            return _constant_field(0.0, {"boolean"})
        if fn in {"and", "and_"} and len(args) == 2:
            if _is_literal(node.args[0], 0.0) or _is_literal(node.args[1], 0.0):
                return _constant_field(0.0, {"boolean"})
            if _is_literal(node.args[0], 1.0):
                return _truthy_field(args[1])
            if _is_literal(node.args[1], 1.0):
                return _truthy_field(args[0])
        if fn in {"or", "or_"} and len(args) == 2:
            if _is_literal(node.args[0], 1.0) or _is_literal(node.args[1], 1.0):
                return _constant_field(1.0, {"boolean"})
            if _is_literal(node.args[0], 0.0):
                return _truthy_field(args[1])
            if _is_literal(node.args[1], 0.0):
                return _truthy_field(args[0])
        if fn == "xor" and len(args) == 2:
            if _same_expr(node.args[0], node.args[1]):
                return _constant_field(0.0, {"boolean"})
            if _is_literal(node.args[0], 0.0):
                return _truthy_field(args[1])
            if _is_literal(node.args[1], 0.0):
                return _truthy_field(args[0])
        if fn == "isnan" and len(args) == 1 and isinstance(node.args[0], Number):
            return _constant_field(0.0, {"boolean"})
        if fn in {"eq", "ne", "lt", "gt", "and", "and_", "or", "or_", "xor", "isnan"}:
            traced = _auto_trace_field(node, args)
            if traced is not None:
                return traced
            return FieldSpec(UnitInfo.dimensionless(), ValueRange.boolean(), frozenset({"boolean"}))
        if fn in {"ln", "exp", "sign", "arctan", "fraction", "xs_rank", "xs_sort", "xstd", "bspline", "rbf_basis", "future_rbf_basis_sum"}:
            traced = _auto_trace_field(node, args)
            if traced is not None:
                return traced
            width = int(_literal_number(node.args[3])) if fn in {"rbf_basis", "future_rbf_basis_sum"} and len(node.args) >= 4 and _literal_number(node.args[3]) is not None else 1
            return FieldSpec(UnitInfo.dimensionless(), _known_dimensionless_range(fn, args), frozenset(), width)
        if fn == "mod" and len(args) == 2:
            if _same_expr(node.args[0], node.args[1]) or _is_literal(node.args[0], 0.0) or _is_literal(node.args[1], 1.0):
                return FieldSpec(args[0].units, ValueRange(0.0, 0.0), args[0].types)
            return FieldSpec(args[0].units, ValueRange.unknown(), args[0].types)
        if fn in {"ceil", "floor", "round"} and args:
            traced = _auto_trace_field(node, args)
            if traced is not None:
                return traced
            return FieldSpec(args[0].units, ValueRange.unknown(), args[0].types)
        if fn == "cat" and args:
            units = args[0].units
            width = 0
            for arg in args:
                if arg.width is None:
                    width = None
                    break
                width += int(arg.width)
            for arg in args[1:]:
                units = units.compatible_or_unknown(arg.units)
            output_types = args[0].types if all(arg.types == args[0].types for arg in args) else frozenset()
            return FieldSpec(units, _range_union(*(arg.range for arg in args)), output_types, width)
        if fn == "col" and args:
            return args[0]
        if fn == "buffer" and args:
            return args[0]
        if fn == "groupby" and len(args) == 3:
            return args[2]
        traced = _auto_trace_field(node, args)
        if traced is not None:
            return traced
        return FieldSpec()

    result = analyze(expr)
    return FormulaMetadata(result.units, result.range, result.types, cfg.fields, cfg.type_graph, tuple(node_metadata))


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
