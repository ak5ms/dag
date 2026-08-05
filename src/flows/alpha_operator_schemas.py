from __future__ import annotations

from trading_dsl_engine.base.dsl import (
    and_, ceil, eq, floor, floordiv, isnan, mod, ne, norm_inv, or_, pow,
    ratio, xor,
)

from flows.alpha_mcts import OperatorSchema, default_operator_schemas


# Operators that are meaningful in an unconstrained alpha expression tree and
# can be described by the generic schema families.  The base list contains all
# temporal, cross-sectional, comparison, conditional, and common math ops.
def all_operator_schemas() -> tuple[OperatorSchema, ...]:
    extra = (
        OperatorSchema("ceil", ceil, "unary_same", 1, 0.4),
        OperatorSchema("floor", floor, "unary_same", 1, 0.4),
        OperatorSchema("norm_inv", norm_inv, "unary_dimensionless", 1, 0.5),
        OperatorSchema("isnan", isnan, "comparison", 1, 0.4),
        OperatorSchema("pow", pow, "binary_numeric", 2, 0.5),
        OperatorSchema("mod", mod, "binary_numeric", 2, 0.3),
        OperatorSchema("floordiv", floordiv, "binary_numeric", 2, 0.2),
        OperatorSchema("ratio", ratio, "binary_numeric", 2, 0.8),
        OperatorSchema("eq", eq, "comparison", 2, 0.3),
        OperatorSchema("ne", ne, "comparison", 2, 0.3),
        OperatorSchema("and", and_, "binary_same", 2, 0.2),
        OperatorSchema("or", or_, "binary_same", 2, 0.2),
        OperatorSchema("xor", xor, "binary_same", 2, 0.2),
    )
    schemas = default_operator_schemas() + extra
    seen: set[str] = set()
    out: list[OperatorSchema] = []
    for schema in schemas:
        if schema.name not in seen:
            out.append(schema)
            seen.add(schema.name)
    return tuple(out)


# These are not omitted accidentally.  They need additional structured holes
# rather than ordinary expression children and therefore should be searched by
# a dedicated schema plugin before being enabled.
STRUCTURAL_OPERATOR_REQUIREMENTS = {
    "Ridge": "variadic feature list and object-valued result",
    "InstrumentBasisMean": "object-valued result",
    "einsum": "subscript grammar and dependent tensor shape",
    "cat": "variadic feature list and output-axis shape",
    "groupby": "key expressions, key hints, and self_ scope",
    "grouped": "group scope",
    "univ": "static universe groups",
    "col": "compile-time column index",
    "buffer": "buffer object result",
    "bspline": "basis configuration",
    "rbf_basis": "basis configuration",
    "future_rbf_basis_sum": "basis configuration",
    "outer": "dependent matrix output shape",
    "to_dt": "datetime object semantics",
    "year": "datetime input",
    "month": "datetime input",
    "day": "datetime input",
    "dayofweek": "datetime input",
    "dayofyear": "datetime input",
    "hour": "datetime input",
    "minute": "datetime input",
    "second": "datetime input",
    "timeofday": "datetime input",
}


def operator_inventory_report() -> dict[str, object]:
    schemas = all_operator_schemas()
    return {
        "searchable": tuple(schema.name for schema in schemas),
        "structural": dict(STRUCTURAL_OPERATOR_REQUIREMENTS),
        "searchable_count": len(schemas),
        "structural_count": len(STRUCTURAL_OPERATOR_REQUIREMENTS),
    }


__all__ = [
    "STRUCTURAL_OPERATOR_REQUIREMENTS", "all_operator_schemas",
    "operator_inventory_report",
]
