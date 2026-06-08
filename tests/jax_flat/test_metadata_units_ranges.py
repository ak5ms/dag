from math import inf

import pytest

from trading_dsl_engine.jax_flat import ValueRange, compile_formula, field, metadata
from trading_dsl_engine.base.metadata import MetadataError


def test_runtime_exposes_unit_exponents_and_range_for_formula():
    schema = metadata(
        {
            "close": field(units={"dollar": 1}, range="real", types=("price",)),
            "volume": field(units={"shares": 1}, range="nonnegative", types=("volume",)),
        },
        type_relations=(("price", "currency"),),
    )

    runtime = compile_formula("close * volume ** 3", metadata=schema, cpp=False)

    assert runtime.get_units().as_dict() == {"dollar": 1.0, "shares": 3.0}
    assert runtime.get_range().as_tuple() == (-inf, inf)
    assert runtime.program.metadata.input_fields["close"].types == frozenset({"price"})
    assert runtime.get_type_relations().closure(("price",)) == frozenset({"price", "currency"})
    graph = runtime.get_type_relations()
    price_idx = graph.types.index("price")
    currency_idx = graph.types.index("currency")
    assert graph.as_matrix()[price_idx][currency_idx]


def test_runtime_range_uses_interval_arithmetic_through_operators():
    runtime = compile_formula(
        "abs(ret - 0.5) / 2",
        metadata={"ret": field(range=(0, 1), types=("return",))},
        cpp=False,
    )

    assert runtime.get_units().as_dict() == {}
    assert runtime.get_range() == ValueRange(0.0, 0.25)


def test_same_variable_division_reduces_to_unit_range():
    runtime = compile_formula(
        "close / close",
        metadata={"close": field(units={"dollar": 1}, range="real", types=("price",))},
        cpp=False,
    )

    assert runtime.get_units().as_dict() == {}
    assert runtime.get_range() == ValueRange(1.0, 1.0)


def test_algebraic_reductions_cover_arithmetic_identities():
    schema = {"close": field(units={"dollar": 1}, range=(10, 20), types=("price",))}

    identity_cases = [
        ("close + 0", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("0 + close", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("close - 0", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("0 - close", {"dollar": 1.0}, ValueRange(-20.0, -10.0)),
        ("close - close", {"dollar": 1.0}, ValueRange(0.0, 0.0)),
        ("close * 1", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("1 * close", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("close * 0", {"dollar": 1.0}, ValueRange(0.0, 0.0)),
        ("close / 1", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("close ** 1", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("close ** 0", {}, ValueRange(1.0, 1.0)),
        ("close * close", {"dollar": 2.0}, ValueRange(100.0, 400.0)),
        ("1 ** close", {}, ValueRange(1.0, 1.0)),
        ("close % close", {"dollar": 1.0}, ValueRange(0.0, 0.0)),
        ("0 % close", {}, ValueRange(0.0, 0.0)),
        ("abs(close)", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
        ("fillna(close, close)", {"dollar": 1.0}, ValueRange(10.0, 20.0)),
    ]
    for formula, units, expected_range in identity_cases:
        runtime = compile_formula(formula, metadata=schema, cpp=False)
        assert runtime.get_units().as_dict() == units, formula
        assert runtime.get_range() == expected_range, formula


def test_algebraic_reductions_cover_conditionals_and_booleans():
    schema = {"signal": field(range=(0, 1), types=("boolean",))}

    cases = [
        ("where(1, signal, 0)", ValueRange(0.0, 1.0)),
        ("where(0, 1, signal)", ValueRange(0.0, 1.0)),
        ("where(signal, 7, 7)", ValueRange(7.0, 7.0)),
        ("signal == signal", ValueRange(1.0, 1.0)),
        ("signal != signal", ValueRange(0.0, 0.0)),
        ("signal < signal", ValueRange(0.0, 0.0)),
        ("signal ^ signal", ValueRange(0.0, 0.0)),
        ("signal & 1", ValueRange(0.0, 1.0)),
        ("signal | 0", ValueRange(0.0, 1.0)),
        ("signal & 0", ValueRange(0.0, 0.0)),
        ("signal | 1", ValueRange(1.0, 1.0)),
        ("isnan(1)", ValueRange(0.0, 0.0)),
    ]
    for formula, expected_range in cases:
        runtime = compile_formula(formula, metadata=schema, cpp=False)
        assert runtime.get_units().as_dict() == {}, formula
        assert runtime.get_range() == expected_range, formula


def test_metadata_auto_traces_nary_ops_for_ranges_and_types():
    schema = {
        "x": field(range=(-1.2, 2.7)),
        "y": field(range=(-2, 3)),
    }

    cases = [
        ("ceil(x)", ValueRange(-1.0, 3.0), frozenset()),
        ("floor(x)", ValueRange(-2.0, 2.0), frozenset()),
        ("round(x)", ValueRange(-1.0, 3.0), frozenset()),
        ("fraction(x)", ValueRange(0.0, 0.8), frozenset()),
        ("sign(x)", ValueRange(-1.0, 1.0), frozenset()),
        ("arctan(x)", ValueRange(-0.8760580505981934, 1.2160906747839564), frozenset()),
        ("x > y", ValueRange(0.0, 1.0), frozenset({"boolean"})),
    ]
    for formula, expected_range, expected_types in cases:
        runtime = compile_formula(formula, metadata=schema, cpp=False)
        assert runtime.get_range() == expected_range, formula
        assert runtime.get_types() == expected_types, formula


def test_unit_incompatible_addition_fails_at_compile_time():
    with pytest.raises(MetadataError, match="add requires compatible units"):
        compile_formula(
            "close + volume",
            metadata={
                "close": field(units={"dollar": 1}, range="real"),
                "volume": field(units={"shares": 1}, range="nonnegative"),
            },
            cpp=False,
        )
