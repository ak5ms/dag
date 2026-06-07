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
