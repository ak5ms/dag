import pytest

from trading_dsl_engine.base.parser import (
    Call,
    FormulaParseError,
    Identifier,
    Number,
    parse_formula,
)


def test_parse_nested_call():
    expr = parse_formula("xs_rank(ewm(div(close, open), 21))")
    assert isinstance(expr, Call)
    assert expr.fn == "xs_rank"
    inner = expr.args[0]
    assert isinstance(inner, Call)
    assert inner.fn == "ewm"


def test_parse_number_and_identifier():
    expr = parse_formula("div(close, 2.5)")
    assert isinstance(expr, Call)
    assert isinstance(expr.args[0], Identifier)
    assert isinstance(expr.args[1], Number)


def test_parse_preserves_integer_and_float_literal_types():
    integer = parse_formula("7")
    floating = parse_formula("7.0")
    negative_integer = parse_formula("-7")
    negative_floating = parse_formula("-7.0")

    assert isinstance(integer, Number)
    assert isinstance(floating, Number)
    assert isinstance(negative_integer, Number)
    assert isinstance(negative_floating, Number)
    assert type(integer.value) is int
    assert type(floating.value) is float
    assert type(negative_integer.value) is int
    assert type(negative_floating.value) is float


def test_parse_keyword_args():
    expr = parse_formula("groupby((key,), x, cumsum(self_), capacity=21)")
    assert isinstance(expr, Call)
    assert expr.fn == "groupby"
    assert len(expr.kwargs) == 1
    name, value = expr.kwargs[0]
    assert name == "capacity"
    assert isinstance(value, Number)
    assert type(value.value) is int
    assert value.value == 21


def test_parse_multiline_formula():
    expr = parse_formula("""
xs_rank(
    ewm(
        div(close, open),
        21
    )
)
""")
    assert isinstance(expr, Call)
    assert expr.fn == "xs_rank"


def test_parse_infix_operators():
    expr = parse_formula("(close + open) * 2 % 3 | (volume != 0)")
    assert isinstance(expr, Call)
    assert expr.fn == "or_"
    left = expr.args[0]
    assert isinstance(left, Call)
    assert left.fn == "mod"
    assert isinstance(left.args[0], Call)
    assert left.args[0].fn == "mul"
    right = expr.args[1]
    assert isinstance(right, Call)
    assert right.fn == "ne"
