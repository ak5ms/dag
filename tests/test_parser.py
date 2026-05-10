import pytest

from trading_dsl_engine.parser import parse_formula, Call, Identifier, Number, FormulaParseError


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


def test_parse_error_keyword_args():
    with pytest.raises(FormulaParseError):
        parse_formula("ewm(close, span=21)")


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
