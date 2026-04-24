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
