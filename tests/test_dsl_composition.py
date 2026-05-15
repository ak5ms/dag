import numpy as np

from trading_dsl_engine import DSLFunctionRegistry, build_engine, compile_formula, register_dsl_function, run_batch_from_mapping
from trading_dsl_engine.dsl import add, div, ewm, ratio, xs_rank


@register_dsl_function("hlc3")
def hlc3(high, low, close):
    return div(add(add(high, low), close), 3.0)


@register_dsl_function("alpha_ratio_rank")
def alpha_ratio_rank(close, open_):
    return xs_rank(ewm(ratio(close, open_), 5.0))


def test_composite_dsl_function_hlc3():
    eng = build_engine("ewm(hlc3(high, low, close), 3)")
    high = np.array([[3.0, 6.0], [4.0, 8.0]])
    low = np.array([[1.0, 2.0], [2.0, 4.0]])
    close = np.array([[2.0, 4.0], [3.0, 6.0]])

    out = run_batch_from_mapping(eng, {"high": high, "low": low, "close": close})
    hlc3_np = (high + low + close) / 3.0
    alpha = 2.0 / (3.0 + 1.0)
    expected = np.empty((2, 2), dtype=np.float64)
    expected[0] = hlc3_np[0]
    expected[1] = alpha * hlc3_np[1] + (1 - alpha) * hlc3_np[0]
    np.testing.assert_allclose(out, expected)


def test_composite_dsl_function_alpha_ratio_rank_matches_builtin_formula():
    f1 = build_engine("alpha_ratio_rank(close, open)")
    f2 = build_engine("xs_rank(ewm(div(close, open), 5))")

    close = np.array([[10.0, 20.0, 25.0], [12.0, 18.0, 30.0]], dtype=np.float64)
    open_ = np.array([[5.0, 10.0, 12.5], [6.0, 9.0, 15.0]], dtype=np.float64)

    y1 = run_batch_from_mapping(f1, {"close": close, "open": open_})
    y2 = run_batch_from_mapping(f2, {"close": close, "open": open_})
    np.testing.assert_allclose(y1, y2)


def test_registry_namespace_isolation():
    reg = DSLFunctionRegistry()

    @register_dsl_function("twice", registry=reg)
    def twice(x):
        return add(x, x)

    eng = build_engine("twice(close)", dsl_registry=reg)
    close = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    out = run_batch_from_mapping(eng, {"close": close})
    np.testing.assert_allclose(out, close * 2.0)


def test_dsl_expansion_is_cached_for_repeated_calls():
    reg = DSLFunctionRegistry()
    call_count = {"n": 0}

    @register_dsl_function("counted", registry=reg)
    def counted(x):
        call_count["n"] += 1
        return add(x, 1.0)

    compiled = compile_formula("add(counted(close), counted(close))", dsl_registry=reg)
    assert compiled.stats.cache_hits > 0
    assert call_count["n"] == 1


def test_python_expr_infix_formula_matches_prefix_string():
    from trading_dsl_engine import var

    close = var("close")
    open_ = var("open")
    volume = var("volume")
    formula = xs_rank(ewm(((close + open_) * 2.0 % 5.0) | (volume != 0.0), 3.0))
    infix_engine = build_engine(formula)
    prefix_engine = build_engine("xs_rank(ewm(or_(mod(mul(add(close, open), 2), 5), ne(volume, 0)), 3))")

    close_np = np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]], dtype=np.float64)
    open_np = np.array([[0.5, 1.0, 1.5], [0.7, 1.2, 1.7]], dtype=np.float64)
    volume_np = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]], dtype=np.float64)
    data = {"close": close_np, "open": open_np, "volume": volume_np}
    np.testing.assert_allclose(
        run_batch_from_mapping(infix_engine, data, out_path=None),
        run_batch_from_mapping(prefix_engine, data, out_path=None),
    )


def test_python_expr_extended_math_magic_methods_match_prefix_string():
    from trading_dsl_engine import build_engine, run_batch_from_mapping, var

    close = var("close")
    formula = (10.0 // close) + (2.0**close) + (3.0 > close) + (close < 2.0)
    prefix_engine = build_engine("add(add(add(floordiv(10, close), pow(2, close)), gt(3, close)), lt(close, 2))")
    infix_engine = build_engine(formula)
    data = {"close": np.array([[1.0, 2.0, 4.0]], dtype=np.float64)}
    np.testing.assert_allclose(
        run_batch_from_mapping(infix_engine, data, out_path=None),
        run_batch_from_mapping(prefix_engine, data, out_path=None),
    )


def test_all_builtin_dsl_operator_helpers_are_importable():
    import trading_dsl_engine.dsl as dsl

    for name in (
        "add",
        "sub",
        "mul",
        "div",
        "mod",
        "eq",
        "ne",
        "and_",
        "or_",
        "xor",
        "where",
        "abs",
        "isnan",
        "fillna",
        "ln",
        "cumsum",
        "shift",
        "ewm",
        "xs_rank",
        "outer",
        "bspline",
        "col",
        "groupby",
        "grouped",
        "rolling_quantile",
        "mean",
        "univ",
        "Ridge",
        "get_beta",
        "get_preds",
    ):
        assert callable(getattr(dsl, name))
