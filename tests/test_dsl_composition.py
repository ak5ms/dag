import numpy as np

from trading_dsl_engine import DSLFunctionRegistry, build_engine, register_dsl_function, run_batch_from_mapping
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
    expected = np.empty((2, 2, 1), dtype=np.float64)
    expected[0, :, 0] = hlc3_np[0]
    expected[1, :, 0] = alpha * hlc3_np[1] + (1 - alpha) * hlc3_np[0]
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
    np.testing.assert_allclose(out[:, :, 0], close * 2.0)
