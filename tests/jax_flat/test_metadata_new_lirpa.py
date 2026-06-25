import pytest

from trading_dsl_engine.base.metadata_new import analyze_jax_range, range_field


@pytest.mark.parametrize("method", ["IBP", "CROWN"])
def test_metadata_new_exposes_get_range_for_jax_function(method):
    def formula(x):
        return x * 2.0 + 1.0

    meta = analyze_jax_range(formula, [range_field((1, 2), -1.0, 2.0)], method=method)

    lower, upper = meta.get_range().as_tuple()
    assert lower == pytest.approx(-1.0, abs=1e-5)
    assert upper == pytest.approx(5.0, abs=1e-5)
