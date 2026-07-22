import jax.numpy as jnp
import numpy as np
import pytest

from trading_dsl_engine.jax_flat import compile_formula, execution_policy
from trading_dsl_engine.jax_flat import engine_legacy


@pytest.mark.parametrize(
    ("formula", "data"),
    [
        (
            "groupby((univ([0, 1]), ts), close, cumsum(cumsum(self_)))",
            {
                "close": jnp.array([[10.0, 20.0], [1.0, 2.0], [20.0, 50.0], [jnp.nan, jnp.nan]]),
                "ts": jnp.array([[1.0, 1.0], [1.0, 2.0], [2.0, 1.0], [2.0, 2.0]]),
            },
        ),
        (
            "close + groupby((univ([0, 1], [2]), open), close, cumsum(self_))",
            {
                "open": jnp.array([[1.0, 1.0, 2.0], [1.0, 2.0, 2.0], [2.0, 1.0, 2.0]]),
                "close": jnp.array([[10.0, 20.0, 30.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            },
        ),
    ],
)
def test_groupby_blockers_use_legacy_execution_without_behavior_change(formula, data):
    planned = compile_formula(formula, cpp=False)
    legacy = engine_legacy.compile_formula(formula, cpp=False)

    assert not execution_policy._eligible(planned.program)
    _, planned_output = planned.run_batch(data)
    _, legacy_output = legacy.run_batch(data)

    np.testing.assert_allclose(
        np.asarray(planned_output),
        np.asarray(legacy_output),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
