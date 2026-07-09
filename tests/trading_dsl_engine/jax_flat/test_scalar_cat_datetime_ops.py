from datetime import datetime, timezone

import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import DSLFunctionRegistry, cat, ceil, dayofyear, floor, hour, register_dsl_function, round, shift, timeofday, to_dt, var
from trading_dsl_engine.base.parser import Expr
from trading_dsl_engine.jax_flat.engine import compile_formula


def _epoch_us(value: datetime) -> int:
    return int(value.replace(tzinfo=timezone.utc).timestamp() * 1_000_000)


def test_cumsum_broadcasts_scalar_literal_to_instrument_array_in_batch():
    runtime = compile_formula("cumsum(1)")
    dummy = jnp.zeros((4, 3), dtype=jnp.float64)

    _, out = runtime.run_batch((dummy,))

    expected = np.array(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
        ]
    )
    np.testing.assert_allclose(np.asarray(out), expected)


def test_cat_stacks_multiple_alpha_vectors_on_last_axis():
    runtime = compile_formula("cat(alpha1, alpha2, alpha3)")
    alpha1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    alpha2 = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    alpha3 = jnp.array([[100.0, 200.0], [300.0, 400.0]])

    _, out = runtime.run_batch((alpha1, alpha2, alpha3))

    expected = np.stack([np.asarray(alpha1), np.asarray(alpha2), np.asarray(alpha3)], axis=-1)
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(np.asarray(out), expected)


def test_compiled_dag_is_levelized_across_independent_cat_branches(monkeypatch):
    formula = "cat(add(x, 1), sub(y, 2))"
    monkeypatch.delenv("TRADING_DSL_ENGINE_DISABLE_LEVELIZED_DAG", raising=False)
    runtime = compile_formula(formula, cpp=False)
    node_names = [type(node.op).__name__ for node in runtime.program.nodes]
    child_ids = [node.child_ids for node in runtime.program.nodes]

    assert node_names == ["InputOp", "LiteralOp", "InputOp", "LiteralOp", "NaryOp", "NaryOp", "NaryOp"]
    assert child_ids == [(), (), (), (), (0, 1), (2, 3), (4, 5)]

    monkeypatch.setenv("TRADING_DSL_ENGINE_DISABLE_LEVELIZED_DAG", "1")
    depth_first_runtime = compile_formula(formula, cpp=False)
    depth_first_child_ids = [node.child_ids for node in depth_first_runtime.program.nodes]
    assert depth_first_child_ids == [(), (), (0, 1), (), (), (3, 4), (2, 5)]


def test_dsl_function_overloads_fall_back_to_builtin_ops_by_signature():
    x = var("x")
    assert shift(x) == shift(x, 1.0, 1.0)
    assert shift(x, 2.0) == shift(x, 2.0, 2.0)
    assert floor(x).fn == "floor"
    assert len(floor(x).args) == 1


def test_dsl_function_overload_annotation_conflicts_raise_loudly():
    reg = DSLFunctionRegistry()

    @register_dsl_function("custom", registry=reg)
    def custom_expr(x: Expr) -> Expr:
        return x

    try:
        @register_dsl_function("custom", registry=reg)
        def custom_conflict(x: str) -> Expr:
            return var(x)
    except TypeError as exc:
        assert "Conflicting annotation" in str(exc)
    else:
        raise AssertionError("conflicting overload annotation did not raise")


def test_datetime_calendar_and_rounding_ops_from_microsecond_timestamps():
    ev_dt = to_dt(var("ev_ts"), unit="us")
    runtime = compile_formula(
        cat(
            dayofyear(ev_dt),
            timeofday(ev_dt),
            hour(ev_dt),
            floor(ev_dt, freq="H"),
            ceil(ev_dt, freq="H"),
            round(ev_dt, freq="H"),
        )
    )
    ev_ts = jnp.array(
        [
            [_epoch_us(datetime(2024, 2, 29, 23, 58, 30)), _epoch_us(datetime(2023, 1, 1, 0, 29, 30))],
            [_epoch_us(datetime(2024, 3, 1, 0, 2, 0)), _epoch_us(datetime(2023, 12, 31, 23, 31, 0))],
        ],
        dtype=jnp.float64,
    )

    _, out = runtime.run_batch((ev_ts,))

    hour_us = 3_600_000_000.0
    ev_microseconds = np.asarray(ev_ts)
    expected = np.stack(
        [
            np.array([[60.0, 1.0], [61.0, 365.0]]),
            np.array([[86_310_000_000.0, 1_770_000_000.0], [120_000_000.0, 84_660_000_000.0]]),
            np.array([[23.0, 0.0], [0.0, 23.0]]),
            np.floor(ev_microseconds / hour_us) * hour_us,
            np.ceil(ev_microseconds / hour_us) * hour_us,
            np.floor(ev_microseconds / hour_us + 0.5) * hour_us,
        ],
        axis=-1,
    )
    assert out.shape == (2, 2, 6)
    np.testing.assert_allclose(np.asarray(out), expected)
