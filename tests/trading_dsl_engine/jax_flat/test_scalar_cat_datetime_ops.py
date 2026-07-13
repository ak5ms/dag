import os
import subprocess
import sys
from datetime import datetime, timezone

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import ndtri

from trading_dsl_engine.base.dsl import DSLFunctionRegistry, cat, ceil, dayofyear, floor, hour, register_dsl_function, round, shift, timeofday, to_dt, var, xs_rank
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


def test_cumsum_prefix_preserves_nan_and_split_batch_state():
    rng = np.random.default_rng(331)
    values = rng.normal(size=(127, 4))
    values[rng.random(values.shape) < 0.2] = np.nan
    runtime = compile_formula("cumsum(x)", cpp=False)

    one_state, one_shot = runtime.run_batch({"x": jnp.asarray(values)})
    split_state, first = runtime.run_batch({"x": jnp.asarray(values[:53])})
    split_state, second = runtime.run_batch({"x": jnp.asarray(values[53:])}, states=split_state)

    np.testing.assert_allclose(
        np.concatenate((np.asarray(first), np.asarray(second))),
        np.asarray(one_shot),
        rtol=1e-12,
        atol=1e-12,
        equal_nan=True,
    )
    for one_leaf, split_leaf in zip(jax.tree.leaves(one_state), jax.tree.leaves(split_state), strict=True):
        np.testing.assert_allclose(np.asarray(split_leaf), np.asarray(one_leaf), rtol=1e-12, atol=1e-12)


def test_cat_stacks_multiple_alpha_vectors_on_last_axis():
    runtime = compile_formula("cat(alpha1, alpha2, alpha3)")
    alpha1 = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    alpha2 = jnp.array([[10.0, 20.0], [30.0, 40.0]])
    alpha3 = jnp.array([[100.0, 200.0], [300.0, 400.0]])

    _, out = runtime.run_batch((alpha1, alpha2, alpha3))

    expected = np.stack([np.asarray(alpha1), np.asarray(alpha2), np.asarray(alpha3)], axis=-1)
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(np.asarray(out), expected)


def test_xs_rank_shape_selected_kernels_preserve_ties_and_nans():
    rng = np.random.default_rng(884)
    for width in (9, 129):
        values = rng.normal(size=(17, width))
        values[::3, 0] = np.nan
        values[::4, 2] = values[::4, 1]
        _, actual = compile_formula(xs_rank(var("x")), cpp=False).run_batch({"x": jnp.asarray(values)})

        expected = np.full_like(values, np.nan)
        for row_idx, row in enumerate(values):
            valid = np.isfinite(row)
            sorted_valid = np.sort(row[valid])
            right_ranks = np.searchsorted(sorted_valid, row[valid], side="right")
            expected[row_idx, valid] = ndtri(right_ranks / (sorted_valid.size + 1.0))

        np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_stateless_dag_cpu_sharding_preserves_rank_padding():
    script = r"""
import numpy as np
import jax.numpy as jnp
from scipy.special import ndtri
from trading_dsl_engine.base.dsl import var, xs_rank
from trading_dsl_engine.jax_flat.engine import compile_formula

rng = np.random.default_rng(722)
values = rng.normal(size=(35, 129))
values[::3, 0] = np.nan
_, actual = compile_formula(xs_rank(var("x")), cpp=False).run_batch({"x": jnp.asarray(values)})
expected = np.full_like(values, np.nan)
for row_idx, row in enumerate(values):
    valid = np.isfinite(row)
    sorted_valid = np.sort(row[valid])
    right_ranks = np.searchsorted(sorted_valid, row[valid], side="right")
    expected[row_idx, valid] = ndtri(right_ranks / (sorted_valid.size + 1.0))
np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-12, atol=1e-12, equal_nan=True)
"""
    env = os.environ.copy()
    env["JAX_NUM_CPU_DEVICES"] = "4"
    env["PYTHONPATH"] = os.path.abspath("src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([sys.executable, "-c", script], check=True, env=env, timeout=60)


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
