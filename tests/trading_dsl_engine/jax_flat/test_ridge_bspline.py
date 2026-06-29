import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import _bspline

RUN_PERF = os.getenv("RUN_PERF_TESTS", "0") == "1"
T_ROWS = 1440 * 365 * 3
N_INSTRUMENTS = 9


def _run(formula, *arrays):
    return compile_formula(formula).run_batch(tuple(jnp.asarray(a, dtype=jnp.float64) for a in arrays))[1]


def _reference_ridge_one_feature(x, y, w, hl, ridge):
    t_rows, n_inst = x.shape
    xx = 0.0
    xy = 0.0
    has_xx = False
    has_xy = False
    last_xx = 0
    last_xy = 0
    beta = 0.0
    betas = np.zeros((t_rows, 1), dtype=np.float64)
    preds = np.full((t_rows, n_inst), np.nan, dtype=np.float64)
    alpha = 1.0 - np.exp(np.log(0.5) / hl)
    for t in range(t_rows):
        preds[t] = np.where(np.isfinite(x[t]) & np.isfinite(y[t]), x[t] * beta, np.nan)
        valid_xx = np.isfinite(x[t]) & np.isfinite(w[t])
        valid_xy = np.isfinite(x[t]) & np.isfinite(y[t]) & np.isfinite(w[t])
        if valid_xx.any():
            snap_xx = np.sum(x[t, valid_xx] * x[t, valid_xx] * w[t, valid_xx])
            if has_xx:
                a = alpha ** (t - last_xx)
                xx = xx * (1.0 - a) + snap_xx * a
            else:
                xx = snap_xx
                has_xx = True
            last_xx = t
        if valid_xy.any():
            snap_xy = np.sum(x[t, valid_xy] * y[t, valid_xy] * w[t, valid_xy])
            if has_xy:
                a = alpha ** (t - last_xy)
                xy = xy * (1.0 - a) + snap_xy * a
            else:
                xy = snap_xy
                has_xy = True
            last_xy = t
        denom = xx * (1.0 + ridge)
        if denom != 0.0 and np.isfinite(denom) and np.isfinite(xy):
            beta = xy / denom
        betas[t, 0] = beta
    return betas, preds


def test_bspline_nary_op_emits_matrix_and_preserves_nan_rows():
    x = np.array([[0.0, 0.5, np.nan], [1.0, -0.25, 1.25]], dtype=np.float64)
    out = np.asarray(_run("bspline(x, 4)", x))

    expected = np.asarray(jax.vmap(lambda row: _bspline(row, 4))(jnp.asarray(x)))
    assert out.shape == (2, 3, 4)
    np.testing.assert_allclose(out, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.nansum(out[:, [0, 1], :], axis=2), np.ones((2, 2)), rtol=1e-12, atol=1e-12)
    assert np.isnan(out[0, 2]).all()


def test_bspline_matrix_can_feed_col_projection():
    x = np.array([[0.1, 0.9], [np.nan, 0.5]], dtype=np.float64)
    basis = np.asarray(_run("bspline(x, 5)", x))
    col = np.asarray(_run("col(bspline(x, 5), 3)", x))

    assert basis.shape == (2, 2, 5)
    assert col.shape == (2, 2)
    np.testing.assert_allclose(col, basis[:, :, 3], rtol=1e-12, atol=1e-12, equal_nan=True)


def test_ridge_object_root_scan_batch_matches_projected_outputs_and_tick_scan():
    x = jnp.asarray(
        [[1.0, 2.0, jnp.nan], [2.0, 3.0, 4.0], [3.0, jnp.nan, 5.0], [4.0, 5.0, 6.0]],
        dtype=jnp.float64,
    )
    y = jnp.asarray(
        [[2.0, 4.0, 6.0], [4.0, 6.0, 8.0], [6.0, jnp.nan, 10.0], [8.0, 10.0, 12.0]],
        dtype=jnp.float64,
    )
    runtime = compile_formula("Ridge(x, y, 2, 0.1)")
    state0 = runtime.init_state(x.shape[1])

    def tick_step(carry, rows):
        return runtime._tick_impl(carry, *rows)

    _, tick_out = jax.lax.scan(tick_step, state0, (x, y))
    _, batch_out = runtime.run_batch((x, y))

    projected_beta = _run("get_beta(Ridge(x, y, 2, 0.1))", x, y)
    projected_preds = _run("get_preds(Ridge(x, y, 2, 0.1))", x, y)

    np.testing.assert_allclose(np.asarray(batch_out.beta), np.asarray(tick_out.beta), rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.asarray(batch_out.preds), np.asarray(tick_out.preds), rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.asarray(batch_out.beta), np.asarray(projected_beta), rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.asarray(batch_out.preds), np.asarray(projected_preds), rtol=1e-12, atol=1e-12, equal_nan=True)


def test_ridge_one_feature_matches_pairwise_nan_reference_and_preds_use_prior_beta():
    x = np.array(
        [[1.0, 2.0, np.nan], [2.0, np.nan, 4.0], [np.nan, 4.0, 5.0], [4.0, 5.0, 6.0]],
        dtype=np.float64,
    )
    y = np.array(
        [[3.0, 5.0, 7.0], [5.0, np.nan, 9.0], [7.0, 9.0, np.nan], [9.0, 11.0, 13.0]],
        dtype=np.float64,
    )
    w = np.array(
        [[1.0, 1.0, np.nan], [1.0, 2.0, 1.0], [np.nan, 1.0, 1.0], [1.0, 1.0, 1.0]],
        dtype=np.float64,
    )

    beta = np.asarray(_run("get_beta(Ridge(x, y, w, 2, 0.1))", x, y, w))
    expected_beta, expected_preds = _reference_ridge_one_feature(x, y, w, hl=2.0, ridge=0.1)
    np.testing.assert_allclose(beta, expected_beta, rtol=1e-12, atol=1e-12, equal_nan=True)

    preds = np.asarray(_run("get_preds(Ridge(x, y, w, 2, 0.1))", x, y, w))
    np.testing.assert_allclose(preds, expected_preds, rtol=1e-12, atol=1e-12, equal_nan=True)


def test_ridge_supports_omitted_weights_and_bspline_matrix_features():
    x = np.array([[0.0, 0.25, 0.5], [0.2, np.nan, 0.8], [0.4, 0.6, 1.0], [0.1, 0.3, 0.9]], dtype=np.float64)
    y = np.array([[1.0, 2.0, 3.0], [1.5, np.nan, 3.5], [2.0, 3.0, 4.0], [1.2, 2.2, 4.2]], dtype=np.float64)

    beta = np.asarray(_run("get_beta(Ridge(bspline(x, 4), y, 3, 0.2))", x, y))
    preds = np.asarray(_run("get_preds(Ridge(bspline(x, 4), y, 3, 0.2))", x, y))

    assert beta.shape == (4, 4)
    assert preds.shape == (4, 3)
    assert np.isfinite(beta[-1]).all()
    assert np.isnan(preds[1, 1])


def test_groupby_univ_only_can_emit_ridge_beta_feature_vectors():
    x1 = np.array(
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4]],
        dtype=np.float64,
    )
    x2 = np.array(
        [[1.0, 0.5, 1.5], [1.2, 0.7, 1.7], [1.4, 0.9, 1.9], [1.1, 0.6, 1.6]],
        dtype=np.float64,
    )
    y = 0.5 + 2.0 * x1 - x2
    w = np.ones_like(x1)

    out = np.asarray(
        _run(
            "groupby((univ([0, 1], [2]), ), y, get_beta(Ridge(bspline(x1, 2), x2, y, w, 8, 0.1)))",
            x1,
            x2,
            y,
            w,
        )
    )

    assert out.shape == (4, 3, 3)
    np.testing.assert_allclose(out[:, 0, :], out[:, 1, :], rtol=1e-12, atol=1e-12, equal_nan=True)
    assert np.isfinite(out[-1]).all()


def test_groupby_univ_only_preserves_feature_vector_lhs_width():
    ev_ts = np.ones((5, 4), dtype=np.float64)
    volume = np.arange(1, 21, dtype=np.float64).reshape(5, 4)
    formula = (
        "groupby((univ([0], [1], [2], [3]), ), "
        "get_beta(Ridge(bspline((cumsum(fillna(ev_ts, 1) / fillna(ev_ts, 1) * 60000000) "
        "+ 1451601660010000) % 86400000000 / 86400000000, 3), ln(volume + 1), 1, 21, 0.0)), "
        "self_ + 0)"
    )

    grouped = np.asarray(_run(formula, ev_ts, volume))
    beta = np.asarray(
        _run(
            "get_beta(Ridge(bspline((cumsum(fillna(ev_ts, 1) / fillna(ev_ts, 1) * 60000000) "
            "+ 1451601660010000) % 86400000000 / 86400000000, 3), ln(volume + 1), 1, 21, 0.0))",
            ev_ts,
            volume,
        )
    )

    assert grouped.shape == (5, 4, 3)
    np.testing.assert_allclose(grouped, np.broadcast_to(beta[:, None, :], grouped.shape), rtol=1e-12, atol=1e-12)


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_bspline_ridge_jax_flat_t_rows():
    x = jax.random.uniform(jax.random.PRNGKey(120), (T_ROWS, N_INSTRUMENTS), dtype=jnp.float64)
    y = 0.5 + 2.0 * x + 0.01 * jax.random.normal(jax.random.PRNGKey(121), x.shape, dtype=jnp.float64)
    w = jnp.ones_like(x)
    runtime = compile_formula("get_preds(Ridge(bspline(x, 6), y, w, 8, 0.1))")

    start = time.perf_counter()
    _, out = runtime.run_batch((x, y, w))
    jax.block_until_ready(out)
    elapsed = time.perf_counter() - start

    print(f"bspline_ridge::jax_flat T_ROWS={T_ROWS} elapsed={elapsed:.3f}s")
    assert out.shape == (T_ROWS, N_INSTRUMENTS)


@pytest.mark.skipif(not RUN_PERF, reason="set RUN_PERF_TESTS=1 to enable perf tests")
def test_perf_groupby_univ_only_ridge_beta_jax_flat_t_rows():
    x1 = jax.random.uniform(jax.random.PRNGKey(120), (T_ROWS, N_INSTRUMENTS), dtype=jnp.float64)
    x2 = jax.random.uniform(jax.random.PRNGKey(120), (T_ROWS, N_INSTRUMENTS), dtype=jnp.float64)
    y = 0.5 + 2.0 * x1 - 1 * x2 + 0.01 * jax.random.normal(jax.random.PRNGKey(121), x1.shape, dtype=jnp.float64)
    w = jnp.ones_like(x1)
    runtime = compile_formula(
        "groupby((univ([0, 1], [2]), ), y, get_beta(Ridge(bspline(x1, 2), x2, y, w, 8, 0.1)))"
    )

    start = time.perf_counter()
    _, out = runtime.run_batch((x1, x2, y, w))
    jax.block_until_ready(out)
    elapsed = time.perf_counter() - start

    print(f"groupby_univ_ridge_beta::jax_flat T_ROWS={T_ROWS} elapsed={elapsed:.3f}s")
    assert out.shape == (T_ROWS, N_INSTRUMENTS, 3)
