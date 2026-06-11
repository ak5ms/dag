from dataclasses import replace

import numpy as np

from trading_dsl_engine.jax_flat import compile_formula
from trading_dsl_engine.jax_flat.engine import _build_state_layout
from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_formula_native
from trading_dsl_engine.jax_flat.ops import RidgeOp


def _run(formula, **data):
    runtime = compile_formula(formula, cpp=False)
    _, out = runtime.run_batch(data)
    return np.asarray(out)


def _with_ridge_statefulness(runtime, *, is_stateful):
    nodes = tuple(
        replace(node, op=replace(node.op, is_stateful=is_stateful)) if isinstance(node.op, RidgeOp) else node
        for node in runtime.program.nodes
    )
    return replace(runtime, program=replace(runtime.program, nodes=nodes, state_layout=_build_state_layout(nodes)))


def _run_with_ridge_statefulness(formula, *, is_stateful, **data):
    runtime = _with_ridge_statefulness(compile_formula(formula, cpp=False), is_stateful=is_stateful)
    _, out = runtime.run_batch(data)
    return np.asarray(out)


def test_cache_clip_norm_and_invert_magic():
    x = np.array([[1.0, 2.0, 4.0], [3.0, np.nan, -1.0]], dtype=np.float64)

    runtime = compile_formula("add(cache(x), 1.0)", cpp=False)
    _, out = runtime.run_batch({"x": x})
    np.testing.assert_allclose(out, x + 1.0, equal_nan=True)
    cached = runtime.get_cached_values()
    assert tuple(cached) == runtime.program.cache_nodes
    np.testing.assert_allclose(next(iter(cached.values())), x, equal_nan=True)

    disk_runtime = compile_formula('cache(x, "disk")', cpp=False)
    _, disk_out = disk_runtime.run_batch({"x": x})
    disk_cached = next(iter(disk_runtime.get_cached_values().values()))
    assert isinstance(disk_cached, np.memmap)
    np.testing.assert_allclose(disk_out, x, equal_nan=True)
    np.testing.assert_allclose(disk_cached, x, equal_nan=True)
    np.testing.assert_allclose(_run("clip(x, 0.0, 2.5)", x=x), np.clip(x, 0.0, 2.5), equal_nan=True)
    np.testing.assert_allclose(_run("xs_norm(x)", x=x)[0], [1.0 / 7.0, 2.0 / 7.0, 4.0 / 7.0])
    np.testing.assert_allclose(_run("~x", x=x), 1.0 - x, equal_nan=True)


def test_xs_rank_uses_n_plus_one_and_normal_scores():
    x = np.array([[1.0, 2.0, 3.0, np.nan]], dtype=np.float64)
    out = _run("xs_rank(x)", x=x)[0]
    expected = _run("norm_inv(cache(r))", r=np.array([[0.25, 0.5, 0.75, np.nan]], dtype=np.float64))[0]
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_hl_zero_ridge_is_stateless_and_projects_current_row_beta():
    x = np.array([[1.0, 2.0, 3.0], [3.0, 1.0, 2.0]], dtype=np.float64)
    y = x + 1.0
    runtime = compile_formula("Ridge(x, y, 0, 0.1)", cpp=False)
    _, batch_out = runtime.run_batch({"x": x, "y": y})
    beta = np.asarray(batch_out.beta)
    preds = np.asarray(batch_out.preds)

    stateful_beta = _run_with_ridge_statefulness("get_beta(Ridge(x, y, 0, 0.1))", is_stateful=True, x=x, y=y)
    np.testing.assert_allclose(beta, stateful_beta, rtol=1e-12, atol=1e-12)

    state = runtime.init_state(x.shape[1])
    assert len(state) == 0
    tick_beta = []
    tick_preds = []
    for x_row, y_row in zip(x, y, strict=True):
        state, tick_out = runtime.tick(state, x_row, y_row)
        tick_beta.append(np.asarray(tick_out.beta))
        tick_preds.append(np.asarray(tick_out.preds))
    np.testing.assert_allclose(np.asarray(tick_beta), beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(tick_preds), preds, rtol=1e-12, atol=1e-12)

    projected_beta = _run("get_beta(Ridge(x, y, 0, 0.1))", x=x, y=y)
    projected_preds = _run("get_preds(Ridge(x, y, 0, 0.1))", x=x, y=y)
    np.testing.assert_allclose(projected_beta, beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(projected_preds, preds, rtol=1e-12, atol=1e-12)

    ridge_op = runtime.program.nodes[runtime.program.outputs[0]].op
    assert not ridge_op.is_stateful


def test_hl_zero_ridge_solves_weighted_design_for_beta_with_nan_consistency():
    x1 = np.array([[1.0, np.nan, 3.0, 4.0]], dtype=np.float64)
    x2 = np.array([[2.0, 4.0, 1.0, np.nan]], dtype=np.float64)
    y = np.array([[5.0, np.nan, 7.0, 11.0]], dtype=np.float64)
    weights = np.array([[1.0, 2.0, 0.5, 1.0]], dtype=np.float64)

    formula = "get_beta(Ridge(cat(x1, x2), y, weights, 0, 0.1))"
    beta = _run(formula, x1=x1, x2=x2, y=y, weights=weights)
    stateful_beta = _run_with_ridge_statefulness(
        formula, is_stateful=True, x1=x1, x2=x2, y=y, weights=weights
    )

    np.testing.assert_allclose(beta, stateful_beta, rtol=1e-12, atol=1e-12)


def test_hl_zero_ridge_nan_cartesian_product_uses_pairwise_moments():
    combos = np.array(
        [
            [has_x, has_w, has_y]
            for has_x in (False, True)
            for has_w in (False, True)
            for has_y in (False, True)
        ],
        dtype=bool,
    )
    finite_x = np.arange(1.0, combos.shape[0] + 1.0, dtype=np.float64)
    finite_w = np.linspace(0.25, 2.0, combos.shape[0], dtype=np.float64)
    finite_y = np.linspace(10.0, 80.0, combos.shape[0], dtype=np.float64)
    x = np.where(combos[:, 0], finite_x, np.nan)[None, :]
    w = np.where(combos[:, 1], finite_w, np.nan)[None, :]
    y = np.where(combos[:, 2], finite_y, np.nan)[None, :]

    runtime = compile_formula("Ridge(x, y, w, 0, 0.1)", cpp=False)
    _, batch_out = runtime.run_batch({"x": x, "y": y, "w": w})
    state = runtime.init_state(x.shape[1])
    state, tick_out = runtime.tick(state, x[0], y[0], w[0])

    stateful_beta = _run_with_ridge_statefulness(
        "get_beta(Ridge(x, y, w, 0, 0.1))", is_stateful=True, x=x, y=y, w=w
    )
    projected_beta = _run("get_beta(Ridge(x, y, w, 0, 0.1))", x=x, y=y, w=w)
    projected_preds = _run("get_preds(Ridge(x, y, w, 0, 0.1))", x=x, y=y, w=w)

    np.testing.assert_allclose(np.asarray(batch_out.beta), stateful_beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(batch_out.preds), projected_preds, rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(np.asarray(tick_out.beta), stateful_beta[0], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(np.asarray(tick_out.preds), projected_preds[0], rtol=1e-12, atol=1e-12, equal_nan=True)
    np.testing.assert_allclose(projected_beta, stateful_beta, rtol=1e-12, atol=1e-12)


def test_cpp_flat_new_stateless_ops_match_jax_flat():
    x = np.array([[0.2, 0.5, 0.8], [0.1, np.nan, 0.9]], dtype=np.float64)
    for formula in ("norm_inv(x)", "xs_norm(x)", "clip(x, 0.25, 0.75)", "cache(xs_rank(x))"):
        jax_runtime = compile_formula(formula, cpp=False)
        cpp_runtime = compile_formula_native(formula)
        _, jax_out = jax_runtime.run_batch({"x": x})
        _, cpp_out = cpp_runtime.run_batch({"x": x})
        np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=1e-7, atol=1e-7, equal_nan=True)
