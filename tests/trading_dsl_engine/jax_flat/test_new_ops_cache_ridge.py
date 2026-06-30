from dataclasses import replace
from itertools import product
import os

import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.jax_flat import compile_formula
from trading_dsl_engine.jax_flat import engine as jax_flat_engine
from trading_dsl_engine.jax_flat.engine import _build_state_layout
from trading_dsl_engine.jax_flat.engine_cpp import compile_formula as compile_formula_native
from trading_dsl_engine.jax_flat.ops import RidgeOp
from trading_dsl_engine.jax_ffi.nnqp import nnqp, solve_direct


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




def test_cache_can_save_materialized_values_on_python_call_object():
    from trading_dsl_engine.base.dsl import cache, ewm, var

    a = var("a")
    b = var("b")
    data = {
        "a": np.array([[1.0, 2.0], [3.0, 5.0], [7.0, 11.0]], dtype=np.float64),
        "b": np.array([[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]], dtype=np.float64),
    }
    cached = cache(ewm(a + b, 5), storage="disk", save="call")
    expr1 = 3.0 + cached

    runtime1 = compile_formula(expr1, cpp=False)
    _, out1 = runtime1.run_batch(data)
    call_value = getattr(cached, "_jax_flat_cached_value")

    assert isinstance(call_value, np.memmap)
    np.testing.assert_allclose(out1, np.asarray(call_value) + 3.0)

    expr2 = expr1 + 4.0
    runtime2 = compile_formula(expr2, cpp=False)
    _, out2 = runtime2.run_batch({})

    assert "a" not in runtime2.program.input_names
    assert "b" not in runtime2.program.input_names
    assert any(name.startswith("__cache_call_") for name in runtime2.program.input_names)
    np.testing.assert_allclose(out2, np.asarray(call_value) + 7.0)


def test_cache_call_save_is_opt_in_and_runtime_default_does_not_modify_call():
    from trading_dsl_engine.base.dsl import cache, var

    x = var("x")
    expr = cache(x)
    data = {"x": np.array([[1.0, 2.0]], dtype=np.float64)}

    compile_formula(expr, cpp=False).run_batch(data)

    assert not hasattr(expr, "_jax_flat_cached_value")


def test_cached_runtime_values_feed_matching_subgraphs_and_tuples():
    x = np.array([[1.0, 2.0, np.nan], [4.0, -1.0, 3.0]], dtype=np.float64)
    y = np.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64)

    cache_runtime_a = compile_formula("cache(x + 1.0)", cpp=False)
    cache_runtime_b = compile_formula("cache(y - 2.0)", cpp=False)
    cache_runtime_a.run_batch({"x": x})
    cache_runtime_b.run_batch({"y": y})

    runtime = compile_formula(
        "((2.0 * (x + 1.0)) + ((x + 1.0) * (y - 2.0)))",
        runtimes=(cache_runtime_a, cache_runtime_b),
        cpp=False,
    )
    _, out = runtime.run_batch({"x": x, "y": y})

    assert any(name.startswith("__cache_runtime_") for name in runtime.program.input_names)
    expected = (2.0 * (x + 1.0)) + ((x + 1.0) * (y - 2.0))
    np.testing.assert_allclose(out, expected, equal_nan=True)


def test_compile_formula_runtime_cache_requires_materialized_values():
    cache_runtime = compile_formula("cache(x + 1.0)", cpp=False)

    with np.testing.assert_raises_regex(ValueError, "Run run_batch first"):
        compile_formula("x + 1.0", runtimes=cache_runtime, cpp=False)


def test_disk_cache_streams_to_memmap_and_cleans_unique_run_files(monkeypatch):
    x = np.arange(10.0, dtype=np.float64).reshape(5, 2)
    monkeypatch.setattr(jax_flat_engine, "_BATCH_CHUNK_SIZE", 2)
    runtime = compile_formula('cache(x, "disk")', cpp=False)

    _, first_out = runtime.run_batch({"x": x})
    first_cached = next(iter(runtime.get_cached_values().values()))
    first_path = first_cached.filename

    assert isinstance(first_out, np.memmap)
    assert first_out is first_cached
    assert os.path.exists(first_path)
    assert "run0" in os.path.basename(first_path)
    np.testing.assert_allclose(first_cached, x)

    _, second_out = runtime.run_batch({"x": x + 1.0})
    second_cached = next(iter(runtime.get_cached_values().values()))
    second_path = second_cached.filename

    assert isinstance(second_out, np.memmap)
    assert second_out is second_cached
    assert second_path != first_path
    assert not os.path.exists(first_path)
    assert os.path.exists(second_path)
    np.testing.assert_allclose(second_cached, x + 1.0)

    runtime.clear_cached_values()
    assert not os.path.exists(second_path)

    nested_runtime = compile_formula('cache(x + 1.0, "disk") * 2.0', cpp=False)
    _, nested_out = nested_runtime.run_batch({"x": x})
    nested_cached = next(iter(nested_runtime.get_cached_values().values()))

    assert isinstance(nested_cached, np.memmap)
    np.testing.assert_allclose(nested_out, (x + 1.0) * 2.0)
    np.testing.assert_allclose(nested_cached, x + 1.0)
    nested_runtime.clear_cached_values()
    assert not os.path.exists(nested_cached.filename)


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


def test_hl_zero_ridge_stateless_all_nan_weights_do_not_emit_nonfinite_beta():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    y = np.array([[2.0, 3.0, 4.0]], dtype=np.float64)
    w = np.array([[np.nan, np.nan, np.nan]], dtype=np.float64)

    stateless_beta = _run("get_beta(Ridge(x, y, w, 0, 0.1))", x=x, y=y, w=w)
    stateful_beta = _run_with_ridge_statefulness(
        "get_beta(Ridge(x, y, w, 0, 0.1))", is_stateful=True, x=x, y=y, w=w
    )
    stateless_preds = _run("get_preds(Ridge(x, y, w, 0, 0.1))", x=x, y=y, w=w)
    stateful_preds = _run_with_ridge_statefulness(
        "get_preds(Ridge(x, y, w, 0, 0.1))", is_stateful=True, x=x, y=y, w=w
    )

    assert np.all(np.isfinite(stateless_beta))
    np.testing.assert_allclose(stateless_beta, stateful_beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stateless_preds, stateful_preds, rtol=1e-12, atol=1e-12)


def test_hl_zero_ridge_stateless_single_nan_weight_skips_weighted_instrument():
    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float64)
    y = np.array([[2.0, 3.0, 4.0]], dtype=np.float64)
    w = np.array([[1.0, np.nan, 1.0]], dtype=np.float64)

    stateless_beta = _run("get_beta(Ridge(x, y, w, 0, 0.1))", x=x, y=y, w=w)
    stateful_beta = _run_with_ridge_statefulness(
        "get_beta(Ridge(x, y, w, 0, 0.1))", is_stateful=True, x=x, y=y, w=w
    )
    stateless_preds = _run("get_preds(Ridge(x, y, w, 0, 0.1))", x=x, y=y, w=w)
    stateful_preds = _run_with_ridge_statefulness(
        "get_preds(Ridge(x, y, w, 0, 0.1))", is_stateful=True, x=x, y=y, w=w
    )

    assert np.all(np.isfinite(stateless_beta))
    assert np.all(np.isfinite(stateless_preds))
    np.testing.assert_allclose(stateless_beta, stateful_beta, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stateless_preds, stateful_preds, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(stateless_preds, x * stateless_beta, rtol=1e-12, atol=1e-12)


def test_ridge_beta_statefulness_cartesian_realistic_nan_inputs():
    rng = np.random.default_rng(20260613)
    n_rows = 10_000
    n_cols = 9
    close = np.exp(rng.normal(0.0, 0.02, size=(n_rows, n_cols))) + 10.0
    open_ = close * (1.0 + rng.normal(0.0, 0.001, size=(n_rows, n_cols)))
    y = rng.normal(size=(n_rows, n_cols))
    half_spread = np.exp(rng.normal(-2.0, 0.4, size=(n_rows, n_cols)))
    nan_mask = rng.random((n_rows, n_cols)) < 0.08
    e_by_sigma = {
        0.0: np.zeros((n_rows, n_cols), dtype=np.float64),
        1.0: rng.normal(0.0, 1.0, size=(n_rows, n_cols)),
    }

    def with_nan_pattern(values, pattern):
        out = values.copy()
        if pattern == "all":
            out[:] = np.nan
        elif pattern == "some":
            out[nan_mask] = np.nan
        return out

    x1 = "xs_rank(ewm(open / close, 30))"
    features = f"cat({x1}, sub(e, {x1}))"
    weights = "fillna(pow(half_spread, -2.0), 0.0)"
    nan_patterns = ("none", "some", "all")

    for hl, lam in product((0.0, 30.0), (0.0, 0.1)):
        formula = f"get_beta(Ridge({features}, y, {weights}, {hl}, {lam}))"
        runtime = compile_formula(formula, cpp=False)
        stateful_runtime = _with_ridge_statefulness(runtime, is_stateful=True)
        for sigma, e in e_by_sigma.items():
            for x_nan, y_nan, w_nan in product(nan_patterns, repeat=3):
                case = f"hl={hl}, lam={lam}, sigma={sigma}, x={x_nan}, y={y_nan}, w={w_nan}"
                data = {
                    "open": with_nan_pattern(open_, x_nan),
                    "close": close,
                    "e": e,
                    "y": with_nan_pattern(y, y_nan),
                    "half_spread": with_nan_pattern(half_spread, w_nan),
                }
                _, out = runtime.run_batch(data)
                beta = np.asarray(out)
                assert np.all(np.isfinite(beta)), case
                if hl == 0.0:
                    _, stateful_out = stateful_runtime.run_batch(data)
                    np.testing.assert_allclose(
                        beta,
                        np.asarray(stateful_out),
                        rtol=1e-10,
                        atol=1e-10,
                        err_msg=case,
                    )


def test_cpp_flat_new_stateless_ops_match_jax_flat():
    x = np.array([[0.2, 0.5, 0.8], [0.1, np.nan, 0.9]], dtype=np.float64)
    for formula in ("norm_inv(x)", "xs_norm(x)", "clip(x, 0.25, 0.75)", "cache(xs_rank(x))"):
        jax_runtime = compile_formula(formula, cpp=False)
        cpp_runtime = compile_formula_native(formula)
        _, jax_out = jax_runtime.run_batch({"x": x})
        _, cpp_out = cpp_runtime.run_batch({"x": x})
        np.testing.assert_allclose(cpp_out, np.asarray(jax_out), rtol=1e-7, atol=1e-7, equal_nan=True)


def test_ridge_nonneg_keyword_projects_negative_coefficients():
    x = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64)
    y = np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], dtype=np.float64)

    unconstrained = _run("get_beta(Ridge(x, y=y, hl=0, lambda_=0.0, nonneg=False))", x=x, y=y)
    constrained = _run("get_beta(Ridge(x, y=y, hl=0, lambda_=0.0, nonneg=True))", x=x, y=y)

    assert np.any(unconstrained[0] < -1e-9)
    np.testing.assert_allclose(constrained[0], np.zeros(1), atol=1e-10)
    assert np.all(constrained >= -1e-10)


def test_native_ridge_nonneg_matches_jax_flat():
    data = {
        "x": np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=np.float64),
        "y": np.array([[-1.0, -2.0, -3.0], [1.0, 2.0, 3.0]], dtype=np.float64),
    }
    formula = "get_beta(Ridge(x, y=y, hl=0, lambda_=0.0, nonneg=True))"
    _, jax_out = compile_formula(formula, cpp=False).run_batch(data)
    _, native_out = compile_formula_native(formula).run_batch(data)
    np.testing.assert_allclose(native_out, np.asarray(jax_out), atol=1e-9)


def test_nnqp_ffi_forward_and_backward():
    A = jnp.array([[2.0, 0.0], [0.0, 2.0]], dtype=jnp.float64)
    c = jnp.array([2.0, -1.0], dtype=jnp.float64)

    beta = jax.jit(nnqp)(A, c)
    np.testing.assert_allclose(np.asarray(beta), np.asarray(solve_direct(np.asarray(A), np.asarray(c))), atol=1e-10)
    np.testing.assert_allclose(np.asarray(beta), np.array([1.0, 0.0]), atol=1e-10)

    def loss(lhs, rhs):
        b = nnqp(lhs, rhs)
        return 0.5 * jnp.sum(b * b)

    dA, dc = jax.jit(jax.grad(loss, argnums=(0, 1)))(A, c)
    np.testing.assert_allclose(np.asarray(dc), np.array([0.5, 0.0]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(dA), np.array([[-0.5, 0.0], [0.0, 0.0]]), atol=1e-10)


def test_native_ridge_nonneg_large_random_nan_panel_matches_jax_flat():
    rng = np.random.default_rng(20260627)
    data = {
        name: rng.normal(size=(100, 9)).astype(np.float64)
        for name in ("x1", "x2", "x3", "y")
    }
    for name, arr in data.items():
        arr[rng.random(arr.shape) < (0.12 if name != "y" else 0.08)] = np.nan

    formula = "get_beta(Ridge(x1, x2, x3, y=y, hl=0, lambda_=0.05, nonneg=True))"
    _, jax_out = compile_formula(formula, cpp=False).run_batch(data)
    _, native_out = compile_formula_native(formula).run_batch(data)

    np.testing.assert_allclose(native_out, np.asarray(jax_out), rtol=1e-8, atol=1e-8)
    assert native_out.shape == (100, 3)
    assert np.all(native_out >= -1e-10)
