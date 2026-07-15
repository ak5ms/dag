import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import ewm, var
from trading_dsl_engine.jax_flat import compile_features, compile_formula, compile_optimized_formula
from trading_dsl_engine.jax_flat.optimized import _node_batch_chunk

jax.config.update("jax_enable_x64", True)


def _nested_ewm(depth: int):
    expr = var("x")
    for level in range(depth):
        expr = ewm(
            expr,
            5.0 + 4.0 * level,
            ignore_na=True,
            adjust=False,
        )
    return expr


def _while_count(lowered) -> int:
    try:
        hlo = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    except Exception:
        hlo = str(lowered.compiler_ir(dialect="stablehlo"))
    return hlo.count(" while(") + hlo.count("stablehlo.while")


def test_pair_fused_depth_five_matches_existing_runtime_and_reduces_loops():
    rng = np.random.default_rng(42)
    data = rng.normal(size=(257, 4))
    data[rng.random(data.shape) < 0.15] = np.nan
    expr = _nested_ewm(5)

    current = compile_formula(expr, cpp=False)
    _, expected = current.run_batch({"x": jnp.asarray(data)})

    optimized = compile_optimized_formula(expr, chunk_size=64)
    _, actual = optimized.run_batch({"x": data}, out_path=None)
    np.testing.assert_allclose(actual, np.asarray(expected), rtol=1e-11, atol=1e-11, equal_nan=True)

    inner = optimized.runtime
    state = inner.init_state(data.shape[1])
    sample = jnp.asarray(data[:64])
    lowered = _node_batch_chunk.lower(
        inner,
        state,
        (sample,),
        jnp.asarray(0, dtype=jnp.int64),
    )
    assert _while_count(lowered) <= 3


def test_observable_shared_ewm_is_not_fused_away():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(193, 3))
    data[rng.random(data.shape) < 0.1] = np.nan

    base = ewm(var("x"), 7.0, ignore_na=True, adjust=False)
    formulas = {
        "base": base,
        "nested": ewm(base, 19.0, ignore_na=True, adjust=False),
    }
    runtime = compile_features(formulas, chunk_size=64)
    _, actual = runtime.run_batch({"x": data}, out_path=None)

    for name, expr in formulas.items():
        current = compile_formula(expr, cpp=False)
        _, expected = current.run_batch({"x": jnp.asarray(data)})
        np.testing.assert_allclose(actual[name], np.asarray(expected), rtol=1e-11, atol=1e-11, equal_nan=True)
