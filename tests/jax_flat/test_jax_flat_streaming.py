import jax
import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import ewm, shift, var
from trading_dsl_engine.jax_flat.engine import compile_formula


def test_jax_flat_streaming_state_layout_and_natural_rank_output():
    runtime = compile_formula("ewm(xs_sort(add(close, open)), 21)")
    state0 = runtime.init_state(4)
    assert len(state0) == 1

    open_row = jnp.array([10.0, 20.0, 30.0, 40.0])
    close_row = jnp.array([11.0, 19.0, 31.0, 39.0])
    state1, out_stream = runtime.tick(state0, open_row, close_row)
    assert out_stream.shape == (4,)
    _, out_tick = runtime.tick(state1, open_row, close_row)
    assert out_tick.shape == (4,)


def test_jax_flat_streaming_jaxpr_has_compact_state_abi():
    runtime = compile_formula("cumsum(div(open, close))")
    state0 = runtime.init_state(3)
    open_row = jnp.array([1.0, 2.0, 3.0])
    close_row = jnp.array([1.0, 1.0, 2.0])

    jaxpr = jax.make_jaxpr(runtime.tick)(state0, open_row, close_row)
    txt = str(jaxpr)
    assert "searchsorted" not in txt
    assert "concatenate" not in txt


def _momentum_runtime():
    returns = var("returns")
    half_life = var("hl")
    signal = shift((returns - ewm(returns, half_life)) / (ewm(returns**2, half_life) ** 0.5), 1, 2)
    return compile_formula(signal)


def test_jax_flat_fold_batch_matches_materialized_autodiff_and_avoids_output_tape():
    returns = jax.random.normal(jax.random.PRNGKey(0), (4, 2), dtype=jnp.float64)
    runtime = _momentum_runtime()

    def materialized_sharpe(raw_half_life):
        half_life = jax.nn.softplus(raw_half_life) + 2.0
        _, weights = runtime.run_batch(
            {"returns": returns, "hl": jnp.broadcast_to(half_life, returns.shape)}
        )
        pnl = (jnp.nan_to_num(weights) * returns).sum(axis=1)
        return pnl.mean() / (pnl.std() + 1e-12)

    def pnl_moments(acc, weights, input_rows):
        returns_row, _hl_row = input_rows
        count, total, total_sq = acc
        pnl = (jnp.nan_to_num(weights) * returns_row).sum()
        return count + 1.0, total + pnl, total_sq + pnl * pnl

    def folded_sharpe(raw_half_life):
        half_life = jax.nn.softplus(raw_half_life) + 2.0
        init = (jnp.array(0.0, dtype=jnp.float64),) * 3
        _, (count, total, total_sq) = runtime.fold_batch(
            {"returns": returns, "hl": jnp.broadcast_to(half_life, returns.shape)},
            init,
            pnl_moments,
        )
        mean = total / count
        variance = jnp.maximum(total_sq / count - mean * mean, 0.0)
        return mean / (jnp.sqrt(variance) + 1e-12)

    raw = jnp.array(2.0, dtype=jnp.float64)
    material_value, material_grad = jax.value_and_grad(materialized_sharpe)(raw)
    folded_value, folded_grad = jax.value_and_grad(folded_sharpe)(raw)

    np.testing.assert_allclose(folded_value, material_value, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(folded_grad, material_grad, rtol=1e-12, atol=1e-12)

    material_jaxpr = str(jax.make_jaxpr(jax.value_and_grad(materialized_sharpe))(raw))
    folded_jaxpr = str(jax.make_jaxpr(jax.value_and_grad(folded_sharpe))(raw))

    # The slow large-example compile path came from differentiating a scalar
    # objective through run_batch's full output array assembly. The folded path
    # still traces the same streaming state transitions, but it does not include
    # the chunk/output scatter tape that scales with materialized batch output.
    assert folded_jaxpr.count("dynamic_update_slice") < material_jaxpr.count("dynamic_update_slice")
    assert len(folded_jaxpr) < len(material_jaxpr)
