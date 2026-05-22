import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_new.engine import compile_formula


def test_jax_new_full_graph_cse_reuses_subexpression_node_once():
    runtime = compile_formula("mul(xstd(ewm(div(open, close), 60)), ewm(div(open, close), 60))")
    op_names = [type(node.op).__name__ for node in runtime.program.nodes]
    assert op_names.count("NaryOp") >= 2
    assert op_names.count("EwmOp") == 1

    state0 = runtime.init_state(4)
    open_row = jnp.array([10.0, 20.0, 30.0, 40.0])
    close_row = jnp.array([11.0, 19.0, 31.0, 39.0])
    _, out = runtime.tick(state0, open_row, close_row)
    assert out.shape == (4,)
    jaxpr = jax.make_jaxpr(runtime.tick)(state0, open_row, close_row)
    txt = str(jaxpr)
    assert "sqrt" in txt
    assert "searchsorted" not in txt
    assert "concatenate" not in txt


def test_jax_new_supports_unary_binary_where_and_cumsum():
    runtime = compile_formula(
        "where(gt(abs(sub(open, close)), 0.1), cumsum(div(open, close)), fillna(open, close))")
    state0 = runtime.init_state(3)
    open_row = jnp.array([1.0, 2.0, jnp.nan])
    close_row = jnp.array([1.0, 1.0, 4.0])
    state1, out1 = runtime.tick(state0, open_row, close_row)
    _, out2 = runtime.tick(state1, open_row, close_row)
    assert out1.shape == (3,)
    assert out2.shape == (3,)
    assert jnp.isfinite(out2[0])
