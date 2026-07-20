import jax
import jax.numpy as jnp

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


def test_jax_flat_execution_plan_labels_compiled_formula_nodes():
    runtime = compile_formula("ewm(add(close, open), 3)", cpp=False)

    plan = runtime.get_execution_plan()

    assert [entry["node_id"] for entry in plan] == list(range(len(plan)))
    assert [entry["children"] for entry in plan] == [(), (), (0, 1), (), (2,)]
    assert plan[2]["label"].endswith("/add")
    assert plan[4]["label"].endswith("/Ewm")
    assert plan[4]["stateful"] is True
