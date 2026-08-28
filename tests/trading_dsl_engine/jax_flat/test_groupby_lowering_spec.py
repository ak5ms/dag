import jax.numpy as jnp
import pytest

import trading_dsl_engine.base.dsl as tde
from trading_dsl_engine.base.parser import Call, KeyTuple, parse_formula
from trading_dsl_engine.jax_flat.engine import compile_formula


def test_groupby_parse_keeps_canonical_three_arg_call_shape():
    expr = parse_formula("groupby((key1, key2), x, cumsum(self_))")
    assert isinstance(expr, Call)
    assert expr.fn == "groupby"
    assert len(expr.args) == 3
    assert isinstance(expr.args[0], KeyTuple)


def test_grouped_expr_apply_lowers_to_canonical_groupby_call():
    expr = tde.var("x").groupby((tde.var("k1"), tde.var("k2"))).apply(tde.cumsum(tde.self_))
    assert isinstance(expr, Call)
    assert expr.fn == "groupby"
    assert len(expr.args) == 3
    assert isinstance(expr.args[0], KeyTuple)


def test_expr_method_chaining_lowers_ops_and_inline_groupby_apply():
    expr = tde.var("a").add(tde.var("b")).groupby(tde.var("key"), tde.cumsum(tde.self_)).xs_rank()
    assert isinstance(expr, Call)
    assert expr.fn == "xs_rank"

    grouped = expr.args[0]
    assert isinstance(grouped, Call)
    assert grouped.fn == "groupby"
    assert len(grouped.args) == 3
    assert isinstance(grouped.args[0], KeyTuple)

    lhs = grouped.args[1]
    assert isinstance(lhs, Call)
    assert lhs.fn == "add"


def test_expr_pipe_and_registered_function_method_chaining():
    name = "test_center_then_scale"
    try:
        @tde.register_dsl_function(name)
        def center_then_scale(x, scale=1.0):
            return (x - x.xs_mean()) * scale

        source = tde.var("x")
        piped = source.pipe(center_then_scale, scale=2.0)
        chained = source.test_center_then_scale(scale=2.0)
        assert repr(piped) == repr(chained)
        assert chained.fn == "mul"
    finally:
        tde.DEFAULT_DSL_REGISTRY._fns.pop(name, None)
        tde._DSL_OP_SIGNATURES.pop(name, None)


def test_jax_flat_rejects_noncanonical_groupby_arity():
    with pytest.raises(ValueError, match="canonical form"):
        compile_formula("groupby(ts, x)")


def test_jax_flat_accepts_canonical_groupby_node_shape_for_lowering_path():
    runtime = compile_formula("groupby((key,), x, cumsum(self_))")
    state = runtime.init_state(1)
    _, out = runtime.tick(state, jnp.array([1.0]), jnp.array([2.0]))
    assert out.shape == (1,)
