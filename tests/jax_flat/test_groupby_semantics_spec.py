import jax
from dataclasses import dataclass
import jax.numpy as jnp
import pytest

from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import GroupbyScalarApplyOp, Op


def test_groupby_contract_requires_canonical_three_arg_form():
    with pytest.raises(ValueError, match="canonical form"):
        compile_formula("groupby(ts, close)")


def test_groupby_contract_scalar_key_nan_bucket_and_incremental_state():
    runtime = compile_formula("groupby((key,), x, cumsum(self_))")
    state = runtime.init_state(2)

    key1 = jnp.array([1.0, jnp.nan])
    x1 = jnp.array([2.0, 3.0])
    state, out1 = runtime.tick_stream(state, key1, x1)

    key2 = jnp.array([1.0, jnp.nan])
    x2 = jnp.array([4.0, 5.0])
    state, out2 = runtime.tick_stream(state, key2, x2)

    assert jnp.allclose(out1, jnp.array([2.0, 3.0]), equal_nan=True)
    assert jnp.allclose(out2, jnp.array([6.0, 8.0]), equal_nan=True)


def test_groupby_contract_accepts_arbitrary_tuple_key_length():
    runtime = compile_formula("groupby((k1, k2, k3), x, cumsum(self_))")
    state = runtime.init_state(1)

    keys = [
        (jnp.array([1.0]), jnp.array([10.0]), jnp.array([100.0])),
        (jnp.array([1.0]), jnp.array([10.0]), jnp.array([100.0])),
    ]
    xs = [jnp.array([2.0]), jnp.array([5.0])]

    state, out1 = runtime.tick_stream(state, *keys[0], xs[0])
    state, out2 = runtime.tick_stream(state, *keys[1], xs[1])

    assert float(out1[0]) == pytest.approx(2.0)
    assert float(out2[0]) == pytest.approx(7.0)


@pytest.mark.xfail(strict=True, reason="jax_flat grouped runtime is not implemented yet")
def test_groupby_contract_allows_single_univ_in_tuple_key():
    runtime = compile_formula("groupby((univ([0, 1]), ts), close, mean(self_))")
    state = runtime.init_state(2)
    ts = jnp.array([1.0, 1.0])
    close = jnp.array([10.0, 20.0])
    _, out = runtime.tick_stream(state, ts, close)

    assert out.shape == (2,)


def test_groupby_contract_rejects_multiple_univ_in_tuple_key():
    with pytest.raises(ValueError, match="at most one univ"):
        compile_formula("groupby((univ([0]), univ([1]), ts), close, mean(self_))")


def test_groupby_contract_scalar_key_supports_non_unary_ops():
    runtime = compile_formula("groupby((key,), x, add(self_, y))")
    state = runtime.init_state(2)

    key = jnp.array([1.0, 2.0])
    x1 = jnp.array([2.0, 3.0])
    y1 = jnp.array([10.0, 20.0])
    state, out1 = runtime.tick_stream(state, key, x1, y1)

    x2 = jnp.array([4.0, 5.0])
    y2 = jnp.array([1.0, 1.0])
    state, out2 = runtime.tick_stream(state, key, x2, y2)

    assert jnp.allclose(out1, jnp.array([12.0, 23.0]), equal_nan=True)
    assert jnp.allclose(out2, jnp.array([5.0, 6.0]), equal_nan=True)


def test_groupby_contract_scalar_key_supports_stateful_dataclass_ops():
    runtime = compile_formula("groupby((key,), x, ewm(self_, 3))")
    state = runtime.init_state(2)

    key = jnp.array([1.0, jnp.nan])
    x1 = jnp.array([2.0, 4.0])
    state, out1 = runtime.tick_stream(state, key, x1)

    x2 = jnp.array([6.0, 8.0])
    state, out2 = runtime.tick_stream(state, key, x2)

    assert jnp.allclose(out1, jnp.array([2.0, 4.0]), equal_nan=True)
    assert jnp.allclose(out2, jnp.array([4.0, 6.0]), equal_nan=True)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _DummyObjOut:
    a: jax.Array
    b: jax.Array


class _DummyObjOp(Op):
    is_stateful = True

    def init_state(self, sample):
        return 0.0

    def tick(self, state, *child_values):
        x = child_values[0]
        out = _DummyObjOut(a=x + state, b=x * 2.0)
        return state + 1.0, out


def test_groupby_contract_supports_dataclass_outputs_in_grouped_apply_op():
    op = GroupbyScalarApplyOp(inner_op=_DummyObjOp(), n_keys=1)
    state = op.init_state(jnp.array([0.0]))
    key = jnp.array([1.0, jnp.nan, 1.0])
    x = jnp.array([2.0, 3.0, 4.0])
    state, out = op.tick(state, key, x)
    assert jnp.allclose(out.a, jnp.array([2.0, 3.0, 5.0]))
    assert jnp.allclose(out.b, jnp.array([4.0, 6.0, 8.0]))
