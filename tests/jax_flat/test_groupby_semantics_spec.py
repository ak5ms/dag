from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from trading_dsl_engine.base.dsl import *
from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import GroupByOp, Op


def test_groupby_contract_requires_canonical_three_arg_form():
    with pytest.raises(ValueError, match="canonical form"):
        compile_formula("groupby(ts, close)")


def test_groupby_contract_scalar_key_nan_bucket_and_incremental_state():
    runtime = compile_formula("groupby((key,), x, cumsum(self_))")
    state = runtime.init_state(2)
    state, out1 = runtime.tick(state, jnp.array([1.0, jnp.nan]), jnp.array([2.0, 3.0]))
    state, out2 = runtime.tick(state, jnp.array([1.0, jnp.nan]), jnp.array([4.0, 5.0]))
    assert jnp.allclose(out1, jnp.array([2.0, 3.0]), equal_nan=True)
    assert jnp.allclose(out2, jnp.array([6.0, 8.0]), equal_nan=True)


def test_groupby_contract_accepts_arbitrary_tuple_key_length():
    runtime = compile_formula("groupby((k1, k2, k3), x, cumsum(self_))")
    state = runtime.init_state(1)
    state, out1 = runtime.tick(state, jnp.array([1.0]), jnp.array([10.0]), jnp.array([100.0]), jnp.array([2.0]))
    state, out2 = runtime.tick(state, jnp.array([1.0]), jnp.array([10.0]), jnp.array([100.0]), jnp.array([5.0]))
    assert float(out1[0]) == pytest.approx(2.0)
    assert float(out2[0]) == pytest.approx(7.0)


def test_groupby_contract_allows_single_univ_in_tuple_key():
    formula = groupby((univ([0], [1]), var("ts")), var("close"), (cumsum(self_)))

    close = jnp.array([
        [10., 20.],
        [1.,  2.],
        [20., 30.],
        [jnp.nan, jnp.nan],
    ])
    ts = jnp.array([
        [1., 1.],
        [1., 2.],
        [2., 1.],
        [2., 2.],
    ])

    runtime = compile_formula(formula)
    state, out = runtime.run_batch((ts, close))

    desired = jnp.array([
         [10., 20.],
         [11., 2. ],
         [20., 50.],
         [jnp.nan, jnp.nan ],
    ])
    assert jnp.allclose(out, desired, equal_nan=True)

def test_groupby_nested_op():
    formula = groupby((univ([0, 1]), var("ts")), var("close"), cumsum(cumsum(self_)))

    close = jnp.array([
        [10., 20.],
        [1.,  2. ],
        [20., 50.],
        [jnp.nan, jnp.nan],
    ])
    ts = jnp.array([
        [1., 1.],
        [1., 2.],
        [2., 1.],
        [2., 2.],
    ])

    runtime = compile_formula(formula)
    state, out = runtime.run_batch((ts, close))

    desired = jnp.array([
         [10., 20.],
         [21., 2. ],
         [20., 90.],
         [jnp.nan, jnp.nan],
    ])

    assert jnp.allclose(out, desired, equal_nan=True)



def test_groupby_batch_supports_non_root_composition():
    formula = "close + groupby((univ([0, 1], [2]), open), close, cumsum(self_))"
    runtime = compile_formula(formula)
    open_ = jnp.array([
        [1.0, 1.0, 2.0],
        [1.0, 2.0, 2.0],
        [2.0, 1.0, 2.0],
    ])
    close = jnp.array([
        [10.0, 20.0, 30.0],
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ])

    state = runtime.init_state(3)

    def step(carry, row):
        open_row, close_row = row
        return runtime.tick(carry, close_row, open_row)

    _, tick_out = jax.lax.scan(step, state, (open_, close))
    _, batch_out = runtime.run_batch((close, open_))

    assert jnp.allclose(batch_out, tick_out, equal_nan=True)

def test_groupby_contract_rejects_multiple_univ_in_tuple_key():
    with pytest.raises(ValueError, match="at most one univ"):
        compile_formula("groupby((univ([0]), univ([1]), ts), close, mean(self_))")


def test_groupby_contract_scalar_key_supports_non_unary_ops():
    runtime = compile_formula("groupby((key,), x, add(self_, y))")
    state = runtime.init_state(2)
    state, out1 = runtime.tick(state, jnp.array([1.0, 2.0]), jnp.array([2.0, 3.0]), jnp.array([10.0, 20.0]))
    state, out2 = runtime.tick(state, jnp.array([1.0, 2.0]), jnp.array([4.0, 5.0]), jnp.array([1.0, 1.0]))
    assert jnp.allclose(out1, jnp.array([12.0, 23.0]), equal_nan=True)
    assert jnp.allclose(out2, jnp.array([5.0, 6.0]), equal_nan=True)


def test_groupby_contract_scalar_key_supports_stateful_dataclass_ops():
    runtime = compile_formula("groupby((key,), x, ewm(self_, 3))")
    state = runtime.init_state(2)
    state, out1 = runtime.tick(state, jnp.array([1.0, jnp.nan]), jnp.array([2.0, 4.0]))
    state, out2 = runtime.tick(state, jnp.array([1.0, jnp.nan]), jnp.array([6.0, 8.0]))
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
        return state + 1.0, _DummyObjOut(a=x + state, b=x * 2.0)


def test_groupby_contract_supports_dataclass_outputs_in_grouped_apply_op():
    op = GroupByOp(inner_op=_DummyObjOp(), n_keys=1)
    state = op.init_state(jnp.array([0.0]))
    state, out = op.tick(state, jnp.array([1.0, jnp.nan, 1.0]), jnp.array([2.0, 3.0, 4.0]))
    assert jnp.allclose(out.a, jnp.array([2.0, 3.0, 5.0]))
    assert jnp.allclose(out.b, jnp.array([4.0, 6.0, 8.0]))
