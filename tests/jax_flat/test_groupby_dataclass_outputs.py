from dataclasses import dataclass

import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat.ops import Op
from trading_dsl_engine.jax_flat.ops_groupby import GroupByOp


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class _DummyObjOut:
    a: jax.Array
    b: jax.Array


class _DummyObjOp(Op):
    is_stateful = True

    def init_state(self, sample):
        del sample
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


def test_groupby_contract_jitted_supports_dataclass_outputs_in_grouped_apply_op():
    op = GroupByOp(inner_op=_DummyObjOp(), n_keys=1)
    state = op.init_state(jnp.array([0.0]))
    _, out = jax.jit(lambda s, k, x: op.tick(s, k, x))(
        state,
        jnp.array([1.0, jnp.nan, 1.0]),
        jnp.array([2.0, 3.0, 4.0]),
    )

    assert jnp.allclose(out.a, jnp.array([2.0, 3.0, 5.0]))
    assert jnp.allclose(out.b, jnp.array([4.0, 6.0, 8.0]))


@dataclass(frozen=True)
class _PlainDataclassOut:
    mean: object
    centered: object


class _StatelessGroupedOp(Op):
    is_stateful = False

    def tick(self, state, x):
        del state
        mean = jnp.nanmean(x)
        return None, _PlainDataclassOut(mean=mean, centered=x - mean)


def test_groupby_contract_auto_registers_plain_dataclass_outputs():
    op = GroupByOp(inner_op=_StatelessGroupedOp(), n_keys=1, universe_groups=((0, 1, 2, 3),))
    state = op.init_state(jnp.array([0.0, 0.0, 0.0, 0.0]))
    _, out = jax.jit(lambda s, k, x: op.tick(s, k, x))(
        state,
        jnp.array([0.0, 0.0, 1.0, 1.0]),
        jnp.array([1.0, 3.0, 10.0, 30.0]),
    )

    assert isinstance(out, _PlainDataclassOut)
    assert jnp.allclose(out.mean, jnp.array([2.0, 2.0, 20.0, 20.0]), equal_nan=True)
    assert jnp.allclose(out.centered, jnp.array([-1.0, 1.0, -10.0, 10.0]), equal_nan=True)