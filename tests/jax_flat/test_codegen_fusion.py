import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat.engine import compile_formula
from trading_dsl_engine.jax_flat.ops import EwmOp, GroupByOp, InputOp, NaryOp


def _fusion_formula() -> str:
    return (
        "mul(mul(mul(cumsum(xs_sort(add(close, open))), xs_sort(add(close, open))), "
        "xs_sort(add(close, open))), xs_sort(add(close, open)))"
    )


def test_cse_reuses_xs_rank_and_add_nodes_once():
    runtime = compile_formula(_fusion_formula())
    op_names = [type(node.op).__name__ for node in runtime.program.nodes]
    assert op_names.count("CumsumOp") == 1
    assert op_names.count("NaryOp") == 5


def test_xla_hlo_and_compiled_ir_include_fused_sort_path():
    runtime = compile_formula(_fusion_formula())
    state0 = runtime.init_state(9)
    open_row = jnp.linspace(1.0, 9.0, 9)
    close_row = jnp.linspace(2.0, 10.0, 9)

    lowered = jax.jit(runtime.tick).lower(state0, open_row, close_row)
    hlo_text = lowered.compiler_ir(dialect="hlo").as_hlo_text()
    compiled_ir_text = lowered.compile().as_text()

    assert "sort" in hlo_text.lower()
    assert "searchsorted" not in hlo_text.lower()
    assert "fusion" in compiled_ir_text.lower() or "sort" in compiled_ir_text.lower()


def test_stateless_chain_has_no_state_leaves_and_xla_fuses_tick_ir():
    runtime = compile_formula("bspline((close + open) / 3, 5)")
    assert runtime.program.state_layout.total_leaves == 0
    assert all(field.index < 0 for field in runtime.program.state_layout.node_fields)

    state0 = runtime.init_state(4)
    open_row = jnp.linspace(1.0, 4.0, 4)
    close_row = jnp.linspace(2.0, 5.0, 4)
    compiled_ir_text = jax.jit(runtime.tick).lower(state0, close_row, open_row).compile().as_text().lower()

    assert "fusion" in compiled_ir_text
    assert "get-tuple-element" not in compiled_ir_text
    assert "divide" in compiled_ir_text
    assert "exponential" in compiled_ir_text or " exp" in compiled_ir_text


def test_groupby_cse_keeps_stateful_inner_ops_scoped_to_groupby():
    runtime = compile_formula(
        "mul("
        "mul(ewm(add(a, b), 3), ewm(add(a, b), 3)), "
        "groupby((key,), field, mul(2, ewm(add(a, b), 3)))"
        ")"
    )

    top_ewm_ids = [idx for idx, node in enumerate(runtime.program.nodes) if isinstance(node.op, EwmOp)]
    groupby_nodes = [node for node in runtime.program.nodes if isinstance(node.op, GroupByOp)]

    assert len(top_ewm_ids) == 1
    assert len(groupby_nodes) == 1

    groupby_node = groupby_nodes[0]
    inner_nodes = groupby_node.op.inner_op.nodes
    inner_ewm_ids = [idx for idx, node in enumerate(inner_nodes) if isinstance(node.op, EwmOp)]

    assert len(inner_ewm_ids) == 1
    assert top_ewm_ids[0] not in groupby_node.child_ids
    assert all(isinstance(runtime.program.nodes[child_id].op, InputOp) for child_id in groupby_node.child_ids)

    top_nary_nodes = [node for node in runtime.program.nodes if isinstance(node.op, NaryOp)]
    inner_nary_nodes = [node for node in inner_nodes if isinstance(node.op, NaryOp)]
    assert len(top_nary_nodes) == 3
    assert len(inner_nary_nodes) == 2
    assert runtime.program.state_layout.total_leaves == 2
    assert groupby_node.op.inner_op.state_layout.total_leaves == 1
