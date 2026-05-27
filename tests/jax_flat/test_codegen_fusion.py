import jax
import jax.numpy as jnp

from trading_dsl_engine.jax_flat.engine import compile_formula


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
