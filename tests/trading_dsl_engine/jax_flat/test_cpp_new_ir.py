from trading_dsl_engine.cpp_new.codegen import emit_source
from trading_dsl_engine.cpp_new.lowering import lower
from trading_dsl_engine.jax_flat.compile import compile_formula


def test_cpp_new_ewm_ir_is_static_and_straight_line():
    program = compile_formula("ewm(close, 4.0)", cpp=False).program
    ir = lower(program, n_instruments=150)
    assert [n.opcode for n in ir.nodes] == ["input", "ewm"]
    assert ir.state_bytes and ir.state_bytes % 64 == 0
    source = emit_source(ir)
    assert "ewm_tick" in source
    assert "switch" not in source
    assert "tick(state, input.row(row), output.row(row), workers)" in source
    assert "lines.append" not in source
    assert ir.canonical_json() == ir.canonical_json()


def test_cpp_new_cache_alias_diagnostics():
    program = compile_formula("ewm(close, 4.0) + ewm(close, 4.0)", cpp=False).program
    # The shared frontend already canonicalizes repeated expressions. The native
    # lowering retains the single transition and its source mapping.
    ir = lower(program, n_instruments=8)
    assert sum(node.opcode == "ewm" for node in ir.nodes) == 1


def test_cpp_new_scratch_coloring_reuses_nonoverlapping_intervals():
    from trading_dsl_engine.cpp_new.lowering import _color_scratch

    slots, size = _color_scratch([(1, 64, 1, 2), (2, 64, 3, 4), (3, 128, 2, 5)])
    assert slots[0].offset == slots[1].offset
    assert slots[0].color == slots[1].color
    assert slots[2].offset != slots[0].offset
    assert size == 192
