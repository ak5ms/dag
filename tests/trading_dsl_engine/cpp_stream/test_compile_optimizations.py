from __future__ import annotations

import importlib

import numpy as np

from trading_dsl_engine.base.dsl import var


compile_module = importlib.import_module(
    "trading_dsl_engine.cpp_stream.python.compile"
)


def _capture_compile_passes(monkeypatch, *, n_instruments):
    calls = []
    real_compile_ir = compile_module.compile_ir

    def counted_compile_ir(*args, **kwargs):
        calls.append(kwargs.get("input_value_types"))
        return real_compile_ir(*args, **kwargs)

    captured = {}

    def fake_compile_program(program, **kwargs):
        captured["program"] = program
        captured["kwargs"] = kwargs
        return program

    monkeypatch.setattr(compile_module, "compile_ir", counted_compile_ir)
    monkeypatch.setattr(compile_module, "_compile_program", fake_compile_program)
    data = {"x": np.arange(36.0).reshape(4, 9)}
    result = compile_module.compile_formula(
        var("x") + 1.0,
        data,
        n_instruments=n_instruments,
    )
    assert result is captured["program"]
    return calls, captured


def test_known_instrument_count_uses_one_ir_build(monkeypatch):
    calls, captured = _capture_compile_passes(
        monkeypatch,
        n_instruments=9,
    )
    assert len(calls) == 1
    assert captured["kwargs"]["n_instruments"] == 9


def test_inferred_instrument_count_rebuilds_exact_types(monkeypatch):
    calls, captured = _capture_compile_passes(
        monkeypatch,
        n_instruments=None,
    )
    assert len(calls) == 2
    assert captured["kwargs"]["n_instruments"] == 9


def test_header_digest_cache_invalidates_after_header_edit(tmp_path):
    cpp_root = tmp_path / "cpp"
    eigen_root = tmp_path / "eigen"
    cpp_root.mkdir()
    macros = eigen_root / "Eigen" / "src" / "Core" / "util" / "Macros.h"
    macros.parent.mkdir(parents=True)
    header = cpp_root / "kernel.hpp"
    header.write_text("#define VALUE 1\n")
    macros.write_text("#define EIGEN_VALUE 1\n")

    first = compile_module._header_digest(str(cpp_root), str(eigen_root))
    assert compile_module._header_digest(str(cpp_root), str(eigen_root)) == first

    header.write_text("#define VALUE 22\n")
    second = compile_module._header_digest(str(cpp_root), str(eigen_root))
    assert second != first


def test_expression_key_cache_does_not_alias_recycled_python_objects():
    from trading_dsl_engine.base.parser import Number
    from trading_dsl_engine.ir.frontend import _expr_key, clear_expr_key_id_memo
    clear_expr_key_id_memo()
    # Expansion creates temporary ASTs. Their Python ids can be reused while
    # one compilation is still running, so id alone is not an identity guard.
    for value in range(2000):
        assert _expr_key(Number(value)) == ('num', ('int', value))
