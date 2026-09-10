from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np

from trading_dsl_engine.base.dsl import var


compile_module = importlib.import_module("trading_dsl_engine.cpp_stream.python.compile")
compiler_support_module = importlib.import_module(
    "trading_dsl_engine.cpp_stream.python.compiler_support"
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

    first = compiler_support_module._header_digest(str(cpp_root), str(eigen_root))
    assert (
        compiler_support_module._header_digest(str(cpp_root), str(eigen_root)) == first
    )

    header.write_text("#define VALUE 22\n")
    second = compiler_support_module._header_digest(str(cpp_root), str(eigen_root))
    assert second != first


def test_compile_metrics_are_attached_by_stage(monkeypatch, tmp_path):
    monkeypatch.setattr(
        compile_module,
        "build_shared",
        lambda source, **kwargs: _fake_build_shared(tmp_path, kwargs),
    )
    runtime = compile_module.compile_formula(
        var("x") + 1.0,
        n_instruments=3,
    )

    metrics = runtime.compile_metrics
    assert metrics is not None
    assert metrics.total_seconds >= 0.0
    assert metrics.native_cache_hit
    assert {
        "frontend",
        "type_analysis",
        "lowering",
        "parallel_planning",
        "code_generation",
        "native_build",
        "dependency_fingerprint_seconds",
        "native_compile_seconds",
        "runtime_setup",
    } <= metrics.stage_seconds.keys()
    assert all(seconds >= 0.0 for seconds in metrics.stage_seconds.values())


def _fake_build_shared(tmp_path: Path, kwargs):
    kwargs["metrics"].update(
        dependency_fingerprint_seconds=0.01,
        native_compile_seconds=0.0,
        native_cache_hit=True,
    )
    return tmp_path / "formula.so", tmp_path / "formula.cpp"
