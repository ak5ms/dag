from __future__ import annotations

import base64

import matplotlib.pyplot as plt
import pydot
import pytest

from trading_dsl_engine.base.dsl import cumsum, var
from trading_dsl_engine.cpp_stream.python.runtime import CppStreamRuntime
from trading_dsl_engine.ir import compile_ir


_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


def _labels(graph) -> list[str]:
    return [
        label
        for node in graph.get_nodes()
        if (label := node.get_label()) is not None
    ]


def test_uncompiled_formula_plot_preserves_shared_call_nodes() -> None:
    shared = cumsum(var("close"))
    formula = shared + shared

    graph = formula.plot(show=False)

    labels = _labels(graph)
    assert labels.count("cumsum") == 1
    assert labels.count("add") == 1
    assert "input\nclose" in labels
    assert graph.get_node("n0")[0].get_peripheries() == "2"


def test_neutral_ir_plot_shows_cse_types_and_groupby_rhs() -> None:
    cse_program = compile_ir("add(mul(x, y), mul(x, y))")
    cse_graph = cse_program.plot(show=False)
    labels = _labels(cse_graph)
    assert sum(": mul\n" in label for label in labels) == 1
    assert any("float64" in label for label in labels)
    assert cse_graph.get_node(f"n{cse_program.output_id}")[0].get_peripheries() == "2"

    grouped_program = compile_ir(
        "groupby((bucket,), close, ewm(cumsum(self_), 3))"
    )
    dot = grouped_program.plot(show=False).to_string()
    assert "GroupBy" in dot
    assert "GroupBy " in dot and " RHS" in dot
    assert "Cumsum" in dot
    assert "Ewm" in dot
    assert "label=self" in dot
    assert "label=rhs" in dot


def test_cpp_stream_runtime_plot_delegates_to_retained_ir() -> None:
    program = compile_ir("xs_rank(ewm(close, 3))")
    runtime = object.__new__(CppStreamRuntime)
    runtime.program = program

    assert runtime.plot(show=False).to_string() == program.plot(show=False).to_string()


def test_plot_display_path_uses_matplotlib_show(monkeypatch) -> None:
    formula = cumsum(var("close"))
    shown: list[bool] = []
    monkeypatch.setattr(pydot.Dot, "create_png", lambda self: _ONE_PIXEL_PNG)
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    formula.plot()

    assert shown == [True]
    plt.close("all")


def test_plot_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="pydot"):
        var("close").plot(backend="unknown", show=False)
