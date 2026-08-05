from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest

from trading_dsl_engine.cpp_stream import InputTypeSpec, compile_formula, source
from trading_dsl_engine.cpp_stream.python.dynamic_ewm import DynamicEwmOp
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir


def _require_native_compiler() -> None:
    if sys.platform == "win32" or shutil.which("g++") is None:
        pytest.skip("cpp_stream integration test requires POSIX and g++")


def _raw(path: Path, width: int):
    return source(path, input_type=InputTypeSpec("float64", width))


def _reference(values: np.ndarray, spans: np.ndarray) -> np.ndarray:
    rows, cols = values.shape
    state = np.zeros(cols, dtype=np.float64)
    initialized = np.zeros(cols, dtype=bool)
    out = np.full_like(values, np.nan)
    for row in range(rows):
        for lane in range(cols):
            span = spans[row, lane]
            if not np.isfinite(span) or span <= 0.0:
                continue
            x = values[row, lane]
            if np.isfinite(x):
                alpha = 2.0 / (span + 1.0)
                if initialized[lane]:
                    state[lane] = alpha * (x - state[lane]) + state[lane]
                else:
                    state[lane] = x
                    initialized[lane] = True
            if initialized[lane]:
                out[row, lane] = state[lane]
    return out


def test_dynamic_ewm_ir_has_span_child() -> None:
    program = compile_ir("ewm(x, span)")
    node = program.nodes[program.output_id]
    assert isinstance(node.op, DynamicEwmOp)
    assert len(node.child_ids) == 2
    assert program.input_names == ("x", "span")


def test_cpp_stream_dynamic_ewm_matches_variable_alpha_reference(tmp_path: Path) -> None:
    _require_native_compiler()
    rows, cols = 257, 9
    rng = np.random.default_rng(42)
    values = rng.normal(size=(rows, cols)).astype(np.float64)
    spans = rng.uniform(2.0, 80.0, size=(rows, cols)).astype(np.float64)
    values[11, 3] = np.nan
    spans[23, 4] = np.nan
    spans[37, 5] = 0.0

    x_path = tmp_path / "x.bin"
    span_path = tmp_path / "span.bin"
    out_path = tmp_path / "out.bin"
    values.tofile(x_path)
    spans.tofile(span_path)

    runtime = compile_formula(
        "ewm(x, span)",
        {"x": _raw(x_path, cols), "span": _raw(span_path, cols)},
        n_instruments=cols,
    )
    runtime.run(out_path=out_path)
    actual = np.fromfile(out_path, dtype=np.float64).reshape(rows, cols)
    expected = _reference(values, spans)
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=2e-14,
        atol=2e-14,
        equal_nan=True,
    )


def test_cpp_stream_dynamic_row_scalar_span_broadcasts(tmp_path: Path) -> None:
    _require_native_compiler()
    rows, cols = 129, 4
    rng = np.random.default_rng(7)
    values = rng.normal(size=(rows, cols)).astype(np.float64)
    row_span = rng.uniform(3.0, 40.0, size=rows).astype(np.float64)
    spans = np.broadcast_to(row_span[:, None], values.shape)

    x_path = tmp_path / "x.bin"
    span_path = tmp_path / "span.bin"
    out_path = tmp_path / "out.bin"
    values.tofile(x_path)
    row_span.tofile(span_path)

    runtime = compile_formula(
        "ewm(x, span)",
        {
            "x": _raw(x_path, cols),
            "span": source(
                span_path,
                input_type=InputTypeSpec("float64", 1),
            ),
        },
        n_instruments=cols,
    )
    runtime.run(out_path=out_path)
    actual = np.fromfile(out_path, dtype=np.float64).reshape(rows, cols)
    np.testing.assert_allclose(
        actual,
        _reference(values, spans),
        rtol=2e-14,
        atol=2e-14,
        equal_nan=True,
    )
