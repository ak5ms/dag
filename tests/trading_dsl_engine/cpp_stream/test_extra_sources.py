from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from trading_dsl_engine.cpp_stream import compile_formula


def test_compile_and_run_ignore_unreferenced_sources(tmp_path: Path) -> None:
    rows, n = 17, 4
    x = np.arange(rows * n, dtype=np.float64).reshape(rows, n)
    data: dict[str, object] = {
        "x": x,
        # Neither object is a valid compatible source. They must never be
        # inspected because the formula does not reference them.
        "unused_invalid": object(),
        "unused_short": np.empty((2, n), dtype=np.float64),
    }
    runtime = compile_formula("x + 1", data)
    assert runtime.input_names == ("x",)
    assert set(runtime.bound_sources or ()) == {"x"}

    output = tmp_path / "ignored_extras.bin"
    runtime.run(data, out_path=output)
    actual = np.memmap(output, mode="r", dtype=np.float64, shape=(rows, n))
    np.testing.assert_array_equal(actual, x + 1.0)


def test_missing_referenced_source_still_errors() -> None:
    x = np.arange(12, dtype=np.float64).reshape(3, 4)
    with pytest.raises(KeyError, match="missing"):
        compile_formula("x + y", {"x": x, "unused": object()})
