from __future__ import annotations

from pathlib import Path

import numpy as np

from flows.utils import mask
from trading_dsl_engine.base.dsl import var, where
from trading_dsl_engine.cpp_stream import compile_formula


def _read(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    return np.asarray(np.memmap(path, mode="r", dtype=np.float64, shape=shape)).copy()


def test_vector_condition_broadcasts_scalar_branches(tmp_path: Path) -> None:
    rows, n = 257, 9
    condition = np.zeros((rows, n), dtype=np.float64)
    condition[:, ::2] = 1.0
    runtime = compile_formula(
        where(var("condition"), 7.0, -3.0),
        {"condition": condition},
        n_instruments=n,
    )
    assert runtime.program.nodes[runtime.program.output_id].value_type.kind == "vector"
    assert runtime.plan.output_shape == (n,)

    serial_path = tmp_path / "where_serial.bin"
    parallel_path = tmp_path / "where_parallel.bin"
    runtime.run(out_path=serial_path, threads=1)
    result = runtime.run(out_path=parallel_path, threads=2, pin_threads=True)
    expected = np.where(condition != 0.0, 7.0, -3.0)
    np.testing.assert_array_equal(_read(serial_path, (rows, n)), expected)
    np.testing.assert_array_equal(_read(parallel_path, (rows, n)), expected)
    assert result.threads == 2


def test_mask_scalar_value_with_vector_tradability_is_vector(tmp_path: Path) -> None:
    rows, n = 1600, 9
    scalar = np.where(np.arange(rows) < 1440, 1.0, 2.0).astype(np.float64)
    tradable = np.ones((rows, n), dtype=np.float64)
    tradable[1380:1500] = 0.0
    # Disable mask's default forward fill here: this test isolates where's
    # vector-condition/scalar-branch broadcasting rather than history semantics.
    formula = mask(var("scalar"), var("tradable"), fill=None)
    runtime = compile_formula(
        formula,
        {"scalar": scalar, "tradable": tradable},
        n_instruments=n,
    )
    assert runtime.program.nodes[runtime.program.output_id].value_type.kind == "vector"
    assert runtime.plan.output_shape == (n,)

    serial_path = tmp_path / "mask_serial.bin"
    parallel_path = tmp_path / "mask_parallel.bin"
    runtime.run(out_path=serial_path, threads=1)
    runtime.run(out_path=parallel_path, threads=2, pin_threads=True)
    expected = np.where(tradable == 1.0, scalar[:, None], np.nan)
    np.testing.assert_array_equal(_read(serial_path, (rows, n)), expected)
    np.testing.assert_array_equal(_read(parallel_path, (rows, n)), expected)
