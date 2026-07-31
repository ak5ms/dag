from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from scipy.special import ndtri

from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.ir import GroupByOp, compile_ir


def test_neutral_ir_builds_without_jax_flat_types():
    program = compile_ir("xs_rank(ewm(close / open, 21))")
    assert program.input_names == ("close", "open")
    assert type(program.nodes[program.output_id].op).__name__ == "XsRankOp"
    assert not any(type(node.op).__module__.startswith("trading_dsl_engine.jax_flat") for node in program.nodes)


def test_neutral_ir_preserves_univ_plus_dynamic_groupby():
    program = compile_ir("groupby((univ([0], [1, 2]), minute), close, ewm(cumsum(self_), 3))")
    group_node = program.nodes[program.output_id]
    assert isinstance(group_node.op, GroupByOp)
    assert group_node.op.static_groups == ((0,), (1, 2))
    assert group_node.op.n_dynamic_keys == 1
    assert tuple(type(node.op).__name__ for node in group_node.op.inner_program.nodes)[-2:] == ("CumsumOp", "EwmOp")


def _require_native_compiler():
    if sys.platform == "win32" or shutil.which("g++") is None:
        pytest.skip("cpp_stream integration test requires POSIX and g++")


def _reference_ewm(values: np.ndarray, span: float) -> np.ndarray:
    alpha = 2.0 / (span + 1.0)
    old_factor = 1.0 - alpha
    rows, cols = values.shape
    state = np.zeros(cols, dtype=np.float64)
    weight = np.zeros(cols, dtype=np.float64)
    initialized = np.zeros(cols, dtype=bool)
    out = np.empty_like(values)
    for t in range(rows):
        for i in range(cols):
            x = values[t, i]
            observation = np.isfinite(x)
            old_weight = weight[i]
            if initialized[i] and observation:
                old_weight *= old_factor
            if observation:
                if initialized[i]:
                    new_weight = alpha
                    if abs(alpha - 0.5) <= 1e-12:
                        new_weight = 1.0 - old_weight
                    if state[i] != x:
                        state[i] = (old_weight * state[i] + new_weight * x) / (old_weight + new_weight)
                    old_weight = 1.0
                else:
                    state[i] = x
                    initialized[i] = True
                    old_weight = 1.0
            weight[i] = old_weight
            out[t, i] = state[i] if initialized[i] else np.nan
    return out


def _reference_rank(values: np.ndarray) -> np.ndarray:
    out = np.full_like(values, np.nan)
    for t, row in enumerate(values):
        finite = np.isfinite(row)
        lanes = np.flatnonzero(finite)
        ordered = lanes[np.argsort(row[lanes], kind="stable")]
        pos = 0
        count = len(ordered)
        while pos < count:
            upper = pos + 1
            while upper < count and row[ordered[upper]] == row[ordered[pos]]:
                upper += 1
            out[t, ordered[pos:upper]] = ndtri(upper / (count + 1.0))
            pos = upper
    return out


def test_cpp_stream_mmap_formula_matches_reference(tmp_path: Path):
    _require_native_compiler()
    rows, cols = 128, 5
    rng = np.random.default_rng(42)
    close = rng.lognormal(4.0, 0.1, (rows, cols)).astype(np.float64)
    open_ = rng.lognormal(4.0, 0.1, (rows, cols)).astype(np.float64)
    close[7, 2] = np.nan
    open_[11, 1] = 0.0
    close_path = tmp_path / "close.bin"
    open_path = tmp_path / "open.bin"
    out_path = tmp_path / "alpha.bin"
    close.tofile(close_path)
    open_.tofile(open_path)

    runtime = compile_formula("xs_rank(ewm(close / open, 21))", n_instruments=cols)
    result = runtime.run_files({"close": close_path, "open": open_path}, out_path=out_path)
    actual = np.fromfile(out_path, dtype=np.float64).reshape(rows, cols)
    ratio = np.divide(close, open_, out=np.full_like(close, np.nan), where=open_ != 0.0)
    expected = _reference_rank(_reference_ewm(ratio, 21.0))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=True)
    assert result.rows == rows


def test_cpp_stream_dense_mixed_groupby_matches_reference(tmp_path: Path):
    _require_native_compiler()
    rows, cols = 96, 3
    rng = np.random.default_rng(7)
    close = rng.normal(size=(rows, cols)).astype(np.float64)
    minute = (np.arange(rows, dtype=np.float64)[:, None] % 4.0) + np.zeros((rows, cols))
    close_path = tmp_path / "close.bin"
    minute_path = tmp_path / "minute.bin"
    out_path = tmp_path / "grouped.bin"
    close.tofile(close_path)
    minute.tofile(minute_path)

    runtime = compile_formula(
        "groupby((univ([0], [1, 2]), minute), close, cumsum(self_))",
        n_instruments=cols,
        key_cardinalities={"minute": 4},
    )
    runtime.run_files({"minute": minute_path, "close": close_path}, out_path=out_path)
    actual = np.fromfile(out_path, dtype=np.float64).reshape(rows, cols)

    state: dict[tuple[int, int], float] = {}
    expected = np.empty_like(close)
    for t in range(rows):
        for lane in range(cols):
            key = (lane, int(minute[t, lane]))
            state[key] = state.get(key, 0.0) + close[t, lane]
            expected[t, lane] = state[key]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
