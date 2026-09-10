from __future__ import annotations

from pathlib import Path
import shutil
import sys

import numpy as np
import pytest
from scipy.special import ndtri

from trading_dsl_engine.cpp_stream import InputTypeSpec, compile_formula, source
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


def _raw(path: Path, width: int):
    return source(path, input_type=InputTypeSpec("float64", width))


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


def _rank_one_group(row: np.ndarray, lanes: list[int], out: np.ndarray) -> None:
    finite_lanes = [lane for lane in lanes if np.isfinite(row[lane])]
    ordered = sorted(finite_lanes, key=lambda lane: row[lane])
    pos = 0
    while pos < len(ordered):
        upper = pos + 1
        while upper < len(ordered) and row[ordered[upper]] == row[ordered[pos]]:
            upper += 1
        score = ndtri(upper / (len(ordered) + 1.0))
        for lane in ordered[pos:upper]:
            out[lane] = score
        pos = upper


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
    data = {"close": _raw(close_path, cols), "open": _raw(open_path, cols)}

    runtime = compile_formula("xs_rank(ewm(close / open, 21))", data, n_instruments=cols)
    result = runtime.run(out_path=out_path)
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
    minute_key = (np.arange(rows, dtype=np.float64)[:, None] % 4.0) + np.zeros((rows, cols))
    close_path = tmp_path / "close.bin"
    minute_path = tmp_path / "minute.bin"
    out_path = tmp_path / "grouped.bin"
    close.tofile(close_path)
    minute_key.tofile(minute_path)
    data = {"minute_key": _raw(minute_path, cols), "close": _raw(close_path, cols)}

    runtime = compile_formula(
        "groupby((univ([0], [1, 2]), minute_key), close, cumsum(self_))",
        data,
        n_instruments=cols,
        key_cardinalities={"minute_key": 4},
    )
    runtime.run(out_path=out_path)
    actual = np.fromfile(out_path, dtype=np.float64).reshape(rows, cols)

    state: dict[tuple[int, int], float] = {}
    expected = np.empty_like(close)
    for t in range(rows):
        for lane in range(cols):
            key = (lane, int(minute_key[t, lane]))
            state[key] = state.get(key, 0.0) + close[t, lane]
            expected[t, lane] = state[key]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_cpp_stream_composite_groupby_nested_state_capture_and_rank(tmp_path: Path):
    _require_native_compiler()
    rows, cols = 72, 5
    rng = np.random.default_rng(19)
    close = rng.normal(size=(rows, cols)).astype(np.float64)
    open_ = rng.normal(size=(rows, cols)).astype(np.float64)
    key0 = (np.arange(rows, dtype=np.float64)[:, None] % 3.0) + np.zeros((rows, cols))
    key1 = ((np.arange(rows, dtype=np.float64)[:, None] // 2.0) % 2.0) + np.zeros((rows, cols))
    paths = {}
    for name, value in {"close": close, "open": open_, "key0": key0, "key1": key1}.items():
        path = tmp_path / f"{name}.bin"
        value.tofile(path)
        paths[name] = _raw(path, cols)
    out_path = tmp_path / "nested_grouped.bin"

    runtime = compile_formula(
        "groupby((univ([0, 1], [2, 3, 4]), key0, key1), close, xs_rank(ewm(cumsum(self_) + open, 3)))",
        paths,
        n_instruments=cols,
        default_group_capacity=16,
    )
    runtime.run(out_path=out_path)
    actual = np.fromfile(out_path, dtype=np.float64).reshape(rows, cols)

    cumulative: dict[tuple[int, int, int], float] = {}
    ewm_value: dict[tuple[int, int, int], float] = {}
    expected = np.full_like(close, np.nan)
    alpha = 0.5
    partitions = (0, 0, 1, 1, 1)
    for t in range(rows):
        before_rank = np.empty(cols, dtype=np.float64)
        group_members: dict[tuple[int, int, int], list[int]] = {}
        for lane in range(cols):
            dynamic = (int(key0[t, lane]), int(key1[t, lane]))
            state_key = (lane, *dynamic)
            cumulative[state_key] = cumulative.get(state_key, 0.0) + close[t, lane]
            x = cumulative[state_key] + open_[t, lane]
            previous = ewm_value.get(state_key)
            current = x if previous is None else (1.0 - alpha) * previous + alpha * x
            ewm_value[state_key] = current
            before_rank[lane] = current
            cross_key = (partitions[lane], *dynamic)
            group_members.setdefault(cross_key, []).append(lane)
        for lanes in group_members.values():
            _rank_one_group(before_rank, lanes, expected[t])

    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11, equal_nan=True)


def test_formula_list_keeps_independent_shapes_and_shared_cse(tmp_path: Path):
    _require_native_compiler()
    rows, cols = 64, 4
    rng = np.random.default_rng(123)
    x = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        ["ewm(x, 3)", "cat(ewm(x, 3), ewm(x, 3))"],
        {"x": x},
        n_instruments=cols,
    )
    result = runtime.run(out_path=tmp_path / "multi.npy")
    first, second = result.load(mmap_mode=None)

    expected = _reference_ewm(x, 3.0)
    np.testing.assert_allclose(first, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        second,
        np.stack((expected, expected), axis=-1),
        rtol=1e-12,
        atol=1e-12,
    )
    assert first.shape == (rows, cols)
    assert second.shape == (rows, cols, 2)
    assert result.logical_output_shapes == ((cols,), (cols, 2))
    assert sum(type(node.op).__name__ == "EwmOp" for node in runtime.program.nodes) == 1
    assert runtime.plan.scratch_slots == 0

    generated = runtime.generated_cpp.read_text()
    assert generated.count('extern "C" int cpp_stream_run_arrays(') == 1
    assert "stackdsl::OutputOffsetNode<" not in generated
    assert "stackdsl::OutputNode<" not in generated
    assert "stackdsl::OutputSliceDst<0>" in generated
    assert f"stackdsl::OutputSliceDst<{cols}>" in generated


def test_one_element_formula_list_returns_one_element_tuple(tmp_path: Path):
    _require_native_compiler()
    x = np.arange(24, dtype=np.float64).reshape(6, 4)
    runtime = compile_formula(["x + 1"], {"x": x}, n_instruments=4)
    result = runtime.run(out_path=tmp_path / "single-list.npy")

    (actual,) = result.load(mmap_mode=None)
    np.testing.assert_allclose(actual, x + 1.0)
    assert actual.shape == x.shape
    assert result.logical_output_shapes == ((4,),)


def test_cross_sectional_vector_ops_broadcast_scalar_secondary_inputs(tmp_path: Path):
    _require_native_compiler()
    x = np.arange(1, 25, dtype=np.float64).reshape(6, 4)
    runtime = compile_formula(
        [
            "xs_weighted_mean(x, 1)",
            "xs_vector_projection(x, 1)",
            "xs_regression_projection(x, 1)",
        ],
        {"x": x},
        n_instruments=4,
    )
    weighted, projection, regression = runtime.run(
        out_path=tmp_path / "scalar-xs.npy"
    ).load(mmap_mode=None)
    expected_mean = np.mean(x, axis=1, keepdims=True)
    np.testing.assert_allclose(weighted, np.broadcast_to(expected_mean, x.shape))
    np.testing.assert_allclose(projection, np.broadcast_to(expected_mean, x.shape))
    assert np.isnan(regression).all()  # a constant regressor has zero variance


def test_formula_list_can_mix_row_and_final_outputs(tmp_path: Path):
    _require_native_compiler()
    x = np.arange(40, dtype=np.float64).reshape(10, 4)
    runtime = compile_formula(
        ["x + 1", "sum(x, axis=0)"],
        {"x": x},
        n_instruments=4,
    )
    result = runtime.run(out_path=tmp_path / "mixed.npy", threads=4)
    rows, final = result.load(mmap_mode=None)

    np.testing.assert_allclose(rows, x + 1.0)
    np.testing.assert_allclose(final, np.sum(x, axis=0))
    assert rows.shape == (10, 4)
    assert final.shape == (4,)
    assert result.output_mode == "mixed"
    assert runtime.parallel_plan.mode == "serial"
    assert result.threads == 1
