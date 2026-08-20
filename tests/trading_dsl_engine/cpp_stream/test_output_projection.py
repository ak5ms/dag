from __future__ import annotations

from pathlib import Path

import numpy as np

from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.python.frontend import compile_ir
from trading_dsl_engine.cpp_stream.python.outputs import build_output_layout
from trading_dsl_engine.ir.types import fixed


EXPENSIVE = (
    "sqrt(abs(sin(x * 1.000001 + y * 0.999999))) "
    "+ tanh((x - y) * (x + y)) "
    "+ exp(abs(x - y) * -1.0)"
)
LAZY_SUBGRAPH = "sin(x * 1.000001 + y * 0.999999) + tanh(x - y)"
LAZY_PARENT = f"sqrt(abs({LAZY_SUBGRAPH}))"


def _public_stages(runtime):
    return [
        stage
        for stage in runtime.plan.stages
        if stage.out.slot is not None and stage.out.slot < 0
    ]


def _source_has_kind(source, kind: str) -> bool:
    return source.kind == kind or any(
        _source_has_kind(part, kind) for part in source.parts
    )


def test_fixed_extent_equal_to_n_is_not_lane_partitionable() -> None:
    program = compile_ir("x", input_value_types={"x": fixed(4)})
    layout = build_output_layout(program, 4)

    assert layout.outputs[0].shape == (4,)
    assert layout.outputs[0].size == 4
    assert layout.outputs[0].lane_partitionable is False
    assert layout.row_lane_partitionable is False


def test_parent_before_lazy_subgraph_reorders_only_execution(
    tmp_path: Path,
) -> None:
    rows, cols = 32, 4
    rng = np.random.default_rng(20260825)
    x = rng.normal(size=(rows, cols)).astype(np.float64)
    y = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        [LAZY_PARENT, LAZY_SUBGRAPH],
        {"x": x, "y": y},
        n_instruments=cols,
    )
    parent, subgraph = runtime.run(
        out_path=tmp_path / "parent-before-lazy-subgraph.npy"
    ).load(mmap_mode=None)

    np.testing.assert_allclose(
        parent,
        np.sqrt(np.abs(subgraph)),
        rtol=1e-15,
        atol=1e-15,
    )
    public = _public_stages(runtime)
    # API order remains parent then subgraph in RunResult.load(), but execution
    # writes the subgraph's later packed offset first so the parent can reuse it.
    assert [stage.out.slot for stage in public] == [-(cols + 1), -1]
    assert _source_has_kind(public[1].inputs[0], "packed_output")


def test_emit_before_lazy_row_output_reuses_requested_row_storage(
    tmp_path: Path,
) -> None:
    rows, cols = 32, 4
    rng = np.random.default_rng(20260826)
    x = rng.normal(size=(rows, cols)).astype(np.float64)
    y = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        [f"emit({LAZY_SUBGRAPH})", LAZY_SUBGRAPH],
        {"x": x, "y": y},
        n_instruments=cols,
    )
    final, row_values = runtime.run(
        out_path=tmp_path / "emit-before-lazy-row.npy"
    ).load(mmap_mode=None)

    np.testing.assert_allclose(final, row_values[-1], rtol=0.0, atol=0.0)
    terminal = runtime.plan.stages[-2:]
    assert [stage.kind for stage in terminal] == ["copy", "emit_last"]
    assert _source_has_kind(terminal[1].inputs[0], "packed_output")


def test_duplicate_lazy_roots_reuse_first_packed_output(tmp_path: Path) -> None:
    rows, cols = 48, 4
    rng = np.random.default_rng(20260820)
    x = rng.normal(size=(rows, cols)).astype(np.float64)
    y = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        [EXPENSIVE, EXPENSIVE],
        {"x": x, "y": y},
        n_instruments=cols,
    )
    first, second = runtime.run(
        out_path=tmp_path / "duplicate-lazy.npy"
    ).load(mmap_mode=None)

    np.testing.assert_allclose(first, second, rtol=0.0, atol=0.0)
    public = _public_stages(runtime)
    assert [stage.kind for stage in public] == ["copy", "copy"]
    assert public[0].inputs[0].kind != "packed_output"
    assert public[1].inputs[0].kind == "packed_output"
    assert runtime.plan.scratch_slots == 0


def test_duplicate_cat_roots_copy_first_packed_tensor(tmp_path: Path) -> None:
    rows, cols = 32, 4
    rng = np.random.default_rng(20260821)
    x = rng.normal(size=(rows, cols)).astype(np.float64)
    y = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        ["cat(x, y)", "cat(x, y)"],
        {"x": x, "y": y},
        n_instruments=cols,
    )
    first, second = runtime.run(
        out_path=tmp_path / "duplicate-cat.npy"
    ).load(mmap_mode=None)

    expected = np.stack((x, y), axis=-1)
    np.testing.assert_allclose(first, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second, expected, rtol=0.0, atol=0.0)
    public = _public_stages(runtime)
    assert [stage.kind for stage in public] == ["cat", "tensor_copy"]
    assert public[1].inputs[0].kind == "packed_output"
    assert runtime.plan.scratch_slots == 0


def test_single_ewm_fanout_uses_direct_output_not_singleton_bundle(
    tmp_path: Path,
) -> None:
    rows, cols = 40, 4
    rng = np.random.default_rng(20260822)
    x = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        ["ewm(x, 3)", "cat(ewm(x, 3), ewm(x, 3))"],
        {"x": x},
        n_instruments=cols,
    )
    first, pair = runtime.run(
        out_path=tmp_path / "single-ewm-fanout.npy"
    ).load(mmap_mode=None)

    np.testing.assert_allclose(pair[..., 0], first, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pair[..., 1], first, rtol=0.0, atol=0.0)
    assert runtime.plan.scratch_slots == 0
    assert not any(
        stage.kind == "ewm_bundle" and len(stage.members) == 1
        for stage in runtime.plan.stages
    )
    assert "stackdsl::PackedOutputSrc<" in runtime.generated_cpp.read_text()


def test_projection_rewrite_recompacts_released_scratch_slots(
    tmp_path: Path,
) -> None:
    rows, cols = 32, 4
    rng = np.random.default_rng(20260823)
    x = rng.normal(size=(rows, cols)).astype(np.float64)
    y = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        [
            "ewm(x, 3)",
            "cat(ewm(x, 3), ewm(x, 3))",
            "xs_rank(y)",
            "xs_rank(y) + 1",
        ],
        {"x": x, "y": y},
        n_instruments=cols,
    )
    ewm_value, pair, rank, shifted = runtime.run(
        out_path=tmp_path / "recompact-public-scratch.npy"
    ).load(mmap_mode=None)

    np.testing.assert_allclose(pair[..., 0], ewm_value, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(pair[..., 1], ewm_value, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(shifted, rank + 1.0, rtol=0.0, atol=0.0)
    positive_scalar_slots = {
        int(candidate.out.slot)
        for stage in runtime.plan.stages
        for candidate in (stage, *stage.members, *stage.epilogues)
        if candidate.out.slot is not None
        and candidate.out.slot >= 0
        and not candidate.out.matrix
        and not candidate.out.tensor
    }
    assert positive_scalar_slots == {0}
    assert runtime.plan.scratch_slots == 1


def test_ewm_component_labels_do_not_allocate_physical_scratch(
    tmp_path: Path,
) -> None:
    rows, cols = 32, 4
    rng = np.random.default_rng(20260824)
    x = rng.normal(size=(rows, cols)).astype(np.float64)
    y = rng.normal(size=(rows, cols)).astype(np.float64)
    z = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        [
            "cat(ewm(x, 3), ewm(y, 3))",
            "xs_rank(z)",
            "xs_rank(z) + 1",
        ],
        {"x": x, "y": y, "z": z},
        n_instruments=cols,
    )
    pair, rank, shifted = runtime.run(
        out_path=tmp_path / "ewm-labels-not-scratch.npy"
    ).load(mmap_mode=None)

    np.testing.assert_allclose(shifted, rank + 1.0, rtol=0.0, atol=0.0)
    assert pair.shape == (rows, cols, 2)
    bundle = next(stage for stage in runtime.plan.stages if stage.kind == "ewm_bundle")
    rank_stage = next(stage for stage in runtime.plan.stages if stage.kind == "xs_rank")
    assert bundle.epilogues
    assert rank_stage.out.slot == 0
    assert runtime.plan.scratch_slots == 1
    assert runtime.plan.scratch_slots.counts == (1, 0, 0, 0, 0, 0)


def test_integral_descendant_does_not_round_trip_through_float_output(
    tmp_path: Path,
) -> None:
    rows, cols = 8, 4
    base = 1 << 60
    x = (
        base + np.arange(1, rows * cols + 1, dtype=np.int64)
    ).reshape(rows, cols)

    runtime = compile_formula(
        ["x", "x % 7"],
        {"x": x},
        n_instruments=cols,
    )
    public_x, remainder = runtime.run(
        out_path=tmp_path / "integral-public-descendant.npy"
    ).load(mmap_mode=None)

    np.testing.assert_allclose(public_x, x.astype(np.float64), rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        remainder,
        (x % 7).astype(np.float64),
        rtol=0.0,
        atol=0.0,
    )
    # The first public output is float64 storage, but modulo must still consume
    # the original int64 input so values above 2**53 retain their low bits.
    assert "stackdsl::PackedOutputSrc<" not in runtime.generated_cpp.read_text()
