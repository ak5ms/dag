from __future__ import annotations

from pathlib import Path
import builtins
import re

import numpy as np

from trading_dsl_engine.base.dsl import cumsum, groupby, self_, var
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.cpp_stream import compile_formula
from trading_dsl_engine.cpp_stream.python.runtime import FormulaResults


def test_nested_formula_mapping_load_map_and_flatten(tmp_path: Path):
    rows, cols = 8, 3
    x = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    runtime = compile_formula(
        {"signals": {"raw": "x", "twice": "x * 2"}, "summary": "sum(x, axis=0)"},
        {"x": x},
        n_instruments=cols,
    )

    loaded = runtime.run(out_path=tmp_path / "nested.npy").load(mmap_mode=None)
    assert isinstance(loaded, FormulaResults)
    assert isinstance(loaded["signals"], FormulaResults)
    np.testing.assert_array_equal(loaded["signals"]["raw"], x)
    np.testing.assert_array_equal(loaded["signals"]["twice"], x * 2)
    np.testing.assert_array_equal(loaded["summary"], x.sum(axis=0))

    shapes = loaded.map(np.shape)
    assert shapes["signals"]["raw"] == (rows, cols)
    assert shapes["summary"] == (cols,)
    assert list(loaded.flatten()) == [
        ("signals", "raw"),
        ("signals", "twice"),
        ("summary",),
    ]

from trading_dsl_engine.base.dsl import (
    Ridge,
    cat,
    cumsum,
    get_beta,
    groupby,
    self_,
    var,
)
def test_ridge_beta_is_invariant_to_feature_formula_order(tmp_path: Path):
    rows, cols = 96, 5
    rng = np.random.default_rng(20260909)
    x = rng.normal(size=(rows, cols)).astype(np.float64)
    y = rng.normal(size=(rows, cols)).astype(np.float64)
    target = (1.75 * x - 0.4 * y + rng.normal(scale=0.03, size=x.shape)).astype(
        np.float64
    )
    first = (var("x") + var("y")).ewm(span=7)
    second = (var("x") - var("y")).rolling_mean(5)

    def beta(features):
        model = Ridge(
            cat(*features),
            y=var("target"),
            weights=1,
            hl=12,
            lambda_=0.05,
            nonneg=True,
        )
        # Include shared expressions as earlier public roots, like a real formula
        # mapping, so root traversal order cannot alter the Ridge transition.
        return {"first": first, "second": second, "beta": get_beta(model)}

    data = {"x": x, "y": y, "target": target}
    forward = compile_formula(beta((first, second)), data, n_instruments=cols).run(
        out_path=tmp_path / "forward.npy"
    ).load(mmap_mode=None)
    reverse = compile_formula(beta((second, first)), data, n_instruments=cols).run(
        out_path=tmp_path / "reverse.npy"
    ).load(mmap_mode=None)

    np.testing.assert_allclose(forward["first"], reverse["first"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(forward["second"], reverse["second"], rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        forward["beta"], reverse["beta"][..., ::-1], rtol=1e-12, atol=1e-12
    )
def test_promoted_row_output_remains_readable_by_final_projection(tmp_path: Path):
    rows, cols = 48, 4
    rng = np.random.default_rng(20260819)
    x = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        ["ewm(x, 3)", "emit(ewm(x, 3))"],
        {"x": x},
        n_instruments=cols,
    )
    result = runtime.run(out_path=tmp_path / "mixed-row-final.npy")
    public_ewm, final = result.load(mmap_mode=None)

    np.testing.assert_allclose(final, public_ewm[-1], rtol=1e-12, atol=1e-12)
    # The duplicate final projection does not participate in hot computation, so
    # the row EWM can still own public storage directly while EmitLast snapshots
    # that row output through the ordinary row loop.
    assert runtime.plan.scratch_slots == 0
    assert [stage.kind for stage in runtime.plan.stages][-1] == "emit_last"
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::PackedOutputTensorSource<" in generated
    assert "stackdsl::EmitLastNode<" in generated
    assert "ctx.row_output" in generated


def test_emit_last_of_promoted_row_output_is_nan_on_empty_input(tmp_path: Path):
    cols = 4
    x = np.empty((0, cols), dtype=np.float64)

    runtime = compile_formula(
        ["ewm(x, 3)", "emit(ewm(x, 3))"],
        {"x": x},
        n_instruments=cols,
    )
    result = runtime.run(out_path=tmp_path / "empty-mixed-row-final.npy")
    public_ewm, final = result.load(mmap_mode=None)

    assert public_ewm.shape == (0, cols)
    assert final.shape == (cols,)
    assert np.isnan(final).all()
    assert [stage.kind for stage in runtime.plan.stages][-1] == "emit_last"
    assert "stackdsl::EmitLastNode<" in runtime.generated_cpp.read_text()


def test_hot_groupby_consumer_keeps_public_subgraph_in_scratch(tmp_path: Path):
    rows, cols = 40, 4
    rng = np.random.default_rng(7)
    x = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        [
            "ewm(x, 3)",
            "groupby(univ([0], [1], [2], [3]), ewm(x, 3), cumsum(self_))",
        ],
        {"x": x},
        n_instruments=cols,
    )
    result = runtime.run(out_path=tmp_path / "public-feed.npy")
    public_ewm, grouped = result.load(mmap_mode=None)

    np.testing.assert_allclose(
        grouped,
        np.cumsum(public_ewm, axis=0),
        rtol=1e-12,
        atol=1e-12,
    )
    assert runtime.plan.scratch_slots == 1
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::PackedOutputSrc<" not in generated
    assert "stackdsl::OutputNode<" not in generated


def test_lazy_public_root_dependency_fuses_ewm_output_epilogues(tmp_path: Path):
    rows, cols = 40, 4
    rng = np.random.default_rng(11)
    x = rng.normal(size=(rows, cols)).astype(np.float64)
    y = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        [
            "ewm(x, span=32)",
            "cat(ewm(x, span=32), ewm(y, span=32))",
        ],
        {"x": x, "y": y},
        n_instruments=cols,
    )
    result = runtime.run(out_path=tmp_path / "public-lazy-dependency.npy")
    public_x, pair = result.load(mmap_mode=None)

    np.testing.assert_allclose(pair[..., 0], public_x, rtol=0.0, atol=0.0)
    # Both requested outputs are projections of the same EWM bundle. The output
    # pass folds Copy/Cat into native bundle epilogues, so logical member slot ids
    # are only component labels and RowContext allocates no scalar scratch arrays.
    assert [stage.kind for stage in runtime.plan.stages] == ["ewm_bundle"]
    assert runtime.plan.scratch_slots == 0
    assert runtime.plan.scratch_slots.counts == (0, 0, 0, 0, 0, 0)
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::EwmDiscardDst" in generated
    assert "stackdsl::EwmEpilogueBinding" in generated
    assert "stackdsl::ScalarScratchLayout<\n    0, 0, 0," in generated


def test_duplicate_final_roots_keep_shared_value_scratch_backed(tmp_path: Path):
    rows, cols = 48, 4
    x = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)

    runtime = compile_formula(
        ["sum(x, axis=0)", "sum(x, axis=0)"],
        {"x": x},
        n_instruments=cols,
    )
    result = runtime.run(out_path=tmp_path / "duplicate-final.npy")
    first, second = result.load(mmap_mode=None)
    expected = np.sum(x, axis=0)

    np.testing.assert_allclose(first, expected, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(second, expected, rtol=0.0, atol=0.0)
    # CSE keeps one temporal accumulator. Because two public outputs need that
    # final value, it remains scratch-backed and both final projections read the
    # same scratch slot; PackedOutputSrc is row-region-only and must not appear.
    assert [stage.kind for stage in runtime.plan.stages].count("reduce") == 1
    assert runtime.plan.scratch_slots == 1
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::PackedOutputSrc<" not in generated


def test_scalar_scratch_is_compacted_per_native_dtype(tmp_path: Path):
    rows, cols = 32, 4
    key = np.tile(np.arange(cols, dtype=np.int64), (rows, 1))
    x = np.arange(rows * cols, dtype=np.float64).reshape(rows, cols)
    formula = groupby(
        Key(var("key") + 1, num_keys=8, dtype="int64"),
        var("x") + 1.0,
        cumsum(self_),
    )

    runtime = compile_formula(
        formula,
        {"key": key, "x": x},
        n_instruments=cols,
    )
    result = runtime.run(out_path=tmp_path / "typed-scratch.npy")
    actual = result.load(mmap_mode=None)
    np.testing.assert_allclose(actual, np.cumsum(x + 1.0, axis=0))

    # The int64 key expression and float64 group feed each use local scalar slot
    # zero in their own native storage. Total logical scalar slots is still two.
    assert runtime.plan.scratch_slots == 2
    scalar_dests = [
        (candidate.dtype, candidate.out.slot)
        for stage in runtime.plan.stages
        for candidate in (stage, *stage.members, *stage.epilogues)
        if candidate.out.slot is not None
        and candidate.out.slot >= 0
        and not candidate.out.matrix
        and not candidate.out.tensor
    ]
    assert ("float64", 0) in scalar_dests
    assert ("int64", 0) in scalar_dests

    generated = runtime.generated_cpp.read_text()
    assert re.search(
        r"using ScalarScratch = stackdsl::ScalarScratchLayout<\s*1, 0, 1,\s*0, 0, 0\s*>;",
        generated,
    )


def test_grouped_ewm_epilogue_uses_no_inner_scalar_scratch(tmp_path: Path):
    rows, cols = 40, 4
    rng = np.random.default_rng(29)
    x = rng.normal(size=(rows, cols)).astype(np.float64)

    runtime = compile_formula(
        "groupby(univ([0], [1], [2], [3]), x, "
        "ewm(self_, 3) + ewm(self_ + 1, 3))",
        {"x": x},
        n_instruments=cols,
    )
    result = runtime.run(out_path=tmp_path / "grouped-ewm-epilogue.npy")
    actual = result.load(mmap_mode=None)

    reference = compile_formula(
        "ewm(x, 3) + ewm(x + 1, 3)",
        {"x": x},
        n_instruments=cols,
    ).run(out_path=tmp_path / "grouped-ewm-reference.npy").load(mmap_mode=None)
    np.testing.assert_allclose(actual, reference, rtol=1e-12, atol=1e-12)

    group = next(
        stage.group for stage in runtime.plan.stages if stage.group is not None
    )
    assert [stage.kind for stage in group.inner.stages] == ["ewm_bundle"]
    assert group.inner.scratch_slots == 0
    assert group.inner.scratch_slots.counts == (0, 0, 0, 0, 0, 0)
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::EwmDiscardDst" in generated
    assert "stackdsl::EwmEpilogueBinding" in generated


def test_ridge_beta_is_invariant_to_output_mapping_key_order(tmp_path: Path) -> None:
    rows, cols = 96, 4
    rng = np.random.default_rng(20260908)
    x1 = rng.normal(size=(rows, cols))
    x2 = rng.normal(size=(rows, cols))
    y = 0.6 * x1 - 0.25 * x2 + rng.normal(scale=0.05, size=(rows, cols))
    paths = {
        "x1": tmp_path / "x1.npy",
        "x2": tmp_path / "x2.npy",
        "y": tmp_path / "y.npy",
    }
    np.save(paths["x1"], x1)
    np.save(paths["x2"], x2)
    np.save(paths["y"], y)

    beta_first = compile_formula(
        {
            "beta": "get_beta(Ridge(cat(x1, x2), y=y, hl=8, lambda_=0.05))",
            "ic_like": "cat(ewm(x1, 3), ewm(x2, 5))",
        },
        paths,
        n_instruments=cols,
    )
    beta_last = compile_formula(
        {
            "ic_like": "cat(ewm(x1, 3), ewm(x2, 5))",
            "beta": "get_beta(Ridge(cat(x1, x2), y=y, hl=8, lambda_=0.05))",
        },
        paths,
        n_instruments=cols,
    )
    first = beta_first.run(out_path=tmp_path / "beta-first.npy").load(mmap_mode=None)
    last = beta_last.run(out_path=tmp_path / "beta-last.npy").load(mmap_mode=None)
    np.testing.assert_allclose(first["beta"], last["beta"], rtol=1e-12, atol=1e-12)

    ridge_first = builtins.next(
        index
        for index, stage in enumerate(beta_first.plan.stages)
        if stage.kind == "ridge"
    )
    ridge_last = builtins.next(
        index
        for index, stage in enumerate(beta_last.plan.stages)
        if stage.kind == "ridge"
    )
    assert ridge_first != ridge_last
