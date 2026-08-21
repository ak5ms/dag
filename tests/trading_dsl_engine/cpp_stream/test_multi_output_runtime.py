from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from trading_dsl_engine.base.dsl import cumsum, groupby, self_, var
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.cpp_stream import compile_formula, run_many


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


def test_native_batch_matches_serial_mixed_output_runtimes(tmp_path: Path):
    rows, cols = 512, 5
    rng = np.random.default_rng(42)
    data = {"x": rng.normal(size=(rows, cols))}
    x = var("x")
    runtimes = (
        compile_formula(
            [x + 1.0, (x + 1.0).mean(axis=0)],
            data,
            n_instruments=cols,
        ),
        compile_formula(
            [x * 2.0, (x * 2.0).std(axis=0)],
            data,
            n_instruments=cols,
        ),
    )
    serial = tuple(
        runtime.run(out_path=tmp_path / f"serial-{index}.npy", threads=1)
        for index, runtime in enumerate(runtimes)
    )
    native = run_many(
        runtimes,
        out_paths=(tmp_path / "native-0.npy", tmp_path / "native-1.npy"),
        workers=2,
        threads_per_runtime=1,
    )

    assert 1 <= native.workers <= 2
    for expected_result, actual_result in zip(serial, native.results):
        expected = expected_result.load(mmap_mode=None)
        actual = actual_result.load(mmap_mode=None)
        assert isinstance(expected, tuple)
        assert isinstance(actual, tuple)
        for expected_value, actual_value in zip(expected, actual):
            np.testing.assert_allclose(
                actual_value,
                expected_value,
                rtol=1e-13,
                atol=1e-13,
                equal_nan=True,
            )


def test_native_batch_rejects_duplicate_output_paths(tmp_path: Path):
    data = {"x": np.arange(16, dtype=np.float64).reshape(8, 2)}
    runtime = compile_formula(var("x") + 1.0, data, n_instruments=2)
    shared = tmp_path / "shared.npy"
    with pytest.raises(ValueError, match="out_paths must be distinct"):
        run_many(
            (runtime, runtime),
            out_paths=(shared, shared),
            workers=2,
        )


def test_gp_search_pure_walk_forward_and_batching_contracts():
    import ast
    import math
    import sys
    import types
    from dataclasses import dataclass
    from statistics import NormalDist

    script = Path(__file__).resolve().parents[3] / "scripts" / "run_gp_alpha_search.py"
    source = script.read_text()
    compile(source, str(script), "exec")
    assert "ThreadPoolExecutor" not in source
    assert "sources_all" not in source
    assert "run_many(" in source
    assert "train_end=(folds[-1].train_end if folds else None)" in source

    tree = ast.parse(source)
    wanted_classes = {"WalkForwardFold", "SharpeComparison", "_CandidateSpec"}
    wanted_functions = {
        "build_anchored_walk_forward",
        "_sharpe_standard_error",
        "compare_sharpes",
        "_make_microbatches",
        "_portfolio_cumulative",
    }
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            nodes.append(node)
        elif isinstance(node, ast.ClassDef) and node.name in wanted_classes:
            nodes.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in wanted_functions:
            nodes.append(node)
    selected = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(selected)

    module_name = "_gp_search_pure_contracts"
    module = types.ModuleType(module_name)
    module.__dict__.update(
        {
            "__name__": module_name,
            "dataclass": dataclass,
            "math": math,
            "NormalDist": NormalDist,
            "np": np,
            "_NORMAL": NormalDist(),
            "FITNESS_BATCH_SIZE": 8,
            "FITNESS_TASKS_PER_WORKER": 1,
        }
    )
    sys.modules[module_name] = module
    try:
        exec(compile(selected, str(script), "exec"), module.__dict__)
        folds = module.build_anchored_walk_forward(
            1_000,
            folds=3,
            validation_fraction=0.10,
        )
        assert [
            (fold.train_end, fold.validation_start, fold.validation_end)
            for fold in folds
        ] == [
            (700, 700, 800),
            (800, 800, 900),
            (900, 900, 1_000),
        ]
        comparable = module.compare_sharpes(
            1.0,
            0.8,
            in_sample_rows=1_000,
            out_of_sample_rows=1_000,
            min_ratio=0.5,
            alpha=0.05,
            require_positive=True,
        )
        decayed = module.compare_sharpes(
            1.0,
            0.3,
            in_sample_rows=1_000,
            out_of_sample_rows=1_000,
            min_ratio=0.5,
            alpha=0.05,
            require_positive=True,
        )
        assert comparable.passed
        assert not decayed.passed
        items = [
            module._CandidateSpec(str(index), object(), float(index + 1))
            for index in range(64)
        ]
        batches = module._make_microbatches(items, workers=64)
        assert len(batches) == 8
        assert all(len(batch) == 8 for batch in batches)
    finally:
        sys.modules.pop(module_name, None)
