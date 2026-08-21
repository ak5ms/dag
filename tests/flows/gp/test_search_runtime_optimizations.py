from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from flows.utils import ewm_std
from trading_dsl_engine.base.dsl import ffill, shift, var, where
from trading_dsl_engine.cpp_stream import compile_formula


_SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "run_gp_alpha_search.py"


def _search_module():
    name = "_run_gp_alpha_search_test_module"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cost_balanced_batches_are_bounded_and_cover_every_candidate(monkeypatch):
    search = _search_module()
    monkeypatch.setattr(search, "FITNESS_BATCH_SIZE", 3)
    monkeypatch.setattr(search, "FITNESS_TASKS_PER_WORKER", 1)
    costs = (100.0, 80.0, 20.0, 10.0, 5.0, 5.0)
    items = [
        search._CandidateSpec(str(index), object(), cost)
        for index, cost in enumerate(costs)
    ]

    batches = search._make_microbatches(items, workers=2)
    assert sorted(item.key for batch in batches for item in batch) == [
        str(index) for index in range(len(items))
    ]
    assert all(1 <= len(batch) <= 3 for batch in batches)
    loads = [sum(item.estimated_cost for item in batch) for batch in batches]
    assert max(loads) - min(loads) <= 20.0

    serial_batches = search._make_microbatches(items, workers=1)
    assert [len(batch) for batch in serial_batches] == [3, 3]


def test_high_core_count_does_not_create_one_compile_unit_per_candidate(monkeypatch):
    search = _search_module()
    monkeypatch.setattr(search, "FITNESS_BATCH_SIZE", 8)
    monkeypatch.setattr(search, "FITNESS_TASKS_PER_WORKER", 1)
    items = [
        search._CandidateSpec(str(index), object(), float(index + 1))
        for index in range(64)
    ]

    batches = search._make_microbatches(items, workers=64)
    assert len(batches) == 8
    assert all(len(batch) == 8 for batch in batches)


def test_interval_union_counts_overlapping_compile_wall_once():
    search = _search_module()
    stages = [
        {"start": 1.0, "end": 4.0},
        {"start": 2.0, "end": 5.0},
        {"start": 7.0, "end": 8.5},
    ]
    assert search._interval_union_seconds(stages, "start", "end") == 5.5


def test_anchored_walk_forward_ends_at_last_row_and_expands_training():
    search = _search_module()
    folds = search.build_anchored_walk_forward(
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
    assert all(fold.train_start == 0 for fold in folds)


def test_anchored_walk_forward_rejects_insufficient_training_rows():
    search = _search_module()
    with pytest.raises(ValueError, match="does not fit"):
        search.build_anchored_walk_forward(
            100,
            folds=4,
            validation_fraction=0.25,
            min_train_rows=20,
        )


def test_two_sharpe_noninferiority_test_accepts_comparable_and_rejects_decay():
    search = _search_module()
    comparable = search.compare_sharpes(
        1.0,
        0.8,
        in_sample_rows=1_000,
        out_of_sample_rows=1_000,
        min_ratio=0.5,
        alpha=0.05,
        require_positive=True,
    )
    decayed = search.compare_sharpes(
        1.0,
        0.3,
        in_sample_rows=1_000,
        out_of_sample_rows=1_000,
        min_ratio=0.5,
        alpha=0.05,
        require_positive=True,
    )
    assert comparable.passed
    assert comparable.noninferiority_p >= 0.95
    assert not decayed.passed
    assert decayed.noninferiority_p < 0.95
    assert 0.0 <= comparable.equality_two_sided_p <= 1.0


def test_pool_batch_loader_removes_only_verified_lane_broadcast():
    search = _search_module()

    class Result:
        def __init__(self, values):
            self.values = values

        def load(self, *, mmap_mode=None):
            del mmap_mode
            return self.values

    expected = np.array([1.0, 2.0, 3.0])
    broadcast = np.broadcast_to(expected, (9, 3)).copy()
    np.testing.assert_array_equal(
        search._pool_batch_values(Result(broadcast), 3),
        expected,
    )

    broadcast[4, 1] = 9.0
    with pytest.raises(RuntimeError, match="differ across instrument lanes"):
        search._pool_batch_values(Result(broadcast), 3)


def test_materialized_fitness_invariants_preserve_alpha_pnl(tmp_path, monkeypatch):
    search = _search_module()
    monkeypatch.setenv("TRADING_DSL_ENGINE_CPP_STREAM_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("TRADING_DSL_ENGINE_CPP_PCH", "0")
    monkeypatch.setenv("TRADING_DSL_ENGINE_CPP_LTO", "0")
    monkeypatch.setattr(search, "LAG", 1)

    rows, instruments = 256, 9
    rng = np.random.default_rng(42)
    data = {
        "alpha": rng.normal(size=(rows, instruments)),
        "roll_rets": rng.normal(0.0, 4.0e-4, size=(rows, instruments)),
        "is_tradable_out0": np.broadcast_to(
            ((np.arange(rows) % 17) != 0)[:, None],
            (rows, instruments),
        ).astype(np.float64),
    }
    data["roll_rets"][11, 3] = 0.0
    data["roll_rets"][37, 5] = 0.08

    clean = search.clean_returns_expr()
    volatility = ewm_std(clean, span=21)
    position = var("alpha") / volatility
    position = shift(position, search.LAG)
    original = (
        shift(
            ffill(
                where(
                    var("is_tradable_out0"),
                    position,
                    float("nan"),
                )
            )
        )
        * clean
    )
    original_result = compile_formula(
        original,
        data,
        n_instruments=instruments,
    ).run(out_path=tmp_path / "original.npy")

    derived_result = compile_formula(
        [clean, volatility],
        data,
        n_instruments=instruments,
    ).run(out_path=tmp_path / "derived.npy")
    clean_values, volatility_values = derived_result.load(mmap_mode=None)
    materialized = data | {
        "clean_rets": np.ascontiguousarray(clean_values),
        "volatility": np.ascontiguousarray(volatility_values),
    }
    optimized_result = compile_formula(
        search.precomputed_alpha_pnl(var("alpha")),
        materialized,
        n_instruments=instruments,
    ).run(out_path=tmp_path / "optimized.npy")

    np.testing.assert_allclose(
        optimized_result.load(mmap_mode=None),
        original_result.load(mmap_mode=None),
        rtol=1e-12,
        atol=1e-14,
        equal_nan=True,
    )


def test_numpy_downsampling_matches_block_sum_and_cumsum():
    search = _search_module()
    values = np.arange(22, dtype=np.float64).reshape(11, 2)
    values[3, 1] = np.nan
    actual = search._portfolio_cumulative(values, 4)
    expected = np.array(
        [
            np.nansum(values[0:4], axis=0),
            np.nansum(values[0:8], axis=0),
            np.nansum(values[0:11], axis=0),
        ]
    )
    np.testing.assert_array_equal(actual, expected)


def test_search_hot_loop_has_no_python_worker_pool_or_duplicate_source_load():
    source = _SCRIPT_PATH.read_text()
    assert "ThreadPoolExecutor" not in source
    assert "sources_all" not in source
    assert "run_many(" in source
    assert "compile_formula(\n        formulas," in source
    assert "train_end=(folds[-1].train_end if folds else None)" in source


def test_inputdata_import_does_not_eagerly_import_jax_backend():
    code = (
        "import sys; import flows.load; "
        "assert 'trading_dsl_engine.jax_flat.engine' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        env=os.environ.copy(),
        check=True,
    )
