from __future__ import annotations

from pathlib import Path

import numpy as np

from trading_dsl_engine.base.dsl import cumsum, groupby, self_, var
from trading_dsl_engine.base.keys import Key
from trading_dsl_engine.cpp_stream import compile_formula


def test_monotonic_group_key_recycles_state_and_matches_hash_grouping(
    tmp_path: Path,
) -> None:
    rows, n, session_rows = 97, 5, 11
    rng = np.random.default_rng(42)
    x = rng.normal(size=(rows, n)).astype(np.float64)
    session_scalar = np.floor_divide(np.arange(rows), session_rows).astype(np.float64)
    session = np.repeat(session_scalar[:, None], n, axis=1)

    baseline = groupby(var("session"), var("x"), cumsum(self_))
    hinted = groupby(
        Key(
            var("session"),
            row_scalar=True,
            dtype="float64",
            monotonic=True,
        ),
        var("x"),
        cumsum(self_),
    )
    data: dict[str, object] = {
        "session": session,
        "x": x,
        "unused_invalid": object(),
        "unused_short": np.empty((2, n), dtype=np.float64),
    }
    baseline_runtime = compile_formula(
        baseline,
        data,
        default_group_capacity=64,
    )
    hinted_runtime = compile_formula(
        hinted,
        data,
        default_group_capacity=64,
    )
    capacities = tuple(
        stage.group.capacity
        for stage in hinted_runtime.plan.stages
        if stage.group is not None
    )
    assert capacities == (1,)

    baseline_path = tmp_path / "baseline.bin"
    hinted_path = tmp_path / "hinted.bin"
    baseline_runtime.run(data, out_path=baseline_path)
    hinted_runtime.run(data, out_path=hinted_path)
    baseline_out = np.memmap(
        baseline_path, mode="r", dtype=np.float64, shape=(rows, n)
    )
    hinted_out = np.memmap(
        hinted_path, mode="r", dtype=np.float64, shape=(rows, n)
    )
    np.testing.assert_array_equal(hinted_out, baseline_out)
