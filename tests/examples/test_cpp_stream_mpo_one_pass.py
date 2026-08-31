from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

from examples import cpp_stream_mpo_one_pass as example


def _fake_data(rows: int = 256, n_assets: int = 3) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(7)
    returns = rng.normal(scale=2e-4, size=(rows, n_assets))
    tradable = np.ones((rows, n_assets), dtype=float)
    tradable[80:90] = 0.0
    tradable[180:190] = 0.0
    returns[tradable == 0.0] = 0.0
    # First bar after each gap carries the gap's accumulated diffusion scale.
    returns[90] *= np.sqrt(11.0)
    returns[190] *= np.sqrt(11.0)
    hs = rng.uniform(3e-5, 8e-5, size=(rows, n_assets))
    ts = np.broadcast_to(
        (1_800_000_000_000_000 + np.arange(rows) * 60_000_000)[:, None],
        (rows, n_assets),
    ).copy()
    return {
        "returns": returns,
        "is_tradable_out0": tradable,
        "vw_halfspread_out0": hs,
        "_ev_ts": ts,
    }


def test_realized_portfolio_pnl_uses_carried_weights_and_spread_cost() -> None:
    returns = np.array([[0.1, -0.2], [0.3, 0.4], [-0.1, 0.2]])
    weights = np.array([[1.0, 0.0], [2.0, -1.0], [1.5, -0.5]])
    hs = np.full_like(returns, 0.01)

    gross, net = example._portfolio_pnl(returns, weights, hs)

    np.testing.assert_allclose(gross, [0.0, 0.3, -0.4])
    np.testing.assert_allclose(net, [-0.01, 0.27, -0.415])


def test_fake_data_runs_full_mpo_example(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(example, "FEATURE_HLS", (2, 4, 8, 16))
    monkeypatch.setattr(example, "RIDGE_HL", 64)
    monkeypatch.setattr(example, "RISK_SPAN", 32)
    monkeypatch.setattr(example, "RISK_MIN_PERIODS", 8)
    monkeypatch.setattr(example, "YHAT_VOL_SPAN", 32)
    monkeypatch.setattr(example, "YHAT_VOL_MIN_PERIODS", 8)

    result, paths = example._run(
        _fake_data(),
        returns=example.var("returns"),
        output_dir=tmp_path,
    )

    assert result.rows == 256
    assert len(paths) == 18
    names = {path.name for path in paths}
    assert "portfolio_pnl.png" in names
    assert "risk_constraint.png" in names
    assert not any(name.startswith("mpo_horizon_") for name in names)
    assert sum(name.startswith("yhat_horizon_") for name in names) == 8
