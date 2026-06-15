"""Compare Ridge betas with a RegXS factor-portfolio reconstruction.

The script generates a few thousand rows for two predictors across nine assets,
injects NaNs into predictors and targets, and saves a plot next to this file.
RegXS estimates the cross-sectional X'X matrix with pairwise intersection counts
for predictor NaNs, exposes factor-portfolio weights through get_fp(...), and the
example reconstructs betas from the requested expression:

    ewm(get_fp(RegXS(...)), lam_) * y

summed across assets for each predictor.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from trading_dsl_engine.jax_flat import compile_formula


ROWS = 4_000
ASSETS = 9
LAM_ = 24.0
RIDGE_LAMBDA = 0.05
OUT = Path(__file__).with_suffix(".svg")


def run_formula(formula: str, data: dict[str, np.ndarray]) -> np.ndarray:
    runtime = compile_formula(formula, cpp=False)
    _, out = runtime.run_batch(data)
    return np.asarray(out)


def main() -> None:
    rng = np.random.default_rng(20260615)
    t = np.arange(ROWS, dtype=np.float64)[:, None]
    asset_loading = np.linspace(-0.6, 0.7, ASSETS, dtype=np.float64)[None, :]

    x1 = rng.normal(size=(ROWS, ASSETS)) + 0.25 * np.sin(t / 37.0) + asset_loading
    x2 = rng.normal(size=(ROWS, ASSETS)) + 0.20 * np.cos(t / 53.0) - 0.5 * asset_loading
    beta_true_1 = 0.65 + 0.20 * np.sin(np.arange(ROWS, dtype=np.float64) / 450.0)
    beta_true_2 = -0.25 + 0.15 * np.cos(np.arange(ROWS, dtype=np.float64) / 600.0)
    y = beta_true_1[:, None] * x1 + beta_true_2[:, None] * x2 + rng.normal(scale=0.15, size=(ROWS, ASSETS))

    x1[rng.random(x1.shape) < 0.05] = np.nan
    x2[rng.random(x2.shape) < 0.07] = np.nan
    y[rng.random(y.shape) < 0.04] = np.nan
    data = {"x1": x1, "x2": x2, "y": y}

    ridge_beta = run_formula(f"get_beta(Ridge(cat(x1, x2), y, 0, {RIDGE_LAMBDA}))", data)
    fp_ewm = run_formula(f"ewm(get_fp(RegXS(cat(x1, x2), {RIDGE_LAMBDA})), {LAM_})", data)
    regxs_beta = np.einsum("tnk,tn->tk", np.nan_to_num(fp_ewm), np.nan_to_num(y))

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(ridge_beta[:, idx], label=f"get_beta(Ridge(...)) beta[{idx}]", linewidth=1.2)
        ax.plot(regxs_beta[:, idx], label=f"sum_assets(ewm(get_fp(RegXS(...)), {LAM_}) * y) beta[{idx}]", linewidth=1.0, alpha=0.85)
        ax.set_ylabel(f"beta[{idx}]")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("row")
    fig.suptitle("RegXS factor portfolios vs stateless Ridge betas (2 predictors, 9 assets, NaNs)")
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
