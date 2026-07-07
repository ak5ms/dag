from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import norm, poisson


def nearest_psd(A: Any, eps: float = 1e-12) -> np.ndarray:
    A = np.asarray(A, dtype=float)
    A = 0.5 * (A + A.T)
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, eps)
    return vecs @ np.diag(vals) @ vecs.T


def nearest_corr(C: Any, eps: float = 1e-12) -> np.ndarray:
    C = nearest_psd(C, eps)
    d = np.sqrt(np.maximum(np.diag(C), eps))
    C = C / np.outer(d, d)
    C = np.clip(C, -0.999, 0.999)
    np.fill_diagonal(C, 1.0)
    return C


def cov_to_corr(S: Any, eps: float = 1e-12) -> np.ndarray:
    S = nearest_psd(S, eps)
    d = np.sqrt(np.maximum(np.diag(S), eps))
    return nearest_corr(S / np.outer(d, d), eps)


def is_tradable_eastern(local_ts: pd.DatetimeIndex) -> np.ndarray:
    dow = local_ts.dayofweek
    mins = local_ts.hour * 60 + local_ts.minute
    daily_break = (mins >= 17 * 60) & (mins < 17 * 60 + 40)
    weekend = ((dow == 4) & (mins >= 17 * 60)) | (dow == 5) | ((dow == 6) & (mins < 17 * 60))
    return np.asarray((~(daily_break | weekend)).astype(np.int8))


def simulate_ar1_gaussian_returns(n_bars: int, mu: Any, S: Any, phi: float = 0.15, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, dtype=float)
    S = nearest_psd(S)
    n_assets = len(mu)
    Q = nearest_psd((1.0 - phi**2) * S)
    chol_Q = np.linalg.cholesky(Q)
    chol_S = np.linalg.cholesky(S)
    x = np.empty((n_bars, n_assets))
    x[0] = rng.standard_normal(n_assets) @ chol_S.T
    eps = rng.standard_normal((n_bars, n_assets)) @ chol_Q.T
    for t in range(1, n_bars):
        x[t] = phi * x[t - 1] + eps[t]
    return mu + x


def ewm_cov_tensor(x: Any, span: float = 63) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    T, N = x.shape
    alpha = 2.0 / (span + 1.0)
    mean = np.zeros(N)
    cov = np.zeros((N, N))
    out = np.empty((T, N, N))
    for t in range(T):
        xt = x[t]
        if t == 0:
            mean = xt.copy()
            cov[:] = 0.0
        else:
            old_mean = mean.copy()
            mean = (1.0 - alpha) * mean + alpha * xt
            cov = (1.0 - alpha) * cov + alpha * np.outer(xt - old_mean, xt - mean)
        out[t] = nearest_psd(cov + 1e-12 * np.eye(N))
    return out


def gaussian_copula_poisson(n_obs: int, lam: Any, corr: Any, rng: np.random.Generator) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)
    corr = nearest_corr(corr)
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_obs, len(lam))) @ L.T
    u = np.clip(norm.cdf(z), 1e-12, 1.0 - 1e-12)
    return np.column_stack([poisson.ppf(u[:, i], mu=lam[i]).astype(int) for i in range(len(lam))])


def simulate_requested_fields(
    n_minutes: int = 14 * 24 * 60,
    start: str = "2026-01-05 00:00:00",
    tz: str = "America/New_York",
    mu: Any = None,
    S: Any = None,
    phi: float = 0.15,
    price0: Any = None,
    volume_lambda: Any = None,
    ewm_span: float = 63,
    spread_k: float = 0.75,
    spread_floor_bps: float = 0.5,
    range_k: float = 1.75,
    zero_volume_when_not_tradable: bool = False,
    seed: int = 42,
) -> dict[str, np.ndarray]:
    mu = np.array([0.000015, 0.000008, -0.000003, 0.000005]) if mu is None else np.asarray(mu, dtype=float)
    if S is None:
        vol = np.array([0.0010, 0.0013, 0.0008, 0.0011])
        corr = np.array(
            [[1.00, 0.55, -0.25, 0.20], [0.55, 1.00, -0.10, 0.35], [-0.25, -0.10, 1.00, -0.30], [0.20, 0.35, -0.30, 1.00]]
        )
        S = np.outer(vol, vol) * corr
    S = nearest_psd(S)
    n_assets = len(mu)
    price0 = np.array([100.0, 75.0, 120.0, 50.0]) if price0 is None else np.asarray(price0, dtype=float)
    volume_lambda = np.array([1200, 900, 650, 1000]) if volume_lambda is None else np.asarray(volume_lambda, dtype=float)
    if len(price0) != n_assets or len(volume_lambda) != n_assets or S.shape != (n_assets, n_assets):
        raise ValueError("mu, S, price0, and volume_lambda must agree on asset count")

    rng = np.random.default_rng(seed)
    rets = simulate_ar1_gaussian_returns(n_minutes, mu, S, phi=phi, seed=seed)
    close = np.exp(np.log(price0) + np.cumsum(rets, axis=0))
    open_ = np.vstack([price0, close[:-1]])
    S_hat = ewm_cov_tensor(rets, ewm_span)
    sigma_hat = np.sqrt(np.maximum(np.diagonal(S_hat, axis1=1, axis2=2), 0.0))
    spread_bps = spread_floor_bps + 1e4 * spread_k * sigma_hat
    spread = close * spread_bps / 1e4
    base_sigma = np.sqrt(np.diag(S))
    sigma_for_range = 0.5 * base_sigma + 0.5 * sigma_hat
    up_excursion = np.abs(rng.standard_normal((n_minutes, n_assets))) * range_k * sigma_for_range
    dn_excursion = np.abs(rng.standard_normal((n_minutes, n_assets))) * range_k * sigma_for_range
    high = np.exp(np.maximum(np.log(open_), np.log(close)) + up_excursion)
    low = np.exp(np.minimum(np.log(open_), np.log(close)) - dn_excursion)
    volume = gaussian_copula_poisson(n_minutes, volume_lambda, cov_to_corr(S), rng)
    local_ts = pd.date_range(start=start, periods=n_minutes, freq="min", tz=tz)
    ev_ts = (local_ts.tz_convert("UTC").view("int64") // 1_000).astype(np.int64)
    is_tradable = is_tradable_eastern(local_ts)
    if zero_volume_when_not_tradable:
        volume = volume * is_tradable[:, None]

    return {
        "_ev_ts": np.repeat(ev_ts, n_assets),
        "asset_id": np.tile(np.arange(n_assets, dtype=np.int64), n_minutes),
        "local_ts": np.repeat(local_ts.astype(str), n_assets),
        "mp_out0.close": close.reshape(-1),
        "mp_out0.high": high.reshape(-1),
        "mp_out0.low": low.reshape(-1),
        "mp_out0.open": open_.reshape(-1),
        "bp_out0.close": (close - spread / 2.0).reshape(-1),
        "ap_out0.close": (close + spread / 2.0).reshape(-1),
        "volume_out0": volume.reshape(-1),
        "is_tradable_out0": np.repeat(is_tradable, n_assets),
    }


def matrix_inputs_from_long_fields(fields: dict[str, Any], columns: Sequence[str]) -> dict[str, np.ndarray]:
    index = pd.MultiIndex.from_arrays([fields["_ev_ts"], fields["asset_id"]], names=["_ev_ts", "asset_id"])
    return {
        column: pd.Series(fields[column], index=index).unstack("asset_id").sort_index(axis=1).to_numpy(dtype=float)
        for column in columns
    }


__all__ = [
    "cov_to_corr",
    "ewm_cov_tensor",
    "gaussian_copula_poisson",
    "is_tradable_eastern",
    "matrix_inputs_from_long_fields",
    "nearest_corr",
    "nearest_psd",
    "simulate_ar1_gaussian_returns",
    "simulate_requested_fields",
]
