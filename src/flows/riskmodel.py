import jax.numpy as jnp
import numpy as np

from trading_dsl_engine.base.dsl import *
from trading_dsl_engine.jax_flat import stateless
from flows.pov import RollRets
from flows.utils import replace, streak

# TODO: build separate overnight risk model (time asynchrony? -> j do vol)
def _cov2corr(cov):
    std_devs = jnp.sqrt(jnp.diag(cov))
    corr = cov / (std_devs[:, None] * std_devs[None, :])
    return corr


def _corr2cov(corr, std):
    return corr * (std[:, None] * std[None, :])


def _inv(matrix):
    sym_mat = 0.5 * (matrix + matrix.T)
    evals, evecs = jnp.linalg.eigh(sym_mat)
    evals_clean = jnp.maximum(evals, 1e-12)
    evals_clean *= jnp.sqrt(sum(evals ** 2) / sum(evals_clean ** 2))  # variance matching
    return evecs @ jnp.diag(1 / evals_clean) @ evecs.T


def _near_psd(matrix):
    sym_mat = 0.5 * (matrix + matrix.T)
    evals, evecs = jnp.linalg.eigh(sym_mat)
    evals_clean = jnp.maximum(evals, 1e-12)
    evals_clean *= jnp.sqrt(sum(evals ** 2) / sum(evals_clean ** 2))  # variance matching
    return evecs @ jnp.diag(evals_clean) @ evecs.T


def covariance_diagnostics(covariance):
    """Numerical/coverage summary for a raw or cleaned covariance snapshot."""
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")

    finite = np.isfinite(cov)
    diag = np.diag(cov)
    valid_diag = np.isfinite(diag) & (diag > 0.0)
    finite_fraction = float(np.mean(finite))
    diagonal_coverage = float(np.mean(valid_diag))

    # Symmetry error only compares entries observed in both directions.
    paired = finite & finite.T
    if np.any(paired):
        symmetry_error = float(np.max(np.abs(cov[paired] - cov.T[paired])))
    else:
        symmetry_error = float("nan")

    if not np.all(finite):
        return {
            "finite": False,
            "finite_fraction": finite_fraction,
            "diagonal_coverage": diagonal_coverage,
            "symmetry_error": symmetry_error,
            "min_eigenvalue": float("nan"),
            "max_eigenvalue": float("nan"),
            "condition_number": float("inf"),
        }

    sym = 0.5 * (cov + cov.T)
    evals = np.linalg.eigvalsh(sym)
    min_eval = float(np.min(evals))
    max_eval = float(np.max(evals))
    condition = float("inf") if min_eval <= 0.0 else max_eval / min_eval
    return {
        "finite": True,
        "finite_fraction": finite_fraction,
        "diagonal_coverage": diagonal_coverage,
        "symmetry_error": symmetry_error,
        "min_eigenvalue": min_eval,
        "max_eigenvalue": max_eval,
        "condition_number": condition,
    }


def sanitize_covariance(
    covariance,
    *,
    min_variance: float = 1e-12,
    max_abs_correlation: float = 0.999,
    max_condition_number: float = 1e8,
    min_diagonal_coverage: float = 1.0,
    min_finite_fraction: float = 0.8,
):
    """Return a finite, symmetric, well-conditioned positive-definite covariance.

    ``flows.riskmodel.cov`` is estimated entry-by-entry because missing/zero return
    observations are skipped. A snapshot can consequently contain NaNs, asymmetric
    availability, or a small PSD violation. Missing off-diagonals are conservatively
    mapped to zero correlation. Missing/non-positive variances can be repaired with
    the median observed variance only when the caller explicitly relaxes the default
    full diagonal-coverage requirement. We reject low-coverage snapshots instead of
    manufacturing a mostly synthetic risk model.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1] or cov.shape[0] == 0:
        raise ValueError("covariance must be a non-empty square matrix")
    if not (min_variance > 0.0):
        raise ValueError("min_variance must be positive")
    if not (0.0 < max_abs_correlation < 1.0):
        raise ValueError("max_abs_correlation must be in (0, 1)")
    if not (max_condition_number > 1.0):
        raise ValueError("max_condition_number must be > 1")
    if not (0.0 < min_diagonal_coverage <= 1.0):
        raise ValueError("min_diagonal_coverage must be in (0, 1]")
    if not (0.0 < min_finite_fraction <= 1.0):
        raise ValueError("min_finite_fraction must be in (0, 1]")

    finite_fraction = float(np.mean(np.isfinite(cov)))
    if finite_fraction < min_finite_fraction:
        raise ValueError(
            "covariance is degenerate: finite pair coverage "
            f"{finite_fraction:.3f} < {min_finite_fraction:.3f}"
        )

    raw_diag = np.diag(cov)
    valid_diag = np.isfinite(raw_diag) & (raw_diag > min_variance)
    coverage = float(np.mean(valid_diag))
    if coverage < min_diagonal_coverage:
        raise ValueError(
            "covariance is degenerate: finite positive diagonal coverage "
            f"{coverage:.3f} < {min_diagonal_coverage:.3f}"
        )

    finite = np.isfinite(cov)
    finite_t = finite.T
    both = finite & finite_t
    only_left = finite & ~finite_t
    only_right = ~finite & finite_t

    sym = np.zeros_like(cov)
    sym[both] = 0.5 * (cov[both] + cov.T[both])
    sym[only_left] = cov[only_left]
    sym[only_right] = cov.T[only_right]

    diag = np.diag(sym).copy()
    valid_diag = np.isfinite(diag) & (diag > min_variance)
    fallback_variance = max(float(np.median(diag[valid_diag])), min_variance)
    diag = np.where(valid_diag, diag, fallback_variance)
    np.fill_diagonal(sym, diag)

    std = np.sqrt(diag)
    corr = sym / np.outer(std, std)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = 0.5 * (corr + corr.T)
    corr = np.clip(corr, -max_abs_correlation, max_abs_correlation)
    np.fill_diagonal(corr, 1.0)

    evals, evecs = np.linalg.eigh(corr)
    corr_floor = max(float(np.max(evals)) / max_condition_number, 1e-12)
    evals = np.maximum(evals, corr_floor)
    corr = (evecs * evals) @ evecs.T

    # Re-normalize after the eigenvalue floor. Positive diagonal congruence preserves PSD.
    corr_scale = np.sqrt(np.maximum(np.diag(corr), 1e-15))
    corr = corr / np.outer(corr_scale, corr_scale)
    clean = corr * np.outer(std, std)
    clean = 0.5 * (clean + clean.T)

    # Volatility dispersion can make covariance ill-conditioned even when correlation is not.
    evals, evecs = np.linalg.eigh(clean)
    largest = float(np.max(evals))
    if not np.isfinite(largest) or largest <= 0.0:
        raise ValueError("covariance is degenerate after cleaning")
    floor = max(largest / max_condition_number, min_variance)
    evals = np.maximum(evals, floor)
    clean = (evecs * evals) @ evecs.T
    clean = 0.5 * (clean + clean.T)
    if not np.all(np.isfinite(clean)):
        raise ValueError("covariance cleaning produced non-finite values")
    return clean


pinv = stateless(lambda x: jnp.linalg.pinv(x), output_kind="matrix", name='pinv')
evals = stateless(lambda x: jnp.linalg.eigvalsh(x), output_kind="vector", name='evals')
inv = stateless(lambda x: _inv(x), output_kind="matrix", name='inv')
cov2corr = stateless(lambda x: _cov2corr(x), output_kind="matrix", name='cov2corr')
corr2cov = stateless(lambda corr, std: _corr2cov(corr, std), output_kind="matrix", name='corr2cov')
near_psd = stateless(lambda x: _near_psd(x), output_kind="matrix", name='near_psd')


def risk_covariance(returns, span: int = 1440 * 21):
    """The production pairwise-missing EWM covariance construction.

    Preserve the existing risk-model convention that both NaN and exact-zero returns
    are missing observations: fill NaN before the outer product, then turn any zero
    product back into NaN so EWM (``ignore_na=True`` by default) skips that pair.
    """
    observed = fillna(returns, 0)
    pair_products = replace(einsum(observed, observed, "i,j->ij"), 0, float("nan"))
    return ewm(pair_products, span)


roll_rets = RollRets().roll_rets()
cov = risk_covariance(roll_rets)
# vol = ffill(ewm_std(roll_rets, 1440*10))

# cov_clean = ffill(corr2cov(near_psd(cov2corr(cov)), vol))
#
# cov_alpha = einsum(
#     fillna(cat(*[(x * vol) for x in alphas]), 0),
#     cov_clean,
#     fillna(cat(*[(x * vol) for x in alphas]), 0),
#     "ki,kl,lj->ij",
# )
ev_ts_ffill = ffill(var("_ev_ts")) + streak(isnan(var("_ev_ts"))) * 60E6
in_session = (ffill(var("session_start0")) < ev_ts_ffill) & (ev_ts_ffill <= ffill(var("session_end0")))

roll_rets_gap = where(~in_session, roll_rets, float("nan"))
roll_rets_gap = replace(roll_rets_gap, 0, float("nan"))
gap_time = shift(streak(isnan(roll_rets_gap)))
roll_rets_gap_scaled = roll_rets_gap / (gap_time / 1440) ** 0.5
roll_rets_session = replace(where(in_session, roll_rets, float("nan")), 0, float("nan"))

vol_session = ewm(roll_rets_session ** 2, 1440, ignore_na=True, adjust=True) ** 0.5
vol_gap = ewm(roll_rets_gap ** 2, 5, ignore_na=True, adjust=True) ** 0.5
vol_comb = where(
    (ffill(var("session_start0")) <= ev_ts_ffill) & (ev_ts_ffill < ffill(var("session_end0")) - 60E6 * 10),
    vol_session,
    vol_gap
)
vol = vol_comb
