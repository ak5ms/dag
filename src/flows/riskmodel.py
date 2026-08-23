import jax.numpy as jnp

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


pinv = stateless(lambda x: jnp.linalg.pinv(x), output_kind="matrix", name='pinv')
evals = stateless(lambda x: jnp.linalg.eigvalsh(x), output_kind="vector", name='evals')
inv = stateless(lambda x: _inv(x), output_kind="matrix", name='inv')
cov2corr = stateless(lambda x: _cov2corr(x), output_kind="matrix", name='cov2corr')
corr2cov = stateless(lambda corr, std: _corr2cov(corr, std), output_kind="matrix", name='corr2cov')
near_psd = stateless(lambda x: _near_psd(x), output_kind="matrix", name='near_psd')

def risk_covariance(
    returns,
    *,
    span: float = 1440 * 21,
    min_periods: int = 0,
    ignore_na: bool = True,
    adjust: bool = False,
):
    """Pairwise-missing exponentially weighted covariance-like second moment.

    Exact-zero returns follow the production convention and are treated as
    missing observations. The returned ``(N, N)`` matrix remains a normal DSL
    expression, so cpp_stream updates it inside the same row transition as any
    downstream optimizer.
    """

    cleaned = replace(returns, 0, float("nan"))
    outer = einsum(fillna(cleaned, 0), fillna(cleaned, 0), "i,j->ij")
    observed_outer = replace(outer, 0, float("nan"))
    return ewm(
        observed_outer,
        span,
        min_periods=min_periods,
        ignore_na=ignore_na,
        adjust=adjust,
    )


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


