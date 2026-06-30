import re
from dataclasses import dataclass
from types import SimpleNamespace

import jax.numpy as jnp

from flows.utils import mask, pct_change, cumprod
from trading_dsl_engine.base.dsl import *
from trading_dsl_engine.jax_flat import stateless


def _in_current_session(ts, session_start, session_end):
    return (
            jnp.isfinite(ts)
            & jnp.isfinite(session_start)
            & jnp.isfinite(session_end)
            & (session_end > session_start)
            & (ts >= session_start)
            & (ts < session_end)
    )


volume_for_fit = stateless(
    lambda volume, ts, session_start, session_end: jnp.where(
        _in_current_session(ts, session_start, session_end),
        jnp.maximum(jnp.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0), 0.0),
        jnp.nan,
    ),
    output_kind="vector",
    output_width=1,
    name="volume_for_fit_session",
)
volume_for_seen = stateless(
    lambda volume, ts, session_start, session_end, is_tradable: jnp.where(
        _in_current_session(ts, session_start, session_end) & jnp.isfinite(is_tradable) & (is_tradable == 1.0),
        jnp.maximum(jnp.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0), 0.0),
        0.0,
    ),
    output_kind="vector",
    output_width=1,
    name="volume_for_seen_session",
)
nonnegative = stateless(
    lambda x: jnp.maximum(jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 0.0),
    output_kind="vector",
    output_width=1,
    name="nonnegative",
)
pct_seen = stateless(
    lambda seen, forecast, ts, session_start: jnp.where(
        jnp.isfinite(ts) & jnp.isfinite(session_start) & (ts >= session_start) & ((seen + forecast) > 0.0),
        seen / (seen + forecast),
        jnp.nan,
    ),
    output_kind="vector",
    output_width=1,
    name="pct_seen_session_volume",
)

PovFields = SimpleNamespace(
    ts = var("_ev_ts"),
    session_start = var("session_start0"),
    session_end = var("session_end0"),
    volume = var("volume_out0"),
    is_tradable = var("is_tradable_out0"),
)


def pov(n_basis: int = 6, h: int = 1440, f: SimpleNamespace = PovFields):
    ts, session_start, session_end, volume, is_tradable = f.ts, f.session_start, f.session_end, f.volume, f.is_tradable
    features = rbf_basis(ts, session_start, session_end, n_basis)
    fit_y = volume_for_fit(volume, ts, session_start, session_end)
    beta = get_beta(InstrumentBasisMean(features, fit_y, 1.0, 21 * h))
    forecast = nonnegative(einsum(beta, future_rbf_basis_sum(ts, session_start, session_end, n_basis, h), "nf,nf->n"))
    seen = groupby(
        (session_start,),
        volume_for_seen(volume, ts, session_start, session_end, is_tradable),
        cumsum(self_),
    )
    pov = pct_seen(seen, forecast, ts, session_start)
    return pov


RollRetsFields = SimpleNamespace(
    wdte=var("wdte_out0"),
    px0=var("mp_out0.close"),
    px1 = var("mp_out1.close"),
    is_tradable_out0 = var("is_tradable_out0"),
    is_tradable_out1 = var("is_tradable_out1"),
)



@dataclass
class RollRets:
    fields = RollRetsFields

    def roll_rets(self, days_roll: int = 2,  **kwargs):
        f = self.fields
        wdte, px0, px1, is_tradable_out0, is_tradable_out1 = f.wdte, f.px0, f.px1, f.is_tradable_out0, f.is_tradable_out1

        # relies on before open (since w0 is shifted)
        w0 = ffill(where(ffill(wdte) == days_roll, 1 - pov(**kwargs), where(ffill(wdte) < days_roll, 0, 1)))
        w1 = 1 - w0
        self.w0, self.w1 = w0, w1

        roll_flag = diff(mask(wdte)) > 0
        ret = pct_change(mask(px0))
        self.ret = ret
        r0 = where(
            roll_flag,
            px0 / mask(px1, is_tradable_out0 & shift(is_tradable_out0)) - 1,
            ret
        )
        r1 = pct_change(mask(px1, is_tradable_out1))
        self.raw_rets = cat(r0, r1)
        # roll_rets = einsum(
        #     cat(mask(shift(w0)), mask(shift(w1))), # TODO: shift makes it so wont reach eod flat -> its actually 0 at session_end (eg 2059)
        #     cat(r0, pct_change(mask(px1, is_tradable_out1))),
        #     "nf,nf->n",
        # )

        roll_rets = mask(shift(w0)) * r0 + mask(shift(w1)) * r1

        return roll_rets

    def adj_factor(self, **kwargs):
        adj_factor = cumprod(1 + self.roll_rets(**kwargs)) / cumprod(1 + self.ret)
        return adj_factor


    def map(self, ws):
        return einsum(
            cat(ws),
            cat(self.w0, self.w1),
            "n,nf->nf"
        )

    def raw_rets(self):
        if not hasattr(self, "raw_rets"):
            self.roll_rets()
        return self.raw_rets



# Price-like fields: multiply by roll ratio.
_MUL_PATTERNS: tuple[re.Pattern[str], ...] = (
    # ap0_out0, ap9_out1, bp3_out0, etc.
    re.compile(r"^[ab]p\d+_out\d+$"),
    # ap_out0.close, bp_out1.high, mp_out0.open, etc.
    re.compile(r"^[abm]p_out\d+\.(?:open|high|low|close)$"),
    # vwap_out0, vwap_mp_out1
    re.compile(r"^vwap(?:_mp)?_out\d+$"),
)
# Quantity / volume-like fields: divide by roll ratio.
_DIV_PATTERNS: tuple[re.Pattern[str], ...] = (
    # volume_a0_out0, volume_b9_out1
    re.compile(r"^volume_[ab]\d+_out\d+$"),
    # volume_out0, volume_out1
    re.compile(r"^volume_out\d+$"),
)

def _mapper(key: str):
    if any(p.match(key) for p in _MUL_PATTERNS):
        return mul
    elif any(p.match(key) for p in _DIV_PATTERNS):
        return div
    else:
        return None

def adj(x: Expr, adj_factor: Expr = None, **kwargs):
    if adj_factor is None:
        adj_factor = RollRets().adj_factor(**kwargs)
    if isinstance(x, Identifier):
        k = x.name
    else:
        raise NotImplementedError
    op = _mapper(k)
    return op(var(k), adj_factor)

if __name__ == "__main__":
    from flows.load import InputData

    in_data = InputData(nrows=1E6)
    import time
    start = time.perf_counter()
    in_data.run(pov())
    end = time.perf_counter()
    print(end-start)
