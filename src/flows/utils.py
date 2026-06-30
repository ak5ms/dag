from trading_dsl_engine.base.dsl import *


def pct_change(x):
    return x / shift(x, 1, 1) - 1.0


def mask(w, tradable_mask=var("is_tradable_out0"), fill=ffill):
    out = where(fillna(tradable_mask,0) != 1, float("nan"), w)
    if fill:
        out = fill(out)
    return out

def cumprod(x: Expr) -> Expr:
    return exp(cumsum(ln(x)))


def replace(x: Expr, y: Expr, z: Expr) -> Expr:
    return where(x==y, z, x)

def ewm_var(x: Expr, span: Expr, min_periods: Expr = None, replace_0: bool = True) -> Expr:
    if not min_periods:
        min_periods = span
    if replace_0:
        x = replace(x, 0, float("nan"))
    out = (ewm(x**2, span, min_periods) - (ewm(x, span, min_periods) ** 2))
    return out

def ewm_std(x: Expr, span: Expr, min_periods: Expr = None, replace_0: bool = True) -> Expr:
    return ewm_var(x, span, min_periods, replace_0) ** 0.5


def ewm_mean(x: Expr, span: Expr, min_periods: Expr = None, replace_0: bool = True) -> Expr:
    if not min_periods:
        min_periods = span
    if replace_0:
        x = replace(x, 0, float("nan"))
    out = ewm(x, span, min_periods)
    return out

def dszl(x: Expr, span: Expr, grouping: Expr = timeofday, time: str = var("_ev_ts"), **kwargs) -> Expr:
    out = groupby(
        (grouping(to_dt(time)),),
        x,
        (self_ - ewm_mean(self_, span=span))/ewm_std(self_, span=span),
        **kwargs
    )
    return out

def streak(cond):
    "# rows since last true"
    z = cond
    cs = cumsum(z)
    baseline = fillna(ffill(where(z == 0, cs, float("nan"))), 0)
    out = where(z == 1, cs - baseline, 0)
    return out

