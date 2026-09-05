from __future__ import annotations

import inspect

import cvxpy as cp
import numpy as np

import examples.cpp_stream_mpo_one_pass as example
from trading_dsl_engine.base.dsl import var


def test_diagnostic_pnls_delegate_to_alpha_search_ic_and_ic1(monkeypatch):
    calls = []

    def fake_ic(signal, **kwargs):
        calls.append(("ic", signal, kwargs))
        return var("ic_out")

    def fake_ic1(signal, **kwargs):
        calls.append(("ic1", signal, kwargs))
        return var("ic1_out")

    monkeypatch.setattr(example, "ic", fake_ic, raising=False)
    monkeypatch.setattr(example, "ic1", fake_ic1, raising=False)
    signal = var("signal")
    rets = var("returns")
    tradable = var("tradable")
    weights = var("weights")

    actual = example._diagnostic_pnls(
        signal,
        roll_rets=rets,
        is_tradable=tradable,
        w=weights,
        lag=2,
        hz=4,
    )

    assert set(actual) == {"ic", "ic1"}
    assert [name for name, _, _ in calls] == ["ic", "ic1"]
    for _, called_signal, kwargs in calls:
        assert called_signal is signal
        assert kwargs["roll_rets"] is rets
        assert kwargs["is_tradable"] is tradable
        assert kwargs["w"] is weights
        assert kwargs["lag"] == 2
        assert kwargs["hz"] == 4
        assert kwargs["hl"] == example.IC_VOL_SPAN


def test_diagnostic_pnls_scalarize_xs_broadcast(monkeypatch):
    monkeypatch.setattr(example, "ic", lambda *args, **kwargs: var("ic_out"))
    monkeypatch.setattr(example, "ic1", lambda *args, **kwargs: var("ic1_out"))

    actual = example._diagnostic_pnls(
        var("signal"),
        roll_rets=var("returns"),
        is_tradable=var("tradable"),
        w=var("weights"),
        lag=1,
        hz=2,
    )

    assert actual["ic"].fn == "sum"
    assert actual["ic1"].fn == "sum"
    assert "axis" in repr(actual["ic"])
    assert "axis" in repr(actual["ic1"])


def test_formula_uses_only_future_tradeable_horizons_and_negative_zscores():
    formula = example._formula(var("returns"))

    assert example.HORIZONS == (2, 4, 8, 16, 32, 64, 128)
    assert example.TRADE_STARTS == (1, 2, 4, 8, 16, 32, 64)
    assert "0_1" not in formula["alpha_pnl"]
    assert set(formula) >= {
        "returns",
        "features",
        "weights",
        "status",
        "mpo_objective",
        "mpo_gross_pnl",
        "risk",
        "alpha_pnl",
        "yhat_pnl",
    }
    assert "mpo_spread_cost" not in formula
    assert "objective" in repr(formula["mpo_objective"])
    assert len(formula["alpha_pnl"]) == len(example.HORIZONS)
    assert len(formula["yhat_pnl"]) == len(example.HORIZONS)

    features = formula["features"]
    assert features.fn == "cat"
    assert len(features.args) == len(example.FEATURE_SPANS)
    for feature in features.args:
        assert feature.fn == "mul"
        assert "xs_gauss" in repr(feature) or "xs_generalized_rank" in repr(feature)

    for horizon in formula["alpha_pnl"].values():
        assert set(horizon) == {"ic", "ic1"}
        assert len(horizon["ic"]) == len(example.FEATURE_SPANS)
        assert len(horizon["ic1"]) == len(example.FEATURE_SPANS)
    for horizon in formula["yhat_pnl"].values():
        assert set(horizon) == {"ic", "ic1"}


def test_mpo_prices_spread_directly_in_objective_without_spread_constraint():
    n_horizons = len(example.HORIZONS)
    n_assets = 3
    zeros = np.zeros((n_horizons, n_assets))
    factors = [np.eye(n_assets) for _ in range(n_horizons)]
    problem = example.MPO.factory(
        zeros,
        np.full(n_assets, 1e-4),
        np.zeros(n_assets),
        np.zeros(n_assets),
        np.ones(n_assets),
        *factors,
        np.ones((n_horizons, n_assets)),
        example.RISK_RADIUS,
    )
    assert problem.is_dpp()
    assert len(problem.constraints) == 5 + n_horizons

    rng = np.random.default_rng(51)
    parameter_values = {
        "expected_returns": rng.normal(scale=2e-4, size=(n_horizons, n_assets)),
        "half_spread": np.array([4e-5, 6e-5, 8e-5]),
        "current_weights": np.array([0.01, -0.02, 0.01]),
        "held_weights": np.zeros(n_assets),
        "execution_allowed": np.ones(n_assets),
        "trade_allowed": np.ones((n_horizons, n_assets)),
        "risk_radius": example.RISK_RADIUS,
        **{f"risk_factor_{h}": np.eye(n_assets) for h in range(n_horizons)},
    }
    for parameter in problem.parameters():
        parameter.value = parameter_values[parameter.name()]

    problem.solve(
        solver=cp.CLARABEL,
        presolve_enable=False,
        tol_gap_abs=1e-10,
        tol_gap_rel=1e-10,
        tol_feas=1e-10,
    )
    assert problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}

    variables = {variable.name(): variable for variable in problem.variables()}
    assert set(variables) == {"weights", "previous_weights", "queued", "held"}
    weights = np.asarray(variables["weights"].value)
    current = parameter_values["current_weights"]
    delta = weights - np.vstack([current, weights[:-1]])
    direct_cost = np.sum(parameter_values["half_spread"] * np.abs(delta))
    expected_objective = (
        -np.sum(parameter_values["expected_returns"] * weights) + direct_cost
    )
    np.testing.assert_allclose(
        float(problem.value),
        expected_objective,
        rtol=2e-7,
        atol=2e-10,
    )


def test_plot_diagnostics_shows_every_figure_and_cumsums_objective():
    source = inspect.getsource(example._plot_diagnostics)
    lines = source.splitlines()
    tight_layout_lines = [
        index for index, line in enumerate(lines) if "fig.tight_layout()" in line
    ]
    assert tight_layout_lines
    for index in tight_layout_lines:
        assert lines[index + 1].strip() == "plt.show()"

    assert '_cum(values["mpo_objective"])' in source
    assert '"mpo_objective.png"' in source


def _native(formula, data, tmp_path):
    from trading_dsl_engine.cpp_stream import compile_formula
    runtime = compile_formula(formula, data, n_instruments=next(iter(data.values())).shape[1])
    return runtime.run(out_path=tmp_path / 'ablation').load()


def test_features_are_scratch10_alphas_times_return_vol(monkeypatch, tmp_path):
    from flows.utils import ewm_std, ts_zscore
    from trading_dsl_engine.base.dsl import abs, cat, where, xs_rank
    from trading_dsl_engine.cpp_stream import xs_gauss
    monkeypatch.setattr(example, 'FEATURE_SPANS', (4,))
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 12)
    rng = np.random.default_rng(52026)
    r = rng.normal(0, .002, (150, 3))
    r[25:29] = 0
    r[50, 0] = .08
    r[77, 1] = np.nan
    rets = var('returns')
    clean = where((abs(rets) <= .05) & (rets != 0), rets, float('nan'))
    alpha = xs_gauss(xs_rank(-ts_zscore(clean, 4)))
    expected = cat(alpha * ewm_std(clean, span=12))
    actual = _native({'actual': example._formula(rets)['features'], 'expected': expected},
                     {'returns': r}, tmp_path)
    np.testing.assert_allclose(actual['actual'], actual['expected'], rtol=1e-12, atol=1e-14, equal_nan=True)


def _ic_oracle(s, r, mask, w, lag, hz):
    """Independent pandas clock-time attribution, including staggered sessions."""
    import pandas as pd
    s, r, opened = pd.DataFrame(s), pd.DataFrame(r), pd.DataFrame(mask).fillna(0).ne(0)
    raw = pd.DataFrame(np.broadcast_to(w, r.shape).copy())
    q = raw.div(raw.sum(axis=1).replace(0, np.nan), axis=0)
    q = q.where(opened & np.isfinite(q)).ffill().fillna(0)
    candidate = s.shift(lag)
    p = candidate.where(opened & np.isfinite(candidate)).ffill().fillna(0)
    clean = r.where(np.isfinite(r), 0)
    x = p.shift(hz)
    y = clean.rolling(hz, min_periods=hz).mean()
    weights = q.shift(hz)
    realized = (clean * (p * q).rolling(hz, min_periods=hz).mean().shift()).sum(axis=1)
    target_time = (x * y * weights).sum(axis=1)
    return x.to_numpy(), y.to_numpy(), weights.to_numpy(), realized.to_numpy(), target_time.to_numpy()


def test_ic1_training_terms_follow_held_positions_and_origin_weights(tmp_path):
    import flows.alpha_search as alpha_search
    import pytest
    helper = getattr(alpha_search, '_ic1_terms', None)
    assert helper is not None, 'ic1 must expose its own shared training terms'
    rng = np.random.default_rng(8302)
    rows, assets = 100, 3
    signal = rng.normal(size=(rows, assets))
    returns = rng.normal(scale=.002, size=(rows, assets))
    mask = np.ones_like(returns)
    mask[23:45, 0] = 0
    mask[27:39, 1] = np.nan
    mask[30:48, 2] = 0
    returns[~(mask == 1)] = np.nan
    returns[45, 0] = .035
    signal[53:57, 1] = np.nan
    weights = rng.uniform(.1, 2, size=returns.shape)
    weights[55:59] = 0
    data = {'signal': signal, 'returns': returns, 'mask': mask, 'liquidity': weights}
    formulas, expected = {}, {}
    for lag, hz in [(0, 1), (1, 1), (2, 4)]:
        for weighted in [False, True]:
            key = f'{lag}_{hz}_{weighted}'
            x, y, q = helper(var('signal'), roll_rets=var('returns'),
                             is_tradable=var('mask'), w=var('liquidity') if weighted else 1,
                             lag=lag, hz=hz)
            formulas[key] = {'x': x, 'y': y, 'w': q}
            expected[key] = _ic_oracle(signal, returns, mask, weights if weighted else 1, lag, hz)
    result = _native(formulas, data, tmp_path)
    for key, want in expected.items():
        for field, target in zip(('x', 'y', 'w'), want[:3]):
            np.testing.assert_allclose(result[key][field], target, rtol=2e-12, atol=2e-14, equal_nan=True)


def test_beta_one_reconciles_raw_alpha_ic_and_ic1(monkeypatch, tmp_path):
    monkeypatch.setattr(example, 'FEATURE_SPANS', (4,))
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 12)
    rng = np.random.default_rng(90426)
    returns = rng.normal(scale=.001, size=(220, 3))
    mask = np.ones_like(returns)
    mask[60:80, 0] = 0
    mask[64:87, 1] = np.nan
    returns[~(mask == 1)] = np.nan
    returns[80, 0] = .03
    hs = rng.uniform(.0001, .0003, size=returns.shape)
    hs[95:99] = np.inf  # all-zero liquidity observation: hold previous weights
    data = {'returns': returns, 'is_tradable_out0': mask, 'vw_halfspread_out0': hs}
    formula = example._formula(var('returns'), beta_override=1.0)
    result = _native({'alpha': formula['alpha_pnl'], 'yhat': formula['yhat_pnl'],
                      'forecasts': formula['forecast_diagnostics'], 'features': formula['features']}, data, tmp_path)
    for key in result['alpha']:
        for name in ['ic', 'ic1']:
            alpha_pnl = next(iter(result['alpha'][key][name].values()))
            yhat_pnl = result['yhat'][key][name]
            np.testing.assert_allclose(yhat_pnl, alpha_pnl, rtol=2e-12, atol=2e-14, equal_nan=True)
            assert np.any(np.abs(yhat_pnl) > 1e-6), 'ablation must not pass on all-zero warmup output'
        start, end = map(int, key.split('_'))
        rates = result['forecasts'][key]['yhat_rate']
        totals = result['forecasts'][key]['yhat_total']
        np.testing.assert_allclose(rates, result['features'][..., 0], equal_nan=True)
        np.testing.assert_allclose(totals, rates * (end - start), equal_nan=True)


def _session_data(rows=360, assets=3, staggered=False):
    rng = np.random.default_rng(777)
    r = rng.normal(scale=.001, size=(rows, assets))
    # A missing-return outage freezes risk before the scheduled close so this
    # test isolates execution. Valid flat prices are tested separately, as
    # is infeasibility when frozen holdings cannot satisfy a new risk budget.
    r[40:120] = np.nan
    opened = np.ones_like(r)
    ts = np.repeat((1_780_000_000_000_000.0 + np.arange(rows) * example.MINUTE_US)[:, None], assets, axis=1)
    starts, ends = np.zeros_like(r), np.zeros_like(r)
    next_starts, next_ends = np.zeros_like(r), np.zeros_like(r)
    for i in range(assets):
        close, reopen = (120 + i * 3, 180 + i * 2) if staggered else (120, 180)
        opened[close:reopen, i] = 0
        r[close:reopen, i] = 0
        r[reopen, i] = .012 * (1 if i % 2 == 0 else -1)
        starts[:, i] = ts[0, i]
        ends[:, i] = ts[close, i]
        next_starts[:, i] = ts[reopen, i]
        next_ends[:, i] = ts[-1, i] + 1000 * example.MINUTE_US
        starts[reopen:, i] = ts[reopen, i]
        ends[reopen:, i] = next_ends[-1, i]
        ts[close:reopen, i] = np.nan
    # A reopening return excluded by scratch10's feature filter must still
    # appear in actual portfolio PnL and risk, never silently disappear.
    r[180, 2] = .075
    return {'returns': r, 'is_tradable_out0': opened,
            'vw_halfspread_out0': np.full_like(r, 1e-5), '_ev_ts': ts,
            'session_start0': starts, 'session_end0': ends,
            'next_session_start0': next_starts, 'next_session_end0': next_ends}


def test_full_native_mpo_executes_next_bar_and_holds_through_gaps(monkeypatch, tmp_path):
    import pandas as pd
    monkeypatch.setattr(example, 'FEATURE_SPANS', (4, 16))
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 24)
    monkeypatch.setattr(example, 'RIDGE_HL', 30)
    monkeypatch.setattr(example, 'RISK_SPAN', 30)
    monkeypatch.setattr(example, 'RISK_MIN_PERIODS', 8)
    data = _session_data()
    formula = example._formula(var('returns'))
    out = _native(formula, data, tmp_path)
    assert np.all(np.isin(out['status'], [1, 4])), np.unique(out['status'], return_counts=True)
    planned = out['planned_weights']
    executed = out['weights']
    wanted = pd.DataFrame(planned).shift().where(data['is_tradable_out0'] != 0).ffill().fillna(0).to_numpy()
    np.testing.assert_allclose(executed, wanted, atol=1e-7, rtol=1e-7)
    held = pd.DataFrame(executed).shift().fillna(0).to_numpy()
    pnl = np.sum(held * np.nan_to_num(data['returns']), axis=1)
    np.testing.assert_allclose(out['mpo_gross_pnl'], pnl, atol=1e-11, rtol=1e-8)
    cost = np.sum(np.abs(executed - held) * data['vw_halfspread_out0'], axis=1)
    np.testing.assert_allclose(out['mpo_realized_cost'], cost, atol=1e-11, rtol=1e-8)
    np.testing.assert_allclose(out['mpo_net_pnl'], pnl - cost, atol=1e-11, rtol=1e-8)
    assert np.max(np.abs(executed)) > .001, 'must exercise real nonzero portfolio state'
    assert out['expected_returns'].shape == (360, 3, 7)
    assert out['trade_allowed'].shape == (360, 3, 7)
    assert np.isfinite(out['expected_returns']).all()
    for i in range(3):
        closed = data['is_tradable_out0'][:, i] == 0
        np.testing.assert_allclose(executed[closed, i], held[closed, i], atol=1e-8)
    # Reopening gap is full-sized in the width-one risk observation.
    assert out['risk_samples']['1_2'][180, 2] == .075
    # Column-major binding must receive L.T, so the SOC equals sqrt(w' S w).
    for h, (key, factor) in enumerate(out['risk_factors'].items()):
        expected_risk_vector = np.einsum('tji,tj->ti', factor, out['planned_path'][:, :, h])
        np.testing.assert_allclose(out['risk'][key][:, 1:], expected_risk_vector, atol=2e-9, rtol=2e-8)
        assert np.max(np.linalg.norm(expected_risk_vector, axis=1)) <= example.RISK_RADIUS + 2e-7
    # At t=119, every return in (1,2] and (2,4] is inside the closure.
    np.testing.assert_array_equal(out['expected_returns'][119, :, :2], 0)


def test_diagnostics_match_scratch10_instrument_sum_and_default_pnl(monkeypatch, tmp_path):
    from flows.alpha_search import ic, ic1, default_alpha_pnl
    from flows.utils import ts_zscore
    from trading_dsl_engine.base.dsl import abs, where, xs_rank
    from trading_dsl_engine.cpp_stream import xs_gauss
    monkeypatch.setattr(example, 'FEATURE_SPANS', (4,))
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 12)
    rng = np.random.default_rng(901)
    returns = rng.normal(scale=.001, size=(130, 3))
    mask = np.ones_like(returns)
    mask[50:66] = 0
    returns[50:66] = 0
    returns[66] = [.013, -.01, .02]
    data = {'returns': returns, 'is_tradable_out0': mask}
    cleaned = where((abs(var('returns')) <= .05) & (var('returns') != 0), var('returns'), float('nan'))
    alpha = xs_gauss(xs_rank(-ts_zscore(cleaned, 4)))
    kwargs = dict(roll_rets=cleaned, is_tradable=var('is_tradable_out0').fillna(0), hl=12, lag=1, hz=1)
    formula = example._formula(var('returns'), fit_weights=1)
    result = _native({'example': formula['alpha_pnl']['1_2'],
                      'scratch_ic': ic(alpha, w=1, **kwargs),
                      'scratch_ic1': ic1(alpha, w=1, **kwargs),
                      'scratch_default': default_alpha_pnl(alpha, **kwargs)}, data, tmp_path)
    for name in ['ic', 'ic1']:
        actual = next(iter(result['example'][name].values()))
        expected = np.nansum(result['scratch_' + name], axis=1)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(actual, np.nansum(result['scratch_default'], axis=1), atol=1e-12, rtol=1e-12)


def _ridge_oracle(x, y, weight, half_life, penalty=.1):
    """Independent pairwise EWM normal equations on observed rows."""
    rows, _, features = x.shape
    alpha = 1 - 2 ** (-1 / half_life)
    xx, xy = np.zeros((features, features)), np.zeros(features)
    seen_xx, seen_xy = np.zeros_like(xx, dtype=bool), np.zeros_like(xy, dtype=bool)
    beta = np.zeros(features)
    result = []
    for t in range(rows):
        for j in range(features):
            valid = np.isfinite(x[t, :, j]) & np.isfinite(y[t]) & np.isfinite(weight[t])
            if valid.any():
                value = np.sum(x[t, valid, j] * weight[t, valid] * y[t, valid])
                xy[j] = xy[j] + alpha * (value - xy[j]) if seen_xy[j] else value
                seen_xy[j] = True
            for k in range(features):
                valid = np.isfinite(x[t, :, j]) & np.isfinite(x[t, :, k]) & np.isfinite(weight[t])
                if valid.any():
                    value = np.sum(x[t, valid, j] * weight[t, valid] * x[t, valid, k])
                    xx[j, k] = xx[j, k] + alpha * (value - xx[j, k]) if seen_xx[j, k] else value
                    seen_xx[j, k] = True
        system = xx + penalty * np.diag(np.diag(xx))
        if np.any(system):
            beta = np.linalg.pinv(system) @ xy
        result.append(beta.copy())
    return np.asarray(result)


def test_fitted_beta_and_yhat_match_normal_equations_without_future_leakage(monkeypatch, tmp_path):
    monkeypatch.setattr(example, 'FEATURE_SPANS', (4, 16))
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 24)
    monkeypatch.setattr(example, 'RIDGE_HL', 30)
    data = _session_data()
    formula = example._formula(var('returns'))
    roots = {'fit': formula['forecast_diagnostics'], 'features': formula['features']}
    out = _native(roots, data, tmp_path / 'original')
    for block in out['fit'].values():
        expected = _ridge_oracle(block['fit_x'], block['target'], block['sample_weight'], 30)
        np.testing.assert_allclose(block['beta_fitted'], expected, rtol=2e-8, atol=2e-10)
        expected_yhat = np.einsum('tnf,tf->tn', out['features'], expected)
        np.testing.assert_allclose(block['yhat_rate'], expected_yhat, rtol=2e-8, atol=2e-12, equal_nan=True)
    shocked = {key: value.copy() for key, value in data.items()}
    cut = 220
    shocked['returns'][cut:] *= -7
    changed = _native(roots, shocked, tmp_path / 'shocked')
    for key, block in out['fit'].items():
        for field in ('fit_x', 'target', 'sample_weight', 'beta_fitted', 'yhat_rate'):
            np.testing.assert_array_equal(block[field][:cut], changed['fit'][key][field][:cut])


def test_ic_dual_reconciles_totals_after_tail_flush_not_rowwise(monkeypatch, tmp_path):
    from flows.alpha_search import ic, ic1
    from flows.utils import ewm_std
    rng = np.random.default_rng(812)
    rows, assets, hz, lag = 150, 3, 4, 2
    r = rng.normal(scale=.001, size=(rows, assets))
    r[-hz:] = 0  # no unobserved remaining return contribution at the tail
    signal = rng.normal(size=r.shape)
    mask = np.ones_like(r)
    mask[50:76, 0] = np.nan
    mask[60:80, 1:] = 0
    r[~(mask == 1)] = 0
    r[76, 0] = .015
    weights = rng.uniform(.1, 3, size=r.shape)
    weights[90:93] = 0
    vol = ewm_std(var('returns'), span=8)
    # Multiplying first cancels public ic's normalization after warmup.
    alpha = var('signal') * vol
    kwargs = dict(roll_rets=var('returns'), is_tradable=var('mask'), hl=8, lag=lag, hz=hz, w=var('w'))
    result = _native({'ic': ic(alpha, **kwargs), 'ic1': ic1(alpha, **kwargs),
                      'scaled_signal': alpha / vol},
                     {'returns': r, 'signal': signal, 'mask': mask, 'w': weights}, tmp_path)
    oracle = _ic_oracle(result['scaled_signal'], r, mask, weights, lag, hz)
    for name, expected in zip(('ic', 'ic1'), oracle[3:]):
        np.testing.assert_allclose(np.nansum(result[name], axis=1), assets * expected, atol=2e-12, rtol=2e-12)
    a, b = (np.nansum(result[name], axis=1) for name in ('ic', 'ic1'))
    np.testing.assert_allclose(a.sum(), b.sum(), atol=2e-12, rtol=2e-12)
    assert np.max(np.abs(a-b)) > 1e-4, 'the two attribution clocks are not pointwise equal for hz>1'


def test_roll_rets_places_the_entire_gap_on_reopening(tmp_path):
    from flows.pov import RollRets
    rows, assets = 12, 2
    clock = np.repeat((1_780_000_000_000_000. + np.arange(rows)*example.MINUTE_US)[:, None], assets, axis=1)
    mask = np.ones((rows, assets))
    mask[4:9] = 0
    price = np.full_like(mask, 100.)
    price[4:9] = 999.  # these closed-session marks must not enter roll_rets
    price[9:] = [110., 90.]
    data = {'vwap_mp_out0': price, 'vwap_mp_out1': price+5,
            'wdte_out0': np.full_like(price, 10.),
            'is_tradable_out0': mask, 'is_tradable_out1': mask,
            '_ev_ts': clock, 'session_start0': np.full_like(clock, clock[0, 0]),
            'session_end0': np.full_like(clock, clock[4, 0]),
            'volume_out0': 10*mask}
    actual = _native(RollRets().roll_rets(), data, tmp_path)
    np.testing.assert_allclose(actual[1:9], 0, atol=1e-14)
    np.testing.assert_allclose(actual[9], [.1, -.1], atol=1e-14)
    np.testing.assert_allclose(actual[10:], 0, atol=1e-14)


def test_risk_keeps_valid_zero_returns_but_not_closed_zero_placeholders(monkeypatch, tmp_path):
    monkeypatch.setattr(example, 'FEATURE_SPANS', (4,))
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 4)
    returns = np.zeros((8, 3))
    returns[0] = [.01, -.01, .02]
    returns[6] = [.03, -.02, .01]
    opened = np.ones_like(returns)
    opened[3:6] = 0
    formula = example._formula(var('returns'))
    out = _native(formula['risk_samples']['1_2'],
                  {'returns': returns, 'is_tradable_out0': opened}, tmp_path)
    np.testing.assert_array_equal(out[1:3], 0.0)
    assert np.isnan(out[3:6]).all()
    np.testing.assert_array_equal(out[6], returns[6])
