from __future__ import annotations

import inspect

import cvxpy as cp
import numpy as np
import pytest

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
        assert feature.fn == "xs_generalized_rank"  # cpp_stream xs_gauss
        rank = feature.args[0]
        assert rank.fn == "xs_rank"
        assert rank.args[0].fn == "sub"
        assert rank.args[0].args[0].value == 0.0

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
        *factors,
        np.ones((n_horizons, n_assets)),
        np.zeros(n_assets),
        np.ones(n_assets),
        np.zeros((n_horizons, n_assets)),
        example.RISK_RADIUS,
    )
    assert problem.is_dpp()
    assert len(problem.constraints) == 7 + n_horizons

    rng = np.random.default_rng(51)
    parameter_values = {
        "expected_returns": rng.normal(scale=2e-4, size=(n_horizons, n_assets)),
        "half_spread": np.array([4e-5, 6e-5, 8e-5]),
        "current_weights": np.array([0.01, -0.02, 0.01]),
        "trade_allowed": np.ones((n_horizons, n_assets)),
        "last_weights": np.zeros(n_assets),
        "execution_allowed": np.ones(n_assets),
        "gap_volatility": np.zeros((n_horizons, n_assets)),
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
    assert set(variables) == {"weights", "previous_weights", "previous_plan", "previous_actual", "gap_exposure"}
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


def _native_values(formula, data, tmp_path):
    from trading_dsl_engine.cpp_stream import compile_formula

    runtime = compile_formula(formula, data, n_instruments=next(iter(data.values())).shape[1])
    return runtime.run(out_path=tmp_path / 'result.npy').load()


def test_default_features_match_scratch_10_values_not_just_sign(tmp_path):
    from flows.utils import ts_zscore, replace
    from trading_dsl_engine.base.dsl import abs as dsl_abs, cat, where, xs_rank
    from trading_dsl_engine.cpp_stream import xs_gauss

    rng = np.random.default_rng(903)
    returns = rng.normal(0, 2e-4, (900, 9))
    returns[300:360] = 0.0
    returns[360] = np.linspace(-.08, .09, 9)
    returns[440, 2] = np.nan
    r = var('returns')
    clean = where(dsl_abs(r) <= .05, replace(r, 0, float('nan')), float('nan'))
    scratch = cat(*(xs_gauss(xs_rank(-ts_zscore(clean, span))) for span in (4, 16, 64, 256)))
    values = _native_values({'actual': example._formula(r)['features'], 'scratch': scratch},
                            {'returns': returns}, tmp_path)
    np.testing.assert_allclose(values['actual'], values['scratch'], rtol=0, atol=0, equal_nan=True)


def test_formula_exposes_regression_pairs_and_vol_scaled_inputs():
    formula = example._formula(var('returns'))
    assert {'volatility', 'scaled_features', 'fit', 'yhat', 'expected_returns'} <= formula.keys()


def test_ridge_ignores_an_unobserved_session_in_observation_time(tmp_path):
    from trading_dsl_engine.base.dsl import Ridge, get_beta

    x = np.ones((102, 2))
    y = np.ones((102, 2))
    x[1:101] = np.nan
    y[1:101] = np.nan
    y[101] = 3.0
    formula = get_beta(Ridge(var('x'), y=var('y'), hl=2, lambda_=0.0))
    values = _native_values(formula, {'x': x, 'y': y}, tmp_path)
    alpha = 1 - .5 ** .5
    np.testing.assert_allclose(values[-1], 1 + 2 * alpha, rtol=1e-13, atol=1e-13)


def _pair_fixture(rows=520, lanes=5):
    rng = np.random.default_rng(519)
    alpha = rng.normal(size=(rows, lanes))
    sigma = np.geomspace(2e-5, 2e-3, lanes)[None, :] * np.ones((rows, lanes))
    sigma[rows // 2:] *= 2
    mask = np.ones((rows, lanes))
    mask[160:200] = 0
    mask[180:190, 1] = np.nan
    mask[330:333, 2] = np.nan
    r = rng.normal(size=(rows, lanes)) * sigma
    r[160:200] = 0
    r[200] = [.09, -.08, .015, -.025, .007]
    r[350, 3] = np.nan
    alpha[240:243, 1] = np.nan
    w = rng.uniform(.1, 2, size=(rows, lanes))
    w[110:114] = 0
    w[210, 2] = 0
    return {'alpha': alpha, 'sigma': sigma, 'returns': r,
            'is_tradable_out0': mask, 'sample_weight': w,
            'vw_halfspread_out0': np.full_like(r, 4e-5)}


def _held_numpy(s, mask, lag):
    import pandas as pd
    candidate = pd.DataFrame(s).shift(lag)
    return candidate.where(np.nan_to_num(mask) != 0).ffill().fillna(0)


def test_canonical_pairs_match_independent_pandas_and_ic1_crossproducts(tmp_path, monkeypatch):
    import pandas as pd
    from flows.utils import ewm_std
    from flows.alpha_search import ic1
    from trading_dsl_engine.base.dsl import isfinite, where

    monkeypatch.setattr(example, 'IC_VOL_SPAN', 16)
    data = _pair_fixture()
    r, a, sigma, mask, w = (var(k) for k in ('returns', 'alpha', 'sigma', 'is_tradable_out0', 'sample_weight'))
    lag, hz = 2, 4
    model = example._forecast((a,), sigma=sigma, returns=r, tradable=mask, w=w,
                              lag=lag, hz=hz, target_valid=where(isfinite(r), 1., 0.))
    # ic1 divides by its own sigma; cancel only that normalization to audit X*Y.
    oracle = ic1(a * sigma * ewm_std(r, span=16), roll_rets=r,
                 is_tradable=mask, hl=16, lag=lag, hz=hz, w=w).mean(axis=[1])
    values = _native_values({'model': model, 'cross': oracle}, data, tmp_path)
    got = values['model']
    pos = _held_numpy(data['alpha'] * data['sigma'], data['is_tradable_out0'], lag)
    norm = data['sample_weight'] / np.where(data['sample_weight'].sum(axis=1, keepdims=True) != 0,
                                          data['sample_weight'].sum(axis=1, keepdims=True), np.nan)
    held_w = _held_numpy(norm, data['is_tradable_out0'], 0)
    expected_x = pos.shift(hz).to_numpy()
    expected_y = pd.DataFrame(np.nan_to_num(data['returns'])).rolling(hz).mean().to_numpy()
    expected_w = held_w.shift(hz).to_numpy()
    np.testing.assert_allclose(got['canonical_x'][..., 0], expected_x, atol=1e-15, equal_nan=True)
    np.testing.assert_allclose(got['canonical_y'], expected_y, atol=1e-15, equal_nan=True)
    np.testing.assert_allclose(got['canonical_weights'], expected_w, atol=1e-15, equal_nan=True)
    cross = np.sum(expected_x * expected_y * expected_w, axis=1)
    np.testing.assert_allclose(np.asarray(values['cross']).reshape(-1)[50:], cross[50:], atol=1e-16, rtol=1e-12)
    eligible = np.asarray(got['eligible'], dtype=bool)
    assert not eligible[350:354, 3].any()  # a missing Y must also exclude XX
    assert np.isnan(got['x'][~eligible]).all()
    assert np.isnan(got['weights'][~eligible]).all()
    # Independent weighted one-feature exponentially weighted normal equations.
    alpha = 1 - .5 ** (1 / example.RIDGE_HL)
    xx = xy = None
    beta = 0.
    expected_beta = []
    for xrow, yrow, wrow in zip(got['x'][..., 0], got['y'], got['weights']):
        valid = np.isfinite(xrow) & np.isfinite(yrow) & np.isfinite(wrow)
        if valid.any():
            new_xx = np.sum(wrow[valid] * xrow[valid] ** 2)
            new_xy = np.sum(wrow[valid] * xrow[valid] * yrow[valid])
            xx = new_xx if xx is None else xx + alpha * (new_xx - xx)
            xy = new_xy if xy is None else xy + alpha * (new_xy - xy)
            beta = xy / (xx * (1 + example.RIDGE_LAMBDA)) if xx > 0 else beta
        expected_beta.append(beta)
    np.testing.assert_allclose(np.asarray(got['beta']).reshape(-1), expected_beta, atol=1e-11, rtol=1e-10)


def test_beta_one_yhat_ic_and_ic1_reconcile_for_every_horizon(tmp_path, monkeypatch):
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 16)
    data = _pair_fixture()
    model = example._formula(var('returns'), features=(var('alpha'),),
                             volatility=var('sigma'), fit_weights=var('sample_weight'),
                             beta_override=1.)
    values = _native_values({key: model[key] for key in ('alpha_pnl', 'yhat_pnl', 'yhat', 'scaled_features')},
                            data, tmp_path)
    np.testing.assert_allclose(values['scaled_features'][..., 0], data['alpha'] * data['sigma'], equal_nan=True)
    for key in values['yhat_pnl']:
        np.testing.assert_allclose(values['yhat'][key], data['alpha'] * data['sigma'], equal_nan=True)
        for kind in ('ic', 'ic1'):
            np.testing.assert_allclose(values['yhat_pnl'][key][kind],
                                       values['alpha_pnl'][key][kind]['feature_0'],
                                       atol=2e-12, rtol=2e-12, equal_nan=True)


def test_known_rho_recovered_with_heterogeneous_volatility_and_missing_targets(tmp_path, monkeypatch):
    from trading_dsl_engine.base.dsl import isfinite, where

    monkeypatch.setattr(example, 'RIDGE_HL', 30)
    monkeypatch.setattr(example, 'RIDGE_LAMBDA', 0.)
    data = _pair_fixture()
    rng = np.random.default_rng(72)
    data['alpha2'] = rng.normal(size=data['alpha'].shape)
    beta = np.array([.17, -.09])
    latent = data['sigma'] * (beta[0] * data['alpha'] + beta[1] * data['alpha2'])
    data['returns'] = np.vstack([np.full((2, 5), np.nan), latent[:-2]])
    data['returns'][350:355, 3] = np.nan
    valid = isfinite(var('returns')) & (var('is_tradable_out0').fillna(0) != 0)
    # Exclude training around closures, whose execution-origin features are held.
    source_open = (var('is_tradable_out0').fillna(0) != 0).shift(1)
    model = example._forecast((var('alpha'), var('alpha2')), sigma=var('sigma'),
                              returns=var('returns'), tradable=var('is_tradable_out0'),
                              w=var('sample_weight'), lag=1, hz=1,
                              target_valid=where(valid & source_open, 1., 0.))
    got = _native_values(model, data, tmp_path)
    # A missing feature causes canonical hold, so remove the deliberately stale rows
    # for this exact-coefficient experiment rather than pretending they are new X.
    eligible = got['eligible'].astype(bool)
    predicted_training = np.einsum('tif,f->ti', np.nan_to_num(got['x']), beta)
    mismatch = eligible & (np.abs(predicted_training - np.nan_to_num(got['y'])) > 1e-12)
    assert not mismatch.any(), np.argwhere(mismatch)[:10]
    np.testing.assert_allclose(got['beta'][80:], np.broadcast_to(beta, got['beta'][80:].shape),
                               atol=1e-11, rtol=1e-10)


def _market_fixture(rows=440, lanes=3):
    rng = np.random.default_rng(717)
    t = np.arange(rows)
    epoch = 1_787_000_000_000_000.
    minute = example.MINUTE_US
    timestamps = epoch + t[:, None] * minute + np.zeros((rows, lanes))
    mask = np.ones((rows, lanes))
    mask[160:200] = 0
    if rows > 274:
        mask[270:274, 1] = np.nan
    rets = rng.normal(scale=2e-4, size=(rows, lanes))
    rets[np.nan_to_num(mask) == 0] = 0
    if rows > 200:
        rets[200] = np.linspace(-.08, .12, lanes)
    if rows > 274:
        rets[274, 1] = .006
    hs = np.full_like(rets, 1e-5)
    hs[np.nan_to_num(mask) == 0] = np.nan
    if rows > 310:
        hs[310, 0] = np.nan
    # Known calendar data, never the future realized tradability mask.
    def field(before, after):
        return epoch + np.where(t[:, None] < 200, before, after) * minute + np.zeros_like(rets)
    return {
        'returns': rets, 'is_tradable_out0': mask, 'vw_halfspread_out0': hs,
        '_ev_ts': timestamps,
        'session_start0': field(-1240, 200), 'session_end0': field(160, 1600),
        'next_session_start0': field(200, 1640), 'next_session_end0': field(1600, 3040),
        'alpha': np.sin(t[:, None] / 19 + np.linspace(0, 2 * np.pi, lanes, endpoint=False)),
        'sigma': np.full_like(rets, 2e-4),
    }


def test_first_planned_vwap_trade_cannot_earn_the_next_observed_vwap_return(tmp_path, monkeypatch):
    import os
    import pytest
    if not os.environ.get('CLARABEL_STATIC_LIBRARY'):
        pytest.skip('native Clarabel library is required')
    monkeypatch.setattr(example, 'RISK_MIN_PERIODS', 4)
    monkeypatch.setattr(example, 'RISK_SPAN', 32)
    data = _market_fixture(rows=20)
    formula = example._formula(var('returns'), features=(var('alpha'),),
                               volatility=var('sigma'), fit_weights=1., beta_override=.2)
    got = _native_values({k: formula[k] for k in ('weights', 'mpo_gross_pnl', 'status')}, data, tmp_path)
    assert np.abs(got['weights']).max() > .001  # ensure this is not a flat-portfolio test
    np.testing.assert_allclose(got['mpo_gross_pnl'][:2], 0., atol=1e-9)


def test_calendar_counts_match_bruteforce_without_looking_at_future_masks(tmp_path):
    data = _market_fixture()
    session = example._session_plan(var('is_tradable_out0').fillna(0), var('vw_halfspread_out0'))
    from trading_dsl_engine.base.dsl import cat
    got = _native_values({'counts': cat(*session['regular_counts']),
                          'allowed': session['trade_allowed'], 'execute': session['execution_allowed']},
                         data, tmp_path)
    def opened(offset):
        ts = data['_ev_ts'] + offset * example.MINUTE_US
        return ((ts >= data['session_start0']) & (ts < data['session_end0'])) | (
            (ts >= data['next_session_start0']) & (ts < data['next_session_end0']))
    for h, (start, end) in enumerate(zip(example.TRADE_STARTS, example.HORIZONS)):
        expected = sum((opened(j) & opened(j - 1)).astype(float) for j in range(start + 1, end + 1))
        np.testing.assert_array_equal(got['counts'][..., h], expected)
    assert not np.asarray(got['counts'])[170, :, :4].any()
    assert got['execute'][200].all()
    assert got['execute'][310, 0] == 0  # a NaN quote never makes trading free
    assert not got['execute'][270:274, 1].any()
    assert got['allowed'][270:274, 1].all()  # future realized masks are unknown


def test_actual_holdings_and_pnl_follow_delayed_fills_and_hold_on_nan_masks(tmp_path, monkeypatch):
    import os
    import pytest
    if not os.environ.get('CLARABEL_STATIC_LIBRARY'):
        pytest.skip('native Clarabel library is required')
    monkeypatch.setattr(example, 'RISK_MIN_PERIODS', 4)
    monkeypatch.setattr(example, 'RISK_SPAN', 32)
    data = _market_fixture()
    formula = example._formula(var('returns'), features=(var('alpha'),),
                               volatility=var('sigma'), fit_weights=1., beta_override=.03)
    keys = ('weights', 'planned_weights', 'execution_allowed', 'status',
            'mpo_gross_pnl', 'mpo_trading_cost', 'mpo_net_pnl')
    got = _native_values({k: formula[k] for k in keys}, data, tmp_path)
    assert np.isfinite(got['weights']).all()
    previous_actual = np.zeros(3)
    expected = []
    for t in range(len(data['returns'])):
        previous_plan = np.zeros(3) if t == 0 else got['planned_weights'][t - 1]
        actual = np.where(got['execution_allowed'][t] != 0, previous_plan, previous_actual)
        expected.append(actual)
        previous_actual = actual
    expected = np.asarray(expected)
    np.testing.assert_allclose(got['weights'], expected, atol=2e-6, rtol=2e-7)
    previous = np.vstack([np.zeros((1, 3)), expected[:-1]])
    gross = np.sum(previous * data['returns'], axis=1)
    cost = np.sum(np.abs(expected - previous) * np.nan_to_num(data['vw_halfspread_out0']), axis=1)
    np.testing.assert_allclose(got['mpo_gross_pnl'], gross, atol=3e-7, rtol=2e-7)
    np.testing.assert_allclose(got['mpo_trading_cost'], cost, atol=1e-8, rtol=2e-7)
    np.testing.assert_allclose(got['mpo_net_pnl'], gross - cost, atol=3e-7, rtol=2e-7)
    np.testing.assert_allclose(got['weights'][160:200], np.broadcast_to(expected[159], (40, 3)), atol=2e-6)
    assert abs(gross[200]) > 1e-5  # real 8%-12% gaps were not filtered out of PnL


def test_gap_events_have_separate_unclipped_risk_and_canonical_fit_masks(tmp_path, monkeypatch):
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 16)
    monkeypatch.setattr(example, 'RISK_MIN_PERIODS', 4)
    data = _market_fixture()
    formula = example._formula(var('returns'), features=(var('alpha'),),
                               volatility=var('sigma'), fit_weights=1., beta_override=.03)
    assert {'gap_event', 'gap_minutes', 'gap_sigma', 'planned_gap_sigma', 'ordinary_returns'} <= formula.keys()
    keys = ('gap_event', 'gap_minutes', 'gap_sigma', 'planned_gap_sigma', 'ordinary_returns', 'signal_returns')
    got = _native_values({**{k: formula[k] for k in keys},
                          'eligible': formula['fit']['1_2']['eligible']}, data, tmp_path)
    np.testing.assert_array_equal(got['gap_event'][200], 1.)
    np.testing.assert_array_equal(got['gap_minutes'][200], 41.)
    assert np.isnan(got['ordinary_returns'][200]).all()
    assert np.isnan(got['signal_returns'][200, [0, 2]]).all()
    assert not got['eligible'][200].any()
    np.testing.assert_allclose(got['gap_sigma'][200], np.abs(data['returns'][200]), atol=1e-14)
    assert (got['planned_gap_sigma'][159, :, 5] > 0).all()  # reopening is 41 minutes away
    assert not got['planned_gap_sigma'][159, :, :5].any()
    assert not got['planned_gap_sigma'][159, :, 6].any()


def test_diagnostics_match_scratch_10_cross_sectional_sum(tmp_path, monkeypatch):
    from flows.alpha_search import ic, ic1
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 16)
    data = _pair_fixture()
    kwargs = dict(roll_rets=var('returns'), is_tradable=var('is_tradable_out0').fillna(0),
                  hl=16, lag=2, hz=4, w=var('sample_weight'))
    got = _native_values({'raw_ic': ic(var('alpha'), **kwargs),
                          'raw_ic1': ic1(var('alpha'), **kwargs),
                          'diagnostics': example._diagnostic_pnls(
                              var('alpha'), **{k:v for k,v in kwargs.items() if k != 'hl'})},
                         data, tmp_path)
    for kind in ('ic', 'ic1'):
        expected = np.nansum(got['raw_' + kind], axis=1)
        np.testing.assert_allclose(got['diagnostics'][kind], expected, atol=1e-12, equal_nan=True)


def test_ic_and_ic1_reconcile_total_pnl_after_complete_tail_not_each_timestamp(tmp_path, monkeypatch):
    monkeypatch.setattr(example, 'IC_VOL_SPAN', 16)
    data = _pair_fixture()
    tail = 8
    for key, value in data.items():
        extra = np.zeros((tail, value.shape[1]))
        data[key] = np.concatenate([value, extra])
    data['alpha'][-tail:] = np.nan
    diagnostics = example._diagnostic_pnls(var('alpha'), roll_rets=var('returns'),
        is_tradable=var('is_tradable_out0').fillna(0), w=var('sample_weight'), lag=2, hz=4)
    got = _native_values(diagnostics, data, tmp_path)
    np.testing.assert_allclose(np.nansum(got['ic']), np.nansum(got['ic1']), atol=2e-12)
    assert np.nansum(np.abs(got['ic'] - got['ic1'])) > .001  # different attribution clocks


def _weekend_fixture(lanes=3):
    """A minute grid, a two-day closure, regime shifts, outages and large gaps."""
    rng = np.random.default_rng(931)
    rows = 3500
    minute = example.MINUTE_US
    epoch = 1_787_000_000_000_000.
    ts = epoch + np.arange(rows)[:, None] * minute + np.zeros((rows, lanes))
    mask = np.ones((rows, lanes))
    mask[180:3060] = 0.
    mask[450:460] = np.nan
    mask[3230:3233, 1] = np.nan
    sigma = np.geomspace(.65, 1.8, lanes) * 2e-4
    raw = np.zeros((rows, lanes))
    for t in range(rows):
        if np.nan_to_num(mask[t]).any():
            raw[t] = -.3 * (raw[t-1] if t else 0.) + rng.normal(size=lanes) * sigma
        if t == 3060:
            raw[t] = np.linspace(-.08, .12, lanes)
        if t == 3200:
            sigma *= 1.6
    raw[3230:3233, 1] = 0.
    raw[3233, 1] = .007
    hs = np.broadcast_to(np.geomspace(8e-6, 1.5e-5, lanes), raw.shape).copy()
    hs[np.nan_to_num(mask) == 0] = np.nan
    hs[3300, 0] = np.nan
    def calendar(before, after):
        return epoch + np.where(np.arange(rows)[:, None] < 3060, before, after) * minute + np.zeros_like(raw)
    data = {'returns': raw, 'is_tradable_out0': mask, 'vw_halfspread_out0': hs,
            '_ev_ts': ts, 'session_start0': calendar(-1260, 3060),
            'session_end0': calendar(180, 4500), 'next_session_start0': calendar(3060, 4540),
            'next_session_end0': calendar(4500, 5980)}
    data['_ev_ts'][500:510] = np.nan  # reconstruct clock without looking forward
    return data


@pytest.mark.parametrize("lanes", [3, 9])
def test_learned_rho_volatility_and_native_mpo_survive_weekend_and_are_causal(tmp_path, monkeypatch, lanes):
    import os
    import pytest
    if not os.environ.get('CLARABEL_STATIC_LIBRARY'):
        pytest.skip('native Clarabel library is required')
    for name, value in [('IC_VOL_SPAN',32), ('RIDGE_HL',120), ('RIDGE_MIN_PERIODS',16),
                        ('RISK_SPAN',128), ('RISK_MIN_PERIODS',16), ('RISK_RADIUS',.01),
                        ('FEATURE_SPANS',(4,16))]:
        monkeypatch.setattr(example, name, value)
    data = _weekend_fixture(lanes)
    formula = example._formula(var('returns'))  # actual estimated sigma and beta, no override
    keys = ('weights','planned_weights','execution_allowed','mpo_gross_pnl','mpo_net_pnl',
            'status','volatility','planned_gap_sigma','gap_sigma')
    outputs = {k:formula[k] for k in keys}
    outputs.update(beta=formula['fit']['1_2']['beta'], count=formula['fit']['1_2']['count'],
                   yhat=formula['yhat']['1_2'], risk=formula['risk']['1_2'])
    runtime = example.compile_formula(outputs, data, n_instruments=lanes)
    source = runtime.generated_cpp.read_text()
    assert source.count('for (std::size_t t = row_begin; t < row_end; ++t)') == 1
    assert source.count('stackdsl::ClarabelNode<') == 1
    got = {k: np.array(v, copy=True) for k,v in runtime.run(out_path=tmp_path/'weekend.npy').load().items()}
    assert np.isin(got['status'], [1.,4.]).all(), np.unique(got['status'],return_counts=True)
    assert np.isfinite(got['yhat'][100:170]).all()
    assert np.abs(got['beta'][100:170]).max() > .001
    assert np.abs(got['weights'][100:170]).max() > .001
    np.testing.assert_array_equal(got['count'][:30], 0.)
    np.testing.assert_allclose(got['weights'][180:3060], np.broadcast_to(got['weights'][179], (2880,lanes)), atol=3e-6)
    np.testing.assert_allclose(got['mpo_gross_pnl'][3060], got['weights'][3059] @ data['returns'][3060], atol=1e-9)
    assert np.all(got['planned_gap_sigma'][179,:,-1] >= example.LONG_GAP_VOL_FLOOR)
    np.testing.assert_allclose(got['gap_sigma'][3060], np.maximum(np.abs(data['returns'][3060]), example.LONG_GAP_VOL_FLOOR), atol=1e-14)
    # Prefix causality, including fit, volatility, holdings, and optimizer state.
    changed = {k:v.copy() for k,v in data.items()}
    cut = 3360
    changed['returns'][cut:] *= -7.
    changed['is_tradable_out0'][cut:, 0] = np.nan
    changed['vw_halfspread_out0'][cut:] *= 20.
    later = _native_values(outputs, changed, tmp_path)
    for k in outputs:
        np.testing.assert_allclose(later[k][:cut], got[k][:cut], atol=0., rtol=0., equal_nan=True)


def test_real_roll_rets_entrypoint_preserves_reopening_mark(tmp_path):
    data = _market_fixture()
    prices = 100. * np.cumprod(1. + data['returns'], axis=0)
    opened = np.nan_to_num(data['is_tradable_out0']) != 0
    data.update(vwap_mp_out0=np.where(opened, prices, np.nan),
                vwap_mp_out1=np.where(opened, prices * 1.01, np.nan),
                is_tradable_out1=data['is_tradable_out0'].copy(),
                wdte_out0=np.full_like(prices, 10.),
                volume_out0=np.where(opened, 100., 0.))
    formula = example._formula()  # real RollRets/POV entrypoint, not a return override
    got = _native_values({k:formula[k] for k in ('returns','gap_event','ordinary_returns')},data,tmp_path)
    np.testing.assert_allclose(got['returns'][1:], data['returns'][1:], atol=2e-14)
    np.testing.assert_allclose(got['returns'][200], [-.08,.02,.12], atol=2e-14)
    assert got['gap_event'][200].all()
    assert np.isnan(got['ordinary_returns'][200]).all()
