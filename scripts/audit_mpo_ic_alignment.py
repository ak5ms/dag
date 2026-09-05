"""Numeric ablation evidence against the repository's independent test oracles."""
from pathlib import Path
import json
import runpy
import numpy as np


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    tests = runpy.run_path(str(repo / 'tests/examples/test_cpp_stream_mpo_diagnostics.py'))
    ex = tests['example']
    var = tests['var']
    native = tests['_native']
    oracle = tests['_ridge_oracle']
    output = repo / '.generated' / 'mpo_ic_alignment_audit'
    output.mkdir(parents=True, exist_ok=True)
    results = {'base_commit': '834eebe165c2c6832c8328c76c00fa596ce815ae',
               'rows': 360, 'vol_span': 24, 'ridge_halflife': 30,
               'ridge_lambda': .1, 'feature_spans': [4, 16], 'cases': {}}

    def error(actual, expected):
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
        return float(np.nanmax(np.abs(actual-expected)))

    for assets in (3, 9):
        ex.FEATURE_SPANS = (4, 16)
        ex.IC_VOL_SPAN = 24
        ex.RIDGE_HL = 30
        data = tests['_session_data'](assets=assets)
        formula = ex._formula(var('returns'))
        roots = {'fit': formula['forecast_diagnostics'], 'features': formula['features']}
        out = native(roots, data, output / f'fit-{assets}')
        coefficient_error = forecast_error = 0.
        for block in out['fit'].values():
            expected = oracle(block['fit_x'], block['target'], block['sample_weight'], 30)
            coefficient_error = max(coefficient_error, error(block['beta_fitted'], expected))
            predicted = np.einsum('tnf,tf->tn', out['features'], expected)
            forecast_error = max(forecast_error, error(block['yhat_rate'], predicted))
            np.testing.assert_allclose(block['beta_fitted'], expected, rtol=2e-8, atol=2e-10)
            np.testing.assert_allclose(block['yhat_rate'], predicted, rtol=2e-8, atol=2e-12, equal_nan=True)
        scaled_data = {k: v.copy() for k, v in data.items()}
        scaled_data['returns'] *= 2.
        scaled = native(roots, scaled_data, output / f'scaled-{assets}')
        beta_scale_error = yhat_scale_error = 0.
        for key, block in out['fit'].items():
            beta_scale_error = max(beta_scale_error, error(scaled['fit'][key]['beta_fitted'], block['beta_fitted']))
            yhat_scale_error = max(yhat_scale_error, error(scaled['fit'][key]['yhat_rate'], 2*block['yhat_rate']))
            np.testing.assert_allclose(scaled['fit'][key]['beta_fitted'], block['beta_fitted'], rtol=2e-8, atol=2e-10)
            np.testing.assert_allclose(scaled['fit'][key]['yhat_rate'], 2*block['yhat_rate'], rtol=2e-8, atol=2e-12, equal_nan=True)
        ex.FEATURE_SPANS = (4,)
        identity_formula = ex._formula(var('returns'), beta_override=1.)
        identity = native({'alpha': identity_formula['alpha_pnl'], 'yhat': identity_formula['yhat_pnl']},
                          data, output / f'beta-one-{assets}')
        identity_error = 0.
        for key in identity['alpha']:
            for name in ('ic', 'ic1'):
                target = next(iter(identity['alpha'][key][name].values()))
                actual = identity['yhat'][key][name]
                identity_error = max(identity_error, error(actual, target))
                np.testing.assert_allclose(actual, target, rtol=2e-12, atol=2e-14)
                assert np.any(np.abs(target) > 1e-6)
        results['cases'][str(assets)] = {
            'beta_vs_normal_equations_max_abs_error': coefficient_error,
            'yhat_vs_normal_equations_max_abs_error': forecast_error,
            'beta_one_ic_and_ic1_max_abs_error': identity_error,
            'double_return_units_beta_max_abs_error': beta_scale_error,
            'double_return_units_yhat_vs_twice_original_max_abs_error': yhat_scale_error,
            'horizons_checked': len(ex.HORIZONS),
        }
        print(assets, results['cases'][str(assets)], flush=True)
    (repo / 'benchmarks/mpo_ic_alignment.json').write_text(json.dumps(results, indent=2)+'\n')
    print('All numeric audits passed.', flush=True)


if __name__ == "__main__":
    main()
