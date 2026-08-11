from pathlib import Path

p = Path('src/flows/riskminer/pool.py')
s = p.read_text()
old = '''        committed = math.isfinite(resulting_score) and (
            not self.entries or delta > self.min_improvement
        )
'''
new = '''        # Treat an empty pool as score 0 for admission. The first alpha must
        # improve the objective too; merely being finite is not sufficient.
        committed = (
            math.isfinite(resulting_score)
            and delta > self.min_improvement
        )
'''
if old not in s:
    if 'The first alpha must' not in s:
        raise SystemExit('pool admission block not found')
else:
    s = s.replace(old, new, 1)
p.write_text(s)

p = Path('tests/flows/riskminer/test_paper_pipeline.py')
s = p.read_text()
if 'test_first_negative_pool_alpha_is_rejected' not in s:
    s += '''\n\ndef test_first_negative_pool_alpha_is_rejected():\n    class NegativeEvaluator(FakePoolEvaluator):\n        def evaluate(self, alphas, *, include_importance=False, **kwargs):\n            result = super().evaluate(\n                alphas, include_importance=include_importance, **kwargs\n            )\n            return PoolEvaluation(\n                score=-0.25,\n                alpha_count=result.alpha_count,\n                compile_seconds=0.0,\n                run_seconds=0.0,\n                native_seconds=0.0,\n                runtime_type="fake",\n                output_path="",\n                output_shape=(1,),\n                coefficient_importance=result.coefficient_importance,\n            )\n\n    pool = RidgeAlphaPool(NegativeEvaluator(), capacity=100, min_improvement=0.0)\n    transition = pool.consider(_pool_alpha("bad_first"))\n    assert transition.additive_delta == -0.25\n    assert not transition.committed\n    assert pool.entries == []\n    assert pool.score == float("-inf")\n'''
p.write_text(s)
