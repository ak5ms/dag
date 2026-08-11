from pathlib import Path

p = Path('src/flows/riskminer/mcts.py')
s = p.read_text()
old = '''        observations: dict[tuple, StackValue] = {}
        step_values: list[StackValue | None] = []
        for action, state in zip(actions, resulting):
            if action == end_id:
                step_values.append(None)
                continue
            value = self.environment.formula_value(state)
            step_values.append(value)
            if value is not None:
                observations.setdefault(value.canonical_key, value)
        candidate_records = [
            {
                "canonical_key": repr(key),
                "depth": value.depth,
                "expr": repr(value.expr),
            }
            for key, value in observations.items()
        ]
'''
new = '''        observations: dict[tuple, StackValue] = {}
        observation_rpn: dict[tuple, str] = {}
        step_values: list[StackValue | None] = []
        for action, state in zip(actions, resulting):
            if action == end_id:
                step_values.append(None)
                continue
            value = self.environment.formula_value(state)
            step_values.append(value)
            if value is not None:
                observations.setdefault(value.canonical_key, value)
                observation_rpn.setdefault(
                    value.canonical_key,
                    self._render_token_ids(state.token_ids),
                )
        candidate_records = [
            {
                "rpn": observation_rpn[key],
                "depth": value.depth,
            }
            for key, value in observations.items()
        ]
'''
if old not in s:
    raise SystemExit('candidate record block not found')
s = s.replace(old, new, 1)
old = '''        self._emit(
            "mcts_terminal_evaluate",
            rpn=rpn,
            expr=repr(terminal_value.expr),
            depth=terminal_value.depth,
            individual_score=individual_score,
        )
'''
new = '''        self._emit(
            "mcts_terminal_evaluate",
            rpn=rpn,
            depth=terminal_value.depth,
            individual_score=individual_score,
        )
'''
if old not in s:
    raise SystemExit('terminal event block not found')
s = s.replace(old, new, 1)
p.write_text(s)

p = Path('scripts/run_riskminer_inputdata.py')
s = p.read_text()
old = '''                f"  alpha_{index:03d} depth={entry.depth} orthogonal_score={entry.individual_score:.8g}",
                f"    rpn: {entry.rpn}",
                f"    expr: {entry.expr!r}",
'''
new = '''                f"  alpha_{index:03d} depth={entry.depth} orthogonal_score={entry.individual_score:.8g}",
                f"    rpn: {entry.rpn}",
'''
if old not in s:
    raise SystemExit('pool tree expr block not found')
s = s.replace(old, new, 1)
s = s.replace('                "expr": repr(entry.expr),\n', '', 1)
p.write_text(s)

p = Path('tests/flows/riskminer/test_paper_pipeline.py')
s = p.read_text()
s = s.replace(
    '    assert candidates["candidates"][0]["expr"]\n',
    '    assert candidates["candidates"][0]["rpn"]\n',
)
if 'test_trace_candidate_records_are_rpn_only' not in s:
    s += '''\n\ndef test_trace_candidate_records_are_rpn_only():\n    config = RiskMinerConfig(\n        max_depth=2, min_formula_depth=2, max_tokens=6, max_stack=3,\n        simulations=1, rollouts_per_expansion=1, evaluation_batch_size=1,\n        archive_size=8, seed=19,\n    )\n    sem = _dimensionless_terminal()\n    vocabulary = build_vocabulary(\n        terminals={"x": sem, "y": sem}, literals=(1.0,)\n    )\n    environment = TypedRPNEnvironment(config=config, vocabulary=vocabulary)\n    events = []\n\n    class TraceReward(FakeRewardModel):\n        def terminal_reward(self, value, *, rpn, individual_score=float("nan")):\n            del value, rpn, individual_score\n            return SimpleNamespace(\n                reward=0.5,\n                transition=SimpleNamespace(\n                    committed=False, previous_score=0.0, resulting_score=0.5,\n                    additive_delta=0.5, pool_size=0, evicted=None,\n                ),\n            )\n\n    result = RewardDenseRiskMCTS(\n        environment, TraceReward(), config=config,\n        on_event=lambda name, payload: events.append((name, payload)),\n    ).search()\n    assert result.metrics.trajectories == 1\n    candidate_events = [payload for name, payload in events if name == "mcts_candidates_evaluate"]\n    assert candidate_events\n    for record in candidate_events[0]["candidates"]:\n        assert "rpn" in record\n        assert "expr" not in record\n        assert "canonical_key" not in record\n    terminal_events = [payload for name, payload in events if name == "mcts_terminal_evaluate"]\n    assert terminal_events\n    assert "rpn" in terminal_events[0]\n    assert "expr" not in terminal_events[0]\n'''
p.write_text(s)
print('RPN-only logging patch applied')
