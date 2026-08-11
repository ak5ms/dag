from pathlib import Path

p = Path('src/flows/riskminer/rpn.py')
s = p.read_text()
old = '''    def legal_actions(self, state: RPNState) -> tuple[int, ...]:
        if state.terminated:
            return ()
        legal: list[int] = []
'''
new = '''    def legal_actions(self, state: RPNState) -> tuple[int, ...]:
        if state.terminated:
            return ()
        # A valid one-value formula already at max_depth cannot be extended
        # without exceeding max_depth. Pushing another value also strands the
        # stack because any later combine with the max-depth value would have
        # depth max_depth + 1. END is therefore the only viable continuation.
        terminal_value = self.formula_value(state)
        if (
            terminal_value is not None
            and terminal_value.depth >= self.config.max_depth
        ):
            return (self.vocabulary.end.token_id,)
        legal: list[int] = []
'''
if old in s:
    s = s.replace(old, new, 1)
elif 'END is therefore the only viable continuation.' not in s:
    raise SystemExit('rpn legal_actions block not found')
p.write_text(s)

p = Path('src/flows/riskminer/mcts.py')
s = p.read_text()
if 'TokenKind' not in s.split('\n', 20)[13]:
    s = s.replace(
        'from .rpn import RPNState, StackValue, TypedRPNEnvironment',
        'from .rpn import RPNState, StackValue, TokenKind, TypedRPNEnvironment',
        1,
    )
old = '''        priors = self.policy.priors(self.environment, state, legal)
        weights = [max(0.0, float(priors.get(token_id, 0.0))) for token_id in choices]
        if not any(weights):
            weights = [1.0] * len(choices)
        return rng.choices(choices, weights=weights, k=1)[0]
'''
new = '''        priors = self.policy.priors(self.environment, state, legal)

        # Correct action-class cardinality bias in rollout completion. With the
        # 69-field InputData grammar, direct token sampling gives terminals most
        # of the aggregate probability even when each terminal is individually
        # no more likely than an operator. Preserve learned token ranking within
        # each structural class, but give a class total weight equal to its mean
        # prior rather than the sum of all member priors.
        groups: dict[str, list[int]] = {}
        for token_id in choices:
            token = self.environment.vocabulary.by_id[token_id]
            if token.kind in {TokenKind.TERMINAL, TokenKind.LITERAL}:
                group = "push"
            else:
                operator = token.operator
                assert operator is not None
                group = "unary" if operator.arity == 1 else "reduce"
            groups.setdefault(group, []).append(token_id)

        weights_by_id: dict[int, float] = {}
        for token_ids in groups.values():
            cardinality = len(token_ids)
            for token_id in token_ids:
                weights_by_id[token_id] = (
                    max(0.0, float(priors.get(token_id, 0.0))) / cardinality
                )
        weights = [weights_by_id[token_id] for token_id in choices]
        if not any(weights):
            weights = [1.0] * len(choices)
        return rng.choices(choices, weights=weights, k=1)[0]
'''
if old in s:
    s = s.replace(old, new, 1)
elif 'Correct action-class cardinality bias' not in s:
    raise SystemExit('mcts rollout block not found')
p.write_text(s)
