from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from flows.riskminer import (
    PoolAlpha,
    PoolEvaluation,
    PolicyTrajectory,
    ReplayBuffer,
    RewardDenseRiskMCTS,
    RidgeAlphaPool,
    RiskMinerConfig,
    RiskSeekingTrainer,
    SearchShape,
    SemanticInfo,
    TypedRPNEnvironment,
    build_vocabulary,
    canonical_expr_key,
    default_operator_catalog,
    split_sources_contiguous,
)
from trading_dsl_engine.base.dsl import var


def _dimensionless_terminal():
    return SemanticInfo(frozenset({"numeric", "dimensionless"}), SearchShape.ROW)


def _tiny_environment(seed=1):
    config = RiskMinerConfig(
        max_depth=1,
        min_formula_depth=1,
        max_tokens=2,
        max_stack=1,
        simulations=1,
        rollouts_per_expansion=1,
        evaluation_batch_size=1,
        archive_size=8,
        replay_capacity=8,
        policy_batch_size=1,
        policy_train_epochs=1,
        seed=seed,
    )
    vocabulary = build_vocabulary(
        terminals={"x": _dimensionless_terminal()}, literals=(), operators=()
    )
    return TypedRPNEnvironment(config=config, vocabulary=vocabulary), config


class FakeRewardModel:
    def __init__(self):
        self.pool = SimpleNamespace(entries=[], score=-1.0)

    def intermediate_rewards(self, values):
        return {value.canonical_key: 1.0 for value in values}

    def terminal_reward(self, value, *, rpn, individual_score=float("nan")):
        del value, rpn, individual_score
        return SimpleNamespace(
            reward=3.0,
            transition=SimpleNamespace(committed=True),
        )


def test_reward_dense_mcts_uses_intermediate_and_terminal_rewards_per_edge():
    environment, config = _tiny_environment()
    mcts = RewardDenseRiskMCTS(environment, FakeRewardModel(), config=config)
    result = mcts.search()
    assert len(result.trajectories) == 1
    trajectory = result.trajectories[0]
    assert trajectory.step_rewards == (1.0, 3.0)
    assert trajectory.reward == 4.0
    root = environment.initial_state()
    edge = mcts.nodes[environment.state_key(root)].edges[
        environment.vocabulary.by_name["x"].token_id
    ]
    assert edge.reward == 1.0
    assert edge.q == 4.0


class FakePoolEvaluator:
    def evaluate(self, alphas, *, include_importance=False, **kwargs):
        del kwargs
        importance = ()
        if include_importance:
            importance = tuple([0.50, 0.10, 0.30][:len(alphas)])
        return PoolEvaluation(
            score=float(len(alphas)),
            alpha_count=len(alphas),
            compile_seconds=0.0,
            run_seconds=0.0,
            native_seconds=0.0,
            runtime_type="fake",
            output_path="",
            output_shape=(1,),
            coefficient_importance=importance,
        )


def _pool_alpha(name: str) -> PoolAlpha:
    expr = var(name)
    return PoolAlpha(expr, canonical_expr_key(expr), name, 1, 0.0)


def test_pool_capacity_evicts_smallest_absolute_ridge_weight():
    pool = RidgeAlphaPool(FakePoolEvaluator(), capacity=2, min_improvement=-1.0)
    pool.consider(_pool_alpha("a"))
    pool.consider(_pool_alpha("b"))
    transition = pool.consider(_pool_alpha("c"))
    assert transition.committed
    assert transition.evicted is not None
    assert transition.evicted.rpn == "b"
    assert [entry.rpn for entry in pool.entries] == ["a", "c"]


def test_replay_buffer_is_bounded_and_batches_trajectories():
    buffer = ReplayBuffer(2)
    for reward in (1.0, 2.0, 3.0):
        buffer.add(
            PolicyTrajectory(
                states=((),), actions=(0,), legal_actions=((0,),), reward=reward
            )
        )
    assert [item.reward for item in buffer.trajectories] == [2.0, 3.0]
    assert len(list(buffer.batches(1, epochs=2, seed=3))) == 4


def test_trainer_alternates_search_quantile_and_policy_update(tmp_path):
    environment, config = _tiny_environment(seed=4)
    trainer = RiskSeekingTrainer(
        vocabulary_size=len(environment.vocabulary),
        config=config,
        output_dir=tmp_path,
    )
    report = trainer.run_iteration(environment, FakeRewardModel(), iteration=1)
    assert report.search.metrics.trajectories == 1
    assert report.reward_quantile_after > report.reward_quantile_before
    assert report.trajectory_quantiles == (report.reward_quantile_before,)
    assert report.policy_losses
    assert report.policy_checkpoint is not None
    assert Path(report.policy_checkpoint).is_file()




def test_neural_policy_starts_from_schema_priors():
    config = RiskMinerConfig(
        max_depth=2, min_formula_depth=1, max_tokens=4, max_stack=2,
        simulations=1, evaluation_batch_size=1, archive_size=1, seed=9,
    )
    vocabulary = build_vocabulary(
        terminals={"x": _dimensionless_terminal()},
        literals=(1.0,),
        operators=(),
    )
    environment = TypedRPNEnvironment(config=config, vocabulary=vocabulary)
    trainer = RiskSeekingTrainer(
        vocabulary_size=len(vocabulary),
        config=config,
        initial_token_priors=tuple(token.prior for token in vocabulary),
    )
    legal = environment.legal_actions(environment.initial_state())
    priors = trainer.policy.priors(
        environment, environment.initial_state(), legal
    )
    assert priors[vocabulary.by_name["x"].token_id] > priors[
        vocabulary.by_name["CONST[1]"].token_id
    ]


def test_invalid_dead_end_episode_is_retained_for_policy_training():
    config = RiskMinerConfig(
        max_depth=1, min_formula_depth=1, max_tokens=2, max_stack=1,
        simulations=1, rollouts_per_expansion=1, evaluation_batch_size=1,
        archive_size=2, invalid_reward=-7.0, seed=3,
    )
    vocabulary = build_vocabulary(
        terminals={
            "price": SemanticInfo(
                frozenset({"numeric", "price"}), SearchShape.ROW
            )
        },
        literals=(),
        operators=(),
    )
    environment = TypedRPNEnvironment(config=config, vocabulary=vocabulary)
    result = RewardDenseRiskMCTS(
        environment, FakeRewardModel(), config=config
    ).search()
    assert result.metrics.invalid_rollouts == 1
    assert len(result.trajectories) == 1
    trajectory = result.trajectories[0]
    assert trajectory.terminal_formula_key is None
    assert trajectory.reward == -7.0
    assert trajectory.step_rewards == (-7.0,)


def test_paper_operator_inventory_and_dynamic_temporal_expression():
    names = {schema.name for schema in default_operator_catalog()}
    required = {
        "sign", "abs", "log", "xs_rank", "add", "sub", "mul", "div",
        "greater", "less", "shift", "rolling_rank", "rolling_skew",
        "rolling_kurt", "rolling_mean", "rolling_median", "rolling_sum",
        "rolling_std", "rolling_var", "rolling_max", "rolling_min",
        "rolling_wma", "ewm", "rolling_cov", "rolling_corr",
        "dynamic_ewm", "dynamic_shift", "dynamic_rolling_corr",
    }
    assert required <= names

    config = RiskMinerConfig(
        max_depth=3, min_formula_depth=3, max_tokens=6, max_stack=3,
        simulations=1, evaluation_batch_size=1, archive_size=1,
    )
    vocabulary = build_vocabulary(
        terminals={
            "price": SemanticInfo(frozenset({"numeric", "price"}), SearchShape.ROW),
            "selector": _dimensionless_terminal(),
        }
    )
    environment = TypedRPNEnvironment(config=config, vocabulary=vocabulary)
    state = environment.initial_state()
    for token in ("price", "selector", "dynamic_ewm", "xs_rank", "END"):
        state = environment.apply(state, vocabulary.by_name[token].token_id)
    assert state.terminated
    assert "where" in repr(environment.formula_value(state).expr)


def test_contiguous_train_validation_test_split_has_no_overlap():
    import numpy as np

    values = np.arange(100 * 3, dtype=float).reshape(100, 3)
    train, validation, test = split_sources_contiguous(
        {"x": values}, train_fraction=0.6, validation_fraction=0.2
    )
    assert (train.start, train.stop) == (0, 60)
    assert (validation.start, validation.stop) == (60, 80)
    assert (test.start, test.stop) == (80, 100)
    assert train.sources["x"][-1, 0] < validation.sources["x"][0, 0]
    assert validation.sources["x"][-1, 0] < test.sources["x"][0, 0]


def test_rejected_capacity_trial_does_not_report_an_actual_eviction():
    class RejectingEvaluator(FakePoolEvaluator):
        def evaluate(self, alphas, *, include_importance=False, **kwargs):
            result = super().evaluate(
                alphas, include_importance=include_importance, **kwargs
            )
            score = float(len(alphas)) if len(alphas) <= 2 else -10.0
            return PoolEvaluation(
                score=score,
                alpha_count=result.alpha_count,
                compile_seconds=0.0,
                run_seconds=0.0,
                native_seconds=0.0,
                runtime_type="fake",
                output_path="",
                output_shape=(1,),
                coefficient_importance=result.coefficient_importance,
            )

    pool = RidgeAlphaPool(RejectingEvaluator(), capacity=2, min_improvement=0.0)
    pool.consider(_pool_alpha("a"))
    pool.consider(_pool_alpha("b"))
    transition = pool.consider(_pool_alpha("c"))
    assert not transition.committed
    assert transition.evicted is None
    assert [entry.rpn for entry in pool.entries] == ["a", "b"]


def test_paper_constant_inventory_and_covariance_units_require_normalization():
    config = RiskMinerConfig(
        max_depth=4,
        min_formula_depth=1,
        max_tokens=8,
        max_stack=3,
        simulations=1,
        evaluation_batch_size=1,
        archive_size=1,
    )
    terminals = {
        "p": SemanticInfo(frozenset({"numeric", "price"}), SearchShape.ROW)
    }
    vocabulary = build_vocabulary(terminals=terminals)
    for literal in (-30.0, -10.0, -5.0, -2.0, -1.0, -0.5, -0.01, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0):
        assert f"CONST[{literal:g}]" in vocabulary.by_name

    environment = TypedRPNEnvironment(config=config, vocabulary=vocabulary)
    state = environment.initial_state()
    for token in ("p", "p", "CONST[5]", "rolling_cov"):
        state = environment.apply(state, vocabulary.by_name[token].token_id)
    assert environment.formula_value(state) is None
    state = environment.apply(state, vocabulary.by_name["xs_rank"].token_id)
    assert environment.formula_value(state) is not None


def test_first_negative_pool_alpha_is_rejected():
    class NegativeEvaluator(FakePoolEvaluator):
        def evaluate(self, alphas, *, include_importance=False, **kwargs):
            result = super().evaluate(
                alphas, include_importance=include_importance, **kwargs
            )
            return PoolEvaluation(
                score=-0.25,
                alpha_count=result.alpha_count,
                compile_seconds=0.0,
                run_seconds=0.0,
                native_seconds=0.0,
                runtime_type="fake",
                output_path="",
                output_shape=(1,),
                coefficient_importance=result.coefficient_importance,
            )

    pool = RidgeAlphaPool(NegativeEvaluator(), capacity=100, min_improvement=0.0)
    transition = pool.consider(_pool_alpha("bad_first"))
    assert transition.additive_delta == -0.25
    assert not transition.committed
    assert pool.entries == []
    assert pool.score == float("-inf")


def test_granular_mcts_and_replay_events_are_emitted(tmp_path):
    environment, config = _tiny_environment(seed=17)
    events = []

    def on_event(name, payload):
        events.append((name, payload))

    trainer = RiskSeekingTrainer(
        vocabulary_size=len(environment.vocabulary),
        config=config,
        output_dir=tmp_path,
        on_event=on_event,
    )
    trainer.run_iteration(environment, FakeRewardModel(), iteration=1)
    names = [name for name, _ in events]
    required = {
        "mcts_search_start",
        "mcts_node_choice",
        "mcts_selection_edge",
        "mcts_rollout_done",
        "mcts_candidates_evaluate",
        "mcts_candidates_scored",
        "mcts_terminal_evaluate",
        "mcts_terminal_result",
        "mcts_episode_done",
        "mcts_backprop_edge",
        "mcts_search_done",
        "replay_reset",
        "replay_snapshot",
        "replay_quantile_update",
        "policy_train_batch_start",
        "policy_train_batch_done",
    }
    assert required <= set(names)

    node_choice = next(payload for name, payload in events if name == "mcts_node_choice")
    assert node_choice["edges"]
    assert {"token", "prior", "q", "visits", "puct"} <= set(node_choice["edges"][0])

    candidates = next(payload for name, payload in events if name == "mcts_candidates_evaluate")
    assert candidates["candidate_count"] >= 1
    assert candidates["candidates"][0]["rpn"]

    snapshot = next(payload for name, payload in events if name == "replay_snapshot")
    assert snapshot["size"] == 1
    assert snapshot["trajectories"][0]["actions"]

    backprop = next(payload for name, payload in events if name == "mcts_backprop_edge")
    assert backprop["visits_after"] == backprop["visits_before"] + 1


def test_trace_candidate_records_are_rpn_only():
    config = RiskMinerConfig(
        max_depth=2, min_formula_depth=2, max_tokens=6, max_stack=3,
        simulations=1, rollouts_per_expansion=1, evaluation_batch_size=1,
        archive_size=8, seed=19,
    )
    sem = _dimensionless_terminal()
    vocabulary = build_vocabulary(
        terminals={"x": sem, "y": sem}, literals=(1.0,)
    )
    environment = TypedRPNEnvironment(config=config, vocabulary=vocabulary)
    events = []

    class TraceReward(FakeRewardModel):
        def terminal_reward(self, value, *, rpn, individual_score=float("nan")):
            del value, rpn, individual_score
            return SimpleNamespace(
                reward=0.5,
                transition=SimpleNamespace(
                    committed=False, previous_score=0.0, resulting_score=0.5,
                    additive_delta=0.5, pool_size=0, evicted=None,
                ),
            )

    result = RewardDenseRiskMCTS(
        environment, TraceReward(), config=config,
        on_event=lambda name, payload: events.append((name, payload)),
    ).search()
    assert result.metrics.trajectories == 1
    candidate_events = [payload for name, payload in events if name == "mcts_candidates_evaluate"]
    assert candidate_events
    for record in candidate_events[0]["candidates"]:
        assert "rpn" in record
        assert "expr" not in record
        assert "canonical_key" not in record
    terminal_events = [payload for name, payload in events if name == "mcts_terminal_evaluate"]
    assert terminal_events
    assert "rpn" in terminal_events[0]
    assert "expr" not in terminal_events[0]
