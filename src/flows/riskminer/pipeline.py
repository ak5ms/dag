from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .config import RiskMinerConfig
from .learned_policy import JaxGRUPolicy
from .pool import CppStreamPoolEvaluator, PoolEvaluation, RidgeAlphaPool
from .reward import CppStreamOrthogonalEvaluator, RewardDensePoolModel
from .rpn import TypedRPNEnvironment, build_vocabulary
from .semantics import SemanticInfo, alpha_terminal_metadata
from .trainer import RiskSeekingTrainer, RiskSeekingTrainingResult


@dataclass(frozen=True)
class CompleteRiskMinerResult:
    training: RiskSeekingTrainingResult
    pool: RidgeAlphaPool
    test_evaluation: PoolEvaluation | None
    intermediate_evaluator: CppStreamOrthogonalEvaluator


def train_cpp_stream_riskminer(
    train_sources: Mapping[str, Any],
    validation_sources: Mapping[str, Any],
    *,
    n_instruments: int,
    work_dir: str | Path,
    config: RiskMinerConfig = RiskMinerConfig(),
    iterations: int = 1,
    terminals: Mapping[str, SemanticInfo] | None = None,
    test_sources: Mapping[str, Any] | None = None,
    policy: JaxGRUPolicy | None = None,
    pool_formula_kwargs: Mapping[str, Any] | None = None,
    compile_kwargs: Mapping[str, Any] | None = None,
    run_kwargs: Mapping[str, Any] | None = None,
    on_event: Callable[[str, Mapping[str, object]], None] | None = None,
) -> CompleteRiskMinerResult:
    """Run the complete neural-prior/reward-dense RiskMiner pipeline.

    Intermediate orthogonalized Sharpe uses the training sources. Terminal pool
    score/admission uses validation sources. The optional test sources are used
    only once, after mining, to evaluate the final frozen pool.
    """

    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    terminal_values = dict(
        alpha_terminal_metadata() if terminals is None else terminals
    )
    vocabulary = build_vocabulary(terminals=terminal_values)
    intermediate = CppStreamOrthogonalEvaluator(
        train_sources,
        n_instruments=n_instruments,
        work_dir=root / "intermediate",
        batch_size=config.evaluation_batch_size,
        compile_kwargs=compile_kwargs,
        run_kwargs=run_kwargs,
        on_event=on_event,
    )
    validation_evaluator = CppStreamPoolEvaluator(
        validation_sources,
        n_instruments=n_instruments,
        work_dir=root / "validation_pool",
        compile_kwargs=compile_kwargs,
        run_kwargs=run_kwargs,
        on_event=on_event,
    )
    pool = RidgeAlphaPool(
        validation_evaluator,
        capacity=config.pool_capacity,
        min_improvement=config.pool_min_improvement,
        formula_kwargs=pool_formula_kwargs,
    )
    reward_model = RewardDensePoolModel(
        intermediate, pool, on_event=on_event
    )
    trainer = RiskSeekingTrainer(
        vocabulary_size=len(vocabulary),
        config=config,
        policy=policy,
        initial_token_priors=tuple(
            token.prior for token in vocabulary
        ),
        output_dir=root / "policy",
        on_event=(
            (lambda event, payload: on_event(event, payload))
            if on_event is not None else None
        ),
    )

    def environment_factory(index: int) -> TypedRPNEnvironment:
        return TypedRPNEnvironment(
            config=replace(config, seed=config.seed + index),
            vocabulary=vocabulary,
            target_types=("dimensionless",),
        )

    training = trainer.run(
        environment_factory, reward_model, iterations=int(iterations)
    )
    test_evaluation = None
    if test_sources is not None and pool.entries:
        test_evaluator = CppStreamPoolEvaluator(
            test_sources,
            n_instruments=n_instruments,
            work_dir=root / "test_pool",
            compile_kwargs=compile_kwargs,
            run_kwargs=run_kwargs,
            on_event=on_event,
        )
        test_evaluation = test_evaluator.evaluate(
            pool.expressions, **dict(pool_formula_kwargs or {})
        )
    return CompleteRiskMinerResult(
        training=training,
        pool=pool,
        test_evaluation=test_evaluation,
        intermediate_evaluator=intermediate,
    )


__all__ = ["CompleteRiskMinerResult", "train_cpp_stream_riskminer"]
