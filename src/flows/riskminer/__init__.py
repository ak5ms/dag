"""Typed, pool-aware, risk-seeking formula search for the dag trading DSL."""

from .config import RiskMinerConfig
from .cpp_stream_eval import (
    BatchExecution,
    CppStreamCandidateEvaluator,
    EvaluationSummary,
    build_candidate_score_formula,
)
from .learned_policy import (
    GRUPolicyConfig,
    JaxGRUPolicy,
    PolicyTrajectory,
    RiskQuantileTracker,
    TrajectoryBatch,
)
from .mcts import (
    ArchiveEntry,
    EdgeStats,
    FormulaArchive,
    RewardDenseRiskMCTS,
    RewardDenseSearchResult,
    RiskMCTS,
    RiskMinerSearchResult,
    SchemaPriorPolicy,
    SearchMetrics,
    TreeNode,
)
from .operators import (
    DEFAULT_DYNAMIC_PERIODS,
    OperatorSchema,
    default_operator_catalog,
)
from .pipeline import CompleteRiskMinerResult, train_cpp_stream_riskminer
from .pool import (
    CppStreamPoolEvaluator,
    PoolAlpha,
    PoolEvaluation,
    PoolTransition,
    RidgeAlphaPool,
    build_ridge_pool_beta_formula,
    build_ridge_pool_score_formula,
    halflife_to_span,
)
from .replay import ReplayBuffer
from .reward import (
    CppStreamOrthogonalEvaluator,
    OrthogonalBatchExecution,
    OrthogonalEvaluationSummary,
    RewardDensePoolModel,
    TerminalReward,
    build_cross_sectional_orthogonal_alpha,
    build_orthogonal_score_formula,
)
from .rpn import (
    RPNState,
    StackValue,
    Token,
    TokenKind,
    TypedRPNEnvironment,
    Vocabulary,
    build_vocabulary,
    canonical_expr_key,
)
from .search import CppStreamRiskMinerResult, search_cpp_stream_alphas
from .semantics import (
    DEFAULT_TYPE_GRAPH,
    INPUTDATA_ALPHA_KEYS,
    SearchShape,
    SemanticInfo,
    TypeGraph,
    alpha_terminal_metadata,
    inputdata_alpha_keys,
    inputdata_alpha_terminal_metadata,
)
from .splits import SourceSplit, split_sources_contiguous
from .trainer import (
    MiningIterationReport,
    RiskSeekingTrainer,
    RiskSeekingTrainingResult,
)

__all__ = [
    "ArchiveEntry", "BatchExecution", "CppStreamCandidateEvaluator",
    "CppStreamOrthogonalEvaluator", "CppStreamPoolEvaluator",
    "CppStreamRiskMinerResult", "CompleteRiskMinerResult", "DEFAULT_DYNAMIC_PERIODS",
    "DEFAULT_TYPE_GRAPH", "EdgeStats", "EvaluationSummary",
    "FormulaArchive", "GRUPolicyConfig", "INPUTDATA_ALPHA_KEYS",
    "JaxGRUPolicy", "MiningIterationReport", "OperatorSchema",
    "OrthogonalBatchExecution", "OrthogonalEvaluationSummary", "PoolAlpha",
    "PoolEvaluation", "PoolTransition", "PolicyTrajectory", "RPNState",
    "ReplayBuffer", "RewardDensePoolModel", "RewardDenseRiskMCTS",
    "RewardDenseSearchResult", "RidgeAlphaPool", "RiskMCTS",
    "RiskMinerConfig", "RiskMinerSearchResult", "RiskQuantileTracker",
    "RiskSeekingTrainer", "RiskSeekingTrainingResult", "SchemaPriorPolicy",
    "SearchMetrics", "SearchShape", "SemanticInfo", "SourceSplit",
    "StackValue", "TerminalReward", "Token", "TokenKind", "TrajectoryBatch",
    "TreeNode", "TypeGraph", "TypedRPNEnvironment", "Vocabulary",
    "alpha_terminal_metadata", "build_candidate_score_formula",
    "build_cross_sectional_orthogonal_alpha", "build_orthogonal_score_formula",
    "build_ridge_pool_beta_formula", "build_ridge_pool_score_formula",
    "build_vocabulary", "canonical_expr_key", "default_operator_catalog",
    "halflife_to_span", "inputdata_alpha_keys",
    "inputdata_alpha_terminal_metadata", "search_cpp_stream_alphas",
    "split_sources_contiguous", "train_cpp_stream_riskminer",
]
