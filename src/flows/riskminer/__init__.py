"""RiskMiner-style typed RPN search for the dag trading DSL."""

from .config import RiskMinerConfig
from .cpp_stream_eval import (
    BatchExecution,
    CppStreamCandidateEvaluator,
    EvaluationSummary,
    build_candidate_score_formula,
)
from .mcts import (
    ArchiveEntry,
    FormulaArchive,
    RiskMCTS,
    RiskMinerSearchResult,
    SchemaPriorPolicy,
    SearchMetrics,
)
from .operators import OperatorSchema, default_operator_catalog
from .pool import (
    CppStreamPoolEvaluator,
    PoolEvaluation,
    build_ridge_pool_score_formula,
    halflife_to_span,
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
    SearchShape,
    SemanticInfo,
    TypeGraph,
    alpha_terminal_metadata,
)

__all__ = [
    "ArchiveEntry",
    "BatchExecution",
    "CppStreamCandidateEvaluator",
    "CppStreamPoolEvaluator",
    "CppStreamRiskMinerResult",
    "DEFAULT_TYPE_GRAPH",
    "EvaluationSummary",
    "FormulaArchive",
    "OperatorSchema",
    "PoolEvaluation",
    "RPNState",
    "RiskMCTS",
    "RiskMinerConfig",
    "RiskMinerSearchResult",
    "SchemaPriorPolicy",
    "SearchMetrics",
    "SearchShape",
    "SemanticInfo",
    "StackValue",
    "Token",
    "TokenKind",
    "TypeGraph",
    "TypedRPNEnvironment",
    "Vocabulary",
    "alpha_terminal_metadata",
    "build_candidate_score_formula",
    "build_ridge_pool_score_formula",
    "build_vocabulary",
    "canonical_expr_key",
    "default_operator_catalog",
    "halflife_to_span",
    "search_cpp_stream_alphas",
]
