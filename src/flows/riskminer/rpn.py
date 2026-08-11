from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import random
from collections.abc import Iterable, Sequence

from trading_dsl_engine.base.dsl import ensure_expr, var
from trading_dsl_engine.base.parser import Call, Expr, Identifier, KeyTuple, Number, String, Universe

from .config import RiskMinerConfig
from .operators import OperatorSchema, default_operator_catalog
from .semantics import (
    DEFAULT_TYPE_GRAPH,
    SearchShape,
    SemanticInfo,
    TypeGraph,
    alpha_terminal_metadata,
    literal_semantics,
    target_type_satisfied,
)


COMMUTATIVE_CALLS = frozenset(
    {"add", "mul", "minimum", "maximum", "eq", "ne", "and", "and_", "or", "or_", "xor"}
)
ASSOCIATIVE_CALLS = frozenset({"add", "mul"})


def canonical_expr_key(expr: Expr) -> tuple:
    if isinstance(expr, Identifier):
        return ("id", expr.name)
    if isinstance(expr, Number):
        value = float(expr.value)
        if math.isnan(value):
            return ("number", "nan")
        if value == 0.0:
            return ("number", "-0" if math.copysign(1.0, value) < 0 else "0")
        return ("number", value)
    if isinstance(expr, String):
        return ("string", expr.value)
    if isinstance(expr, Universe):
        return ("universe", expr.groups)
    if isinstance(expr, KeyTuple):
        return ("tuple", tuple(canonical_expr_key(item) for item in expr.items))
    if isinstance(expr, Call):
        children = [canonical_expr_key(arg) for arg in expr.args]
        if expr.fn in ASSOCIATIVE_CALLS:
            flattened: list[tuple] = []
            for child in children:
                if len(child) >= 3 and child[0] == "call" and child[1] == expr.fn:
                    flattened.extend(child[2])
                else:
                    flattened.append(child)
            children = flattened
        if expr.fn in COMMUTATIVE_CALLS:
            children.sort(key=repr)
        kwargs = tuple(
            sorted(
                ((name, canonical_expr_key(value)) for name, value in expr.kwargs),
                key=lambda item: item[0],
            )
        )
        return ("call", expr.fn, tuple(children), kwargs)
    return (type(expr).__name__, repr(expr))


@dataclass(frozen=True)
class StackValue:
    expr: Expr
    semantics: SemanticInfo
    depth: int
    canonical_key: tuple
    literal_value: float | None = None

    @classmethod
    def make(
        cls,
        expr: Expr,
        semantics: SemanticInfo,
        *,
        depth: int = 1,
        literal_value: float | None = None,
    ) -> "StackValue":
        return cls(
            ensure_expr(expr),
            semantics,
            depth,
            canonical_expr_key(ensure_expr(expr)),
            literal_value,
        )


class TokenKind(str, Enum):
    TERMINAL = "terminal"
    LITERAL = "literal"
    OPERATOR = "operator"
    END = "end"


@dataclass(frozen=True)
class Token:
    token_id: int
    name: str
    kind: TokenKind
    prior: float
    value: StackValue | None = None
    operator: OperatorSchema | None = None


@dataclass(frozen=True)
class RPNState:
    token_ids: tuple[int, ...] = ()
    stack: tuple[StackValue, ...] = ()
    terminated: bool = False

    @property
    def token_count(self) -> int:
        return len(self.token_ids)


class Vocabulary:
    def __init__(self, tokens: Sequence[Token]) -> None:
        self.tokens = tuple(tokens)
        self.by_id = {token.token_id: token for token in self.tokens}
        self.by_name = {token.name: token for token in self.tokens}
        if len(self.by_id) != len(self.tokens):
            raise ValueError("duplicate token ID")
        if len(self.by_name) != len(self.tokens):
            raise ValueError("duplicate token name")

    def __iter__(self):
        return iter(self.tokens)

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def end(self) -> Token:
        return self.by_name["END"]


def build_vocabulary(
    *,
    terminals: dict[str, SemanticInfo] | None = None,
    literals: Iterable[float] = (
        -30.0,
        -10.0,
        -5.0,
        -2.0,
        -1.0,
        -0.5,
        -0.01,
        0.0,
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
        10.0,
        20.0,
        30.0,
        40.0,
        50.0,
        60.0,
        120.0,
        240.0,
        1440.0,
    ),
    operators: Sequence[OperatorSchema] | None = None,
) -> Vocabulary:
    tokens: list[Token] = []
    next_id = 0
    for name, semantics in sorted(
        (alpha_terminal_metadata() if terminals is None else terminals).items()
    ):
        tokens.append(
            Token(
                next_id,
                name,
                TokenKind.TERMINAL,
                1.5 if name == "soft_side_wavg" else 1.0,
                StackValue.make(var(name), semantics),
            )
        )
        next_id += 1
    for value in literals:
        numeric = float(value)
        tokens.append(
            Token(
                next_id,
                f"CONST[{numeric:g}]",
                TokenKind.LITERAL,
                0.45,
                StackValue.make(
                    ensure_expr(numeric),
                    literal_semantics(numeric),
                    literal_value=numeric,
                ),
            )
        )
        next_id += 1
    for operator in tuple(default_operator_catalog() if operators is None else operators):
        tokens.append(
            Token(
                next_id,
                operator.name,
                TokenKind.OPERATOR,
                operator.prior,
                operator=operator,
            )
        )
        next_id += 1
    tokens.append(Token(next_id, "END", TokenKind.END, 1.0))
    return Vocabulary(tokens)


class TypedRPNEnvironment:
    def __init__(
        self,
        *,
        config: RiskMinerConfig = RiskMinerConfig(),
        vocabulary: Vocabulary | None = None,
        target_types: Iterable[str] = ("dimensionless",),
        type_graph: TypeGraph = DEFAULT_TYPE_GRAPH,
    ) -> None:
        self.config = config
        self.vocabulary = vocabulary or build_vocabulary()
        self.target_types = frozenset(target_types)
        self.type_graph = type_graph

    def initial_state(self) -> RPNState:
        return RPNState()

    def formula_value(self, state: RPNState) -> StackValue | None:
        if len(state.stack) != 1:
            return None
        value = state.stack[0]
        if value.depth < self.config.min_formula_depth:
            return None
        if value.semantics.shape not in {
            SearchShape.ROW,
            SearchShape.BOOLEAN_ROW,
        }:
            return None
        if not target_type_satisfied(
            value.semantics,
            self.target_types,
            self.type_graph,
        ):
            return None
        return value

    def can_terminate(self, state: RPNState) -> bool:
        return not state.terminated and self.formula_value(state) is not None

    def legal_actions(self, state: RPNState) -> tuple[int, ...]:
        if state.terminated:
            return ()
        legal: list[int] = []
        for token in self.vocabulary:
            if self._can_apply(state, token):
                legal.append(token.token_id)
        return tuple(legal)

    def _can_apply(self, state: RPNState, token: Token) -> bool:
        if state.token_count >= self.config.max_tokens:
            return False
        if token.kind is TokenKind.END:
            return self.can_terminate(state)
        if token.kind in {TokenKind.TERMINAL, TokenKind.LITERAL}:
            if len(state.stack) >= self.config.max_stack:
                return False
            new_stack_size = len(state.stack) + 1
            remaining = self.config.max_tokens - (state.token_count + 1)
            return remaining >= new_stack_size
        operator = token.operator
        assert operator is not None
        if len(state.stack) < operator.arity:
            return False
        args = state.stack[-operator.arity :]
        if not operator.validate([arg.semantics for arg in args]):
            return False
        depth = 1 + max(arg.depth for arg in args)
        if depth > self.config.max_depth:
            return False
        new_stack_size = len(state.stack) - operator.arity + 1
        remaining = self.config.max_tokens - (state.token_count + 1)
        return remaining >= new_stack_size

    def apply(self, state: RPNState, token_id: int) -> RPNState:
        if state.terminated:
            raise ValueError("cannot extend a terminated RPN state")
        token = self.vocabulary.by_id[token_id]
        if not self._can_apply(state, token):
            raise ValueError(f"illegal RPN token {token.name!r}")
        token_ids = state.token_ids + (token_id,)
        if token.kind is TokenKind.END:
            return RPNState(token_ids, state.stack, True)
        if token.kind in {TokenKind.TERMINAL, TokenKind.LITERAL}:
            assert token.value is not None
            return RPNState(token_ids, state.stack + (token.value,), False)

        operator = token.operator
        assert operator is not None
        args = state.stack[-operator.arity :]
        exprs = [arg.expr for arg in args]
        literals = [arg.literal_value for arg in args]
        expr = operator.build(exprs, literals)
        semantics = operator.infer([arg.semantics for arg in args])
        depth = 1 + max(arg.depth for arg in args)
        result = StackValue.make(expr, semantics, depth=depth)
        return RPNState(
            token_ids,
            state.stack[: -operator.arity] + (result,),
            False,
        )

    def state_key(self, state: RPNState) -> tuple:
        return (
            state.terminated,
            state.token_count,
            tuple(
                (
                    value.canonical_key,
                    tuple(sorted(value.semantics.types)),
                    value.semantics.shape.value,
                    value.depth,
                    value.literal_value,
                )
                for value in state.stack
            ),
        )

    def render_tokens(self, state: RPNState) -> str:
        return " ".join(
            self.vocabulary.by_id[token_id].name for token_id in state.token_ids
        )

    def sample_action(
        self,
        state: RPNState,
        rng: random.Random,
        *,
        prefer_end_probability: float = 0.25,
    ) -> int:
        legal = list(self.legal_actions(state))
        if not legal:
            raise RuntimeError("RPN state has no legal actions")
        end_id = self.vocabulary.end.token_id
        if end_id in legal and rng.random() < prefer_end_probability:
            return end_id
        non_end = [token_id for token_id in legal if token_id != end_id]
        choices = non_end or legal
        weights = [max(0.0, self.vocabulary.by_id[item].prior) for item in choices]
        return rng.choices(choices, weights=weights, k=1)[0]


__all__ = [
    "RPNState",
    "StackValue",
    "Token",
    "TokenKind",
    "TypedRPNEnvironment",
    "Vocabulary",
    "build_vocabulary",
    "canonical_expr_key",
]
