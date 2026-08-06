from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Iterable, Mapping, Sequence

from trading_dsl_engine.base.dsl import ensure_expr, var
from trading_dsl_engine.base.parser import Expr

from flows.riskminer.canonical import canonical_string
from flows.riskminer.operators import OperatorSchema, default_operator_schemas
from flows.riskminer.semantics import (
    DEFAULT_TYPE_RELATIONS,
    SemanticInfo,
    TypeRelations,
    default_market_semantics,
    literal_semantics,
    target_satisfied,
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
    terminal_name: str | None = None
    literal_value: float | None = None
    operator: OperatorSchema | None = None


@dataclass(frozen=True)
class StackValue:
    expr: Expr
    semantics: SemanticInfo
    depth: int
    node_count: int
    canonical_key: str


@dataclass(frozen=True)
class RPNState:
    token_ids: tuple[int, ...] = ()
    stack: tuple[StackValue, ...] = ()
    terminated: bool = False
    invalid_reason: str | None = None

    @property
    def complete(self) -> bool:
        return self.terminated and self.invalid_reason is None and len(self.stack) == 1


def _literal_state_key(value: float | None) -> tuple[str, float | None]:
    if value is not None and math.isnan(value):
        return ("nan", None)
    return ("value", value)


class RPNEnvironment:
    def __init__(
        self,
        *,
        terminals: Mapping[str, SemanticInfo] | None = None,
        operators: Sequence[OperatorSchema] | None = None,
        literals: Iterable[float] = (
            float("nan"), -1.0, 0.0, 1.0, 2.0, 5.0, 10.0,
            20.0, 60.0, 120.0, 240.0, 1440.0,
        ),
        target_types: Iterable[str] = ("dimensionless",),
        relations: TypeRelations = DEFAULT_TYPE_RELATIONS,
        max_depth: int = 8,
        max_tokens: int = 32,
        max_stack: int = 12,
    ) -> None:
        if max_depth < 0 or max_tokens < 1 or max_stack < 1:
            raise ValueError("max_depth, max_tokens, and max_stack must be positive")
        self.terminals = dict(default_market_semantics() if terminals is None else terminals)
        self.operators = tuple(default_operator_schemas() if operators is None else operators)
        self.target_types = frozenset(target_types)
        self.relations = relations
        self.max_depth = int(max_depth)
        self.max_tokens = int(max_tokens)
        self.max_stack = int(max_stack)
        self.tokens = self._build_tokens(tuple(float(value) for value in literals))
        self.token_by_id = {token.token_id: token for token in self.tokens}
        self.token_by_name = {token.name: token for token in self.tokens}
        self.end_token = self.token_by_name["END"]

    def _build_tokens(self, literals: tuple[float, ...]) -> tuple[Token, ...]:
        tokens: list[Token] = []
        for name in sorted(self.terminals):
            prior = 2.5 if name == "soft_side_wavg" else 1.0
            tokens.append(Token(len(tokens), name, TokenKind.TERMINAL, prior, terminal_name=name))
        for value in literals:
            label = "nan" if math.isnan(value) else f"{value:g}"
            tokens.append(Token(len(tokens), f"literal:{label}", TokenKind.LITERAL, 0.45, literal_value=value))
        for operator in self.operators:
            tokens.append(Token(len(tokens), operator.name, TokenKind.OPERATOR, operator.prior, operator=operator))
        tokens.append(Token(len(tokens), "END", TokenKind.END, 2.5))
        return tuple(tokens)

    def initial_state(self) -> RPNState:
        return RPNState()

    def candidate(self, state: RPNState) -> StackValue | None:
        if len(state.stack) != 1:
            return None
        value = state.stack[0]
        if value.semantics.shape not in {"row", "boolean"}:
            return None
        if not target_satisfied(value.semantics, self.target_types, self.relations):
            return None
        return value

    def state_key(self, state: RPNState) -> tuple:
        stack = tuple(
            (
                value.canonical_key,
                value.depth,
                tuple(sorted(self.relations.closure(value.semantics.types))),
                value.semantics.shape,
                _literal_state_key(value.semantics.literal_value),
            )
            for value in state.stack
        )
        return (stack, len(state.token_ids), state.terminated, state.invalid_reason)

    def legal_tokens(self, state: RPNState) -> tuple[Token, ...]:
        if state.terminated or state.invalid_reason is not None:
            return ()
        if len(state.token_ids) >= self.max_tokens:
            return (self.end_token,) if self.candidate(state) is not None else ()
        remaining_after = self.max_tokens - len(state.token_ids) - 1
        legal: list[Token] = []
        for token in self.tokens:
            if token.kind is TokenKind.END:
                if self.candidate(state) is not None:
                    legal.append(token)
                continue
            if token.kind in {TokenKind.TERMINAL, TokenKind.LITERAL}:
                new_stack_size = len(state.stack) + 1
                if new_stack_size <= self.max_stack and new_stack_size - 1 <= remaining_after:
                    legal.append(token)
                continue
            operator = token.operator
            assert operator is not None
            if len(state.stack) < operator.arity:
                continue
            operands = state.stack[-operator.arity :]
            if 1 + max(value.depth for value in operands) > self.max_depth:
                continue
            result = operator.apply(
                tuple(value.expr for value in operands),
                tuple(value.semantics for value in operands),
                self.relations,
            )
            if result is None:
                continue
            new_stack_size = len(state.stack) - operator.arity + 1
            if new_stack_size - 1 <= remaining_after:
                legal.append(token)
        return tuple(legal)

    def step(self, state: RPNState, token_id: int) -> RPNState:
        token = self.token_by_id[token_id]
        legal_ids = {candidate.token_id for candidate in self.legal_tokens(state)}
        if token_id not in legal_ids:
            return RPNState(
                state.token_ids + (token_id,),
                state.stack,
                terminated=True,
                invalid_reason=f"illegal token {token.name!r}",
            )
        next_tokens = state.token_ids + (token_id,)
        if token.kind is TokenKind.END:
            return RPNState(next_tokens, state.stack, terminated=True)
        if token.kind is TokenKind.TERMINAL:
            assert token.terminal_name is not None
            expression = var(token.terminal_name)
            value = StackValue(
                expression,
                self.terminals[token.terminal_name],
                0,
                1,
                canonical_string(expression),
            )
            return RPNState(next_tokens, state.stack + (value,))
        if token.kind is TokenKind.LITERAL:
            assert token.literal_value is not None
            expression = ensure_expr(token.literal_value)
            value = StackValue(
                expression,
                literal_semantics(token.literal_value),
                0,
                1,
                canonical_string(expression),
            )
            return RPNState(next_tokens, state.stack + (value,))
        operator = token.operator
        assert operator is not None
        operands = state.stack[-operator.arity :]
        result = operator.apply(
            tuple(value.expr for value in operands),
            tuple(value.semantics for value in operands),
            self.relations,
        )
        if result is None:
            return RPNState(
                next_tokens,
                state.stack,
                terminated=True,
                invalid_reason=f"semantic rejection in {operator.name}",
            )
        expression, semantics = result
        value = StackValue(
            expression,
            semantics,
            1 + max(item.depth for item in operands),
            1 + sum(item.node_count for item in operands),
            canonical_string(expression),
        )
        return RPNState(next_tokens, state.stack[: -operator.arity] + (value,))

    def parse(self, token_names: Sequence[str]) -> RPNState:
        state = self.initial_state()
        for name in token_names:
            state = self.step(state, self.token_by_name[name].token_id)
        return state

    def format_tokens(self, token_ids: Sequence[int]) -> str:
        return " ".join(self.token_by_id[token_id].name for token_id in token_ids)


__all__ = ["RPNEnvironment", "RPNState", "StackValue", "Token", "TokenKind"]
