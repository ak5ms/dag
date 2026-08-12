from __future__ import annotations

import math

import benchmark_riskminer_deep_cpp_stream as benchmark
from flows.riskminer import TokenKind
from trading_dsl_engine.base.parser import Call


_IDEMPOTENT_UNARY = frozenset(
    {"abs", "fraction", "purify", "sign", "xs_rank", "xs_pct_rank"}
)
_REJECT_EQUAL_BINARY = frozenset(
    {"add", "sub", "div", "fillna", "minimum", "maximum"}
)


class QualityDeepTypedRPNEnvironment(benchmark.DeepTypedRPNEnvironment):
    """Add generic structural quality constraints to the typed grammar.

    These rules are expression-shape/algebra rules, not feature-pair allowlists:

    * a literal cannot be the root of an alpha program;
    * literals are parameter values, not arbitrary affine rescalings;
    * an operation cannot consume only compile-time constants;
    * exact equal-operand identities such as ``x-x`` and ``max(x,x)`` are removed;
    * directly repeated idempotent transforms are removed;
    * EWM/rolling parameters that make the operator an identity are removed.
    """

    def _can_apply(self, state, token):
        if not super()._can_apply(state, token):
            return False

        if token.kind is TokenKind.LITERAL:
            return bool(state.stack)
        if token.kind is not TokenKind.OPERATOR or token.operator is None:
            return True

        operator = token.operator
        arguments = state.stack[-operator.arity :]
        if not arguments:
            return False
        if all(argument.semantics.static for argument in arguments):
            return False

        # Literals are legal for bounded temporal/history parameters. With the
        # current vocabulary there are no threshold/comparison operators, so a
        # literal in generic arithmetic is only a redundant scale or offset.
        if operator.family in {"compatible_binary", "numeric_binary"} and any(
            argument.literal_value is not None for argument in arguments
        ):
            return False

        if (
            operator.arity == 2
            and operator.name in _REJECT_EQUAL_BINARY
            and arguments[0].canonical_key == arguments[1].canonical_key
        ):
            return False

        if operator.arity == 1 and operator.name in _IDEMPOTENT_UNARY:
            expression = arguments[0].expr
            if isinstance(expression, Call) and expression.fn == operator.name:
                return False

        if operator.name in {"xs_rank", "xs_pct_rank"}:
            expression = arguments[0].expr
            if isinstance(expression, Call) and expression.fn in {
                "xs_rank", "xs_pct_rank"
            }:
                return False

        if operator.name == "ewm":
            span = arguments[1].literal_value
            if span is None or not math.isfinite(span) or span <= 1.0:
                return False
        if operator.name in {"rolling_mean", "rolling_std"}:
            periods = arguments[1].literal_value
            if periods is None or periods <= 1.0:
                return False

        return True


benchmark.DeepTypedRPNEnvironment = QualityDeepTypedRPNEnvironment


if __name__ == "__main__":
    benchmark.main()
