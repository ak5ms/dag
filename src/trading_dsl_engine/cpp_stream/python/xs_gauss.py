from __future__ import annotations

from trading_dsl_engine.base.dsl import register_dsl_function, xs_generalized_rank
from trading_dsl_engine.base.parser import Expr


# cpp_stream transports xs_gauss through XsGeneralizedRankOp using the otherwise
# equivalent signed-zero spelling of power=0.  The generated C++ specializes on
# the -0.0 bit pattern, so ordinary xs_generalized_rank(..., 0) is unchanged.
_XS_GAUSS_POWER_TAG = -0.0


@register_dsl_function("xs_gauss")
def xs_gauss(x: Expr) -> Expr:
    """Magnitude-spaced Gaussian cross-sectional scores scaled to unit std."""

    return xs_generalized_rank(x, _XS_GAUSS_POWER_TAG)


__all__ = ["xs_gauss"]
