from __future__ import annotations

from trading_dsl_engine.base.dsl import register_dsl_function, xs_generalized_rank
from trading_dsl_engine.base.parser import Expr


# cpp_stream transports xs_gauss through XsGeneralizedRankOp using the otherwise
# equivalent signed-zero spelling of power=0.  naryop.hpp specializes that exact
# compile-time bit pattern to XsGaussNode, with no runtime tag check and no change
# to ordinary xs_generalized_rank(..., 0).
_XS_GAUSS_POWER_TAG = -0.0


@register_dsl_function("xs_gauss")
def xs_gauss(x: Expr) -> Expr:
    """Spacing-aware Gaussian shape plus bounded mean/RMS location."""

    return xs_generalized_rank(x, _XS_GAUSS_POWER_TAG)


__all__ = ["xs_gauss"]
