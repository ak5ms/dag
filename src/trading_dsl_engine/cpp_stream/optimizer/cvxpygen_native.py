"""Compatibility imports for the former CVXPYgen-backed module.

New code should import :mod:`clarabel_native`. The generated-program ABI keeps
its historical Python type alias so cached formula code and downstream imports
do not need an immediate rename.
"""

from trading_dsl_engine.cpp_stream.optimizer.clarabel_native import *  # noqa: F403
