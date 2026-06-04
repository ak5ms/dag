from trading_dsl_engine.jax_flat.custom import StatelessJaxCall, StatelessJaxFunction, stateless
from trading_dsl_engine.jax_flat.engine import JaxFlatRuntime, compile_formula


def compile_formula_cpp(*args, **kwargs):
    from trading_dsl_engine.jax_flat.engine_cpp import compile_formula_cpp as _compile_formula_cpp

    return _compile_formula_cpp(*args, **kwargs)


def __getattr__(name: str):
    if name == "CppFlatRuntime":
        from trading_dsl_engine.jax_flat.engine_cpp import CppFlatRuntime

        return CppFlatRuntime
    raise AttributeError(name)


__all__ = [
    "JaxFlatRuntime",
    "CppFlatRuntime",
    "compile_formula",
    "compile_formula_cpp",
    "StatelessJaxCall",
    "StatelessJaxFunction",
    "stateless",
]
