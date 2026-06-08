from trading_dsl_engine.base.metadata import MetadataConfig, NodeMetadata, TypeRelationGraph, UnitInfo, ValueRange, field, metadata
from trading_dsl_engine.jax_flat.custom import StatelessJaxCall, StatelessJaxFunction, stateless
from trading_dsl_engine.jax_flat.engine import JaxFlatRuntime, compile_formula


def __getattr__(name: str):
    if name == "CppFlatRuntime":
        from trading_dsl_engine.jax_flat.engine_cpp import CppFlatRuntime

        return CppFlatRuntime
    raise AttributeError(name)


__all__ = [
    "JaxFlatRuntime",
    "CppFlatRuntime",
    "compile_formula",
    "StatelessJaxCall",
    "StatelessJaxFunction",
    "stateless",
    "MetadataConfig",
    "NodeMetadata",
    "TypeRelationGraph",
    "UnitInfo",
    "ValueRange",
    "field",
    "metadata",
]
