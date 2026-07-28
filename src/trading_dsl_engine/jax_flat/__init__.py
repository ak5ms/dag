from pathlib import Path
import importlib
import importlib.util

from trading_dsl_engine.base.metadata import MetadataConfig, NodeMetadata, TypeRelationGraph, UnitInfo, ValueRange, field, metadata
from trading_dsl_engine.jax_flat.custom import RollingJaxCall, StatelessJaxCall, StatelessJaxFunction, rolling, stateless
from trading_dsl_engine.jax_flat.engine import JaxFlatRuntime, compile_formula
from trading_dsl_engine._native_build import ensure_native_extension_current


def _load_cpp_flat():
    module_name = __name__ + "._cpp_flat"
    spec = importlib.util.find_spec(module_name)
    extension = Path(spec.origin) if spec is not None and spec.origin is not None else None
    ensure_native_extension_current(Path(__file__).resolve().parents[3], "cpp_flat", extension)
    importlib.invalidate_caches()
    return importlib.import_module(module_name)


def __getattr__(name: str):
    if name == "_cpp_flat":
        return _load_cpp_flat()
    if name == "CppFlatRuntime":
        from trading_dsl_engine.jax_flat.engine_cpp import CppFlatRuntime

        return CppFlatRuntime
    raise AttributeError(name)


__all__ = [
    "JaxFlatRuntime",
    "CppFlatRuntime",
    "_cpp_flat",
    "compile_formula",
    "StatelessJaxCall",
    "RollingJaxCall",
    "StatelessJaxFunction",
    "stateless",
    "rolling",
    "MetadataConfig",
    "NodeMetadata",
    "TypeRelationGraph",
    "UnitInfo",
    "ValueRange",
    "field",
    "metadata",
]
