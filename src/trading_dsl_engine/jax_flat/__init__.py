from pathlib import Path
import importlib
import importlib.util
import subprocess
import sys

from trading_dsl_engine.base.metadata import MetadataConfig, NodeMetadata, TypeRelationGraph, UnitInfo, ValueRange, field, metadata
from trading_dsl_engine.jax_flat.custom import RollingJaxCall, StatelessJaxCall, StatelessJaxFunction, rolling, stateless
from trading_dsl_engine.jax_flat.engine import JaxFlatRuntime, compile_formula
from trading_dsl_engine.jax_flat.optimized import (
    OptimizedJaxFlatRuntime,
    compile_features,
    compile_formula as compile_optimized_formula,
)
# Applies the measured CPU execution policy to the optimized runtime module.
from trading_dsl_engine.jax_flat import optimized_planner as _optimized_planner


def _load_cpp_flat():
    module_name = __name__ + "._cpp_flat"
    if importlib.util.find_spec(module_name) is None:
        if importlib.util.find_spec("setuptools") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "setuptools", "wheel"], check=True)
        root = Path(__file__).resolve().parents[3]
        subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=root, check=True)
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
    "OptimizedJaxFlatRuntime",
    "CppFlatRuntime",
    "_cpp_flat",
    "compile_formula",
    "compile_optimized_formula",
    "compile_features",
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
