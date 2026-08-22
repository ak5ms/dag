from __future__ import annotations

import os
import shlex

import includeigen
import jax
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _split_env_flags(name: str) -> list[str]:
    return shlex.split(os.environ.get(name, ""))


def _cpp_compile_args() -> list[str]:
    """Return aggressive-but-IEEE-safe native extension compile flags."""
    if os.name == "nt":
        args = ["/O2", "/DNDEBUG", "/DEIGEN_NO_DEBUG"]
        if _env_flag("TRADING_DSL_ENGINE_CPP_LTO", default=True):
            args.append("/GL")
        return args + _split_env_flags("TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS")

    args = [
        "-O3",
        "-DNDEBUG",
        "-DEIGEN_NO_DEBUG",
        "-fvisibility=hidden",
        "-fno-math-errno",
        "-funroll-loops",
    ]
    if _env_flag("TRADING_DSL_ENGINE_CPP_NATIVE", default=True):
        args.extend(["-march=native", "-mtune=native"])
    if _env_flag("TRADING_DSL_ENGINE_CPP_LTO", default=True):
        args.append("-flto")
    return args + _split_env_flags("TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS")


def _cpp_link_args() -> list[str]:
    if os.name == "nt":
        args = ["/LTCG"] if _env_flag("TRADING_DSL_ENGINE_CPP_LTO", default=True) else []
        return args + _split_env_flags("TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS")

    args = ["-Wl,-O3"]
    if _env_flag("TRADING_DSL_ENGINE_CPP_LTO", default=True):
        args.append("-flto")
    return args + _split_env_flags("TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS")


ext_modules = [
    Pybind11Extension(
        "trading_dsl_engine.jax_flat._cpp_flat",
        ["src/trading_dsl_engine/jax_flat/engine.cpp"],
        depends=["src/trading_dsl_engine/jax_flat/ops.cpp"],
        include_dirs=["src", includeigen.get_include(), "/usr/include/eigen3"],
        cxx_std=23,
        extra_compile_args=_cpp_compile_args(),
        extra_link_args=_cpp_link_args(),
    ),
    Pybind11Extension(
        "trading_dsl_engine.jax_ffi.nnqp._eigen_nnqp",
        ["src/trading_dsl_engine/jax_ffi/nnqp/eigen_nnqp.cc"],
        include_dirs=["src", jax.ffi.include_dir(), includeigen.get_include(), "/usr/include/eigen3"],
        cxx_std=17,
        extra_compile_args=_cpp_compile_args() + ["-DEIGEN_MPL2_ONLY"],
        extra_link_args=_cpp_link_args(),
    ),
]

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={
        "trading_dsl_engine.cpp_stream": [
            "cpp/stackdsl/*.hpp",
            "cpp/stackdsl/ops/*.hpp",
            "python/templates/*.j2",
            "optimizer/templates/*.j2",
        ],
    },
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
