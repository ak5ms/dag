from __future__ import annotations

from pathlib import Path

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


ext_modules = [
    Pybind11Extension(
        "trading_dsl_engine.jax_flat._cpp_flat",
        ["src/trading_dsl_engine/jax_flat/engine.cpp"],
        cxx_std=17,
        extra_compile_args=["-O3"],
    )
]

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
