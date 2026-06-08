from __future__ import annotations

from pathlib import Path

import includeigen
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


ext_modules = [
    Pybind11Extension(
        "trading_dsl_engine.jax_flat._cpp_flat",
        ["src/trading_dsl_engine/jax_flat/engine.cpp"],
        depends=["src/trading_dsl_engine/jax_flat/ops.cpp"],
        include_dirs=[includeigen.get_include(), "/usr/include/eigen3"],
        cxx_std=23,
        extra_compile_args=["-O3", "-std=c++2b"],
    )
]

setup(
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
