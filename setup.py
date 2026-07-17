from __future__ import annotations

import os
from pathlib import Path
import shlex

import includeigen
import jax
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import find_packages, setup


_REPO_ROOT = Path(__file__).resolve().parent


def _env_flag(name: str, *, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def _split_env_flags(name: str) -> list[str]:
    return shlex.split(os.environ.get(name, ""))


def _cpp_pgo_args(*, linking: bool) -> list[str]:
    """Return opt-in GCC-compatible profile-guided optimization flags."""
    mode = os.environ.get("TRADING_DSL_ENGINE_CPP_PGO", "off").strip().lower()
    if mode in {"", "0", "false", "no", "off"}:
        return []
    if mode not in {"generate", "use"}:
        raise ValueError("TRADING_DSL_ENGINE_CPP_PGO must be one of: off, generate, use")
    if os.name == "nt":
        raise RuntimeError("TRADING_DSL_ENGINE_CPP_PGO currently supports Unix-like GCC-compatible toolchains only")

    raw_dir = os.environ.get("TRADING_DSL_ENGINE_CPP_PGO_DIR", ".pgo-data")
    profile_dir = Path(raw_dir).expanduser()
    if not profile_dir.is_absolute():
        profile_dir = _REPO_ROOT / profile_dir
    profile_dir = profile_dir.resolve()

    if mode == "generate":
        profile_dir.mkdir(parents=True, exist_ok=True)
        return [f"-fprofile-generate={profile_dir}"]

    if not profile_dir.is_dir():
        raise RuntimeError(
            f"PGO profile directory does not exist: {profile_dir}. "
            "Build with TRADING_DSL_ENGINE_CPP_PGO=generate and run the training workload first."
        )
    args = [f"-fprofile-use={profile_dir}"]
    if not linking:
        args.append("-fprofile-correction")
    return args


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
    return args + _cpp_pgo_args(linking=False) + _split_env_flags("TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS")


def _cpp_link_args() -> list[str]:
    if os.name == "nt":
        args = ["/LTCG"] if _env_flag("TRADING_DSL_ENGINE_CPP_LTO", default=True) else []
        return args + _split_env_flags("TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS")

    args = ["-Wl,-O3"]
    if _env_flag("TRADING_DSL_ENGINE_CPP_LTO", default=True):
        args.append("-flto")
    return args + _cpp_pgo_args(linking=True) + _split_env_flags("TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS")


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
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
