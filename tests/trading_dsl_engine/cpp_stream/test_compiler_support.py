from __future__ import annotations

import importlib
from pathlib import Path
import warnings


compiler_support = importlib.import_module(
    "trading_dsl_engine.cpp_stream.python.compiler_support"
)


def test_auto_compiler_prefers_icpx_on_x86_64(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("TRADING_DSL_ENGINE_CPP_COMPILER_WATERFALL", raising=False)
    monkeypatch.setattr(compiler_support.platform, "machine", lambda: "x86_64")
    paths = {"icpx": "/opt/intel/oneapi/compiler/bin/icpx", "g++": "/usr/bin/g++"}
    monkeypatch.setattr(compiler_support.shutil, "which", paths.get)

    assert compiler_support._compiler() == paths["icpx"]


def test_auto_compiler_discovers_standard_oneapi_install(monkeypatch, tmp_path):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("TRADING_DSL_ENGINE_CPP_COMPILER_WATERFALL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setattr(compiler_support.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        compiler_support.shutil,
        "which",
        lambda name: "/usr/bin/g++" if name == "g++" else None,
    )
    compiler = tmp_path / "2026.1" / "bin" / "icpx"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("#!/bin/sh\n")
    compiler.chmod(0o755)
    monkeypatch.setattr(compiler_support, "_ONEAPI_COMPILER_ROOT", tmp_path)

    assert compiler_support._compiler() == str(compiler)


def test_auto_compiler_falls_back_to_gcc_on_unsupported_arch(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("TRADING_DSL_ENGINE_CPP_COMPILER_WATERFALL", raising=False)
    monkeypatch.setattr(compiler_support.platform, "machine", lambda: "aarch64")
    paths = {"icpx": "/opt/intel/oneapi/compiler/bin/icpx", "g++": "/usr/bin/g++"}
    monkeypatch.setattr(compiler_support.shutil, "which", paths.get)

    assert compiler_support._compiler() == paths["g++"]


def test_explicit_cxx_override_wins(monkeypatch):
    monkeypatch.setenv("CXX", "clang++")
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_COMPILER_WATERFALL",
        "icpx,g++",
    )
    monkeypatch.setattr(compiler_support.platform, "machine", lambda: "x86_64")
    paths = {
        "icpx": "/opt/intel/oneapi/compiler/bin/icpx",
        "clang++": "/usr/bin/clang++",
        "g++": "/usr/bin/g++",
    }
    monkeypatch.setattr(compiler_support.shutil, "which", paths.get)

    assert compiler_support._compiler() == paths["clang++"]


def test_compiler_waterfall_can_be_overridden(monkeypatch):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_COMPILER_WATERFALL",
        "clang++,g++,icpx",
    )
    monkeypatch.setattr(compiler_support.platform, "machine", lambda: "x86_64")
    paths = {
        "clang++": "/usr/bin/clang++",
        "g++": "/usr/bin/g++",
        "icpx": "/opt/intel/oneapi/compiler/bin/icpx",
    }
    monkeypatch.setattr(compiler_support.shutil, "which", paths.get)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert compiler_support._compiler() == paths["clang++"]
    assert caught == []


def test_missing_preferred_icpx_warns_once_and_falls_back(monkeypatch, tmp_path):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("TRADING_DSL_ENGINE_CPP_COMPILER_WATERFALL", raising=False)
    monkeypatch.setattr(compiler_support.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        compiler_support.shutil,
        "which",
        lambda name: "/usr/bin/g++" if name == "g++" else None,
    )
    monkeypatch.setattr(compiler_support, "_ONEAPI_COMPILER_ROOT", tmp_path / "system")
    monkeypatch.setattr(compiler_support, "_warned_missing_icpx", False, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert compiler_support._compiler() == "/usr/bin/g++"
        assert compiler_support._compiler() == "/usr/bin/g++"

    messages = [str(item.message) for item in caught]
    assert len(messages) == 1
    assert "ICX" in messages[0]
    assert "g++" in messages[0]
    assert "install_icx()" in messages[0]
    assert "~/intel/oneapi" in messages[0]


def test_user_home_icx_is_discovered(monkeypatch, tmp_path):
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.delenv("TRADING_DSL_ENGINE_CPP_COMPILER_WATERFALL", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(compiler_support.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(compiler_support.shutil, "which", lambda name: None)
    monkeypatch.setattr(compiler_support, "_ONEAPI_COMPILER_ROOT", tmp_path / "system")
    compiler = tmp_path / "intel" / "oneapi" / "bin" / "icpx"
    compiler.parent.mkdir(parents=True)
    compiler.write_text("#!/bin/sh\n")
    compiler.chmod(0o755)

    assert compiler_support._compiler() == str(compiler)


def test_install_icx_creates_user_home_compiler_without_sudo(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(compiler_support.platform, "system", lambda: "Linux")
    monkeypatch.setattr(compiler_support.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(compiler_support.shutil, "which", lambda name: None)

    downloaded = []

    def fake_urlretrieve(url, destination):
        downloaded.append((url, Path(destination)))
        Path(destination).write_text("#!/bin/sh\n")
        return str(destination), None

    monkeypatch.setattr(compiler_support.urllib.request, "urlretrieve", fake_urlretrieve)

    commands = []

    def fake_run(command, **kwargs):
        commands.append(list(command))
        assert "sudo" not in command
        if command[0] == "bash":
            prefix = Path(command[command.index("-p") + 1])
            conda = prefix / "bin" / "conda"
            conda.parent.mkdir(parents=True, exist_ok=True)
            conda.write_text("#!/bin/sh\n")
            conda.chmod(0o755)
        elif Path(command[0]).name == "conda":
            prefix = Path(command[command.index("-p") + 1])
            compiler = prefix / "bin" / "icpx"
            compiler.parent.mkdir(parents=True, exist_ok=True)
            compiler.write_text("#!/bin/sh\n")
            compiler.chmod(0o755)
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(compiler_support.subprocess, "run", fake_run)

    compiler = Path(compiler_support.install_icx())
    assert compiler == tmp_path / "intel" / "oneapi" / "bin" / "icpx"
    assert downloaded
    assert commands
    assert all("sudo" not in command for command in commands)
    conda_command = next(command for command in commands if Path(command[0]).name == "conda")
    assert "dpcpp_linux-64" in conda_command
    assert "https://software.repos.intel.com/python/conda/" in conda_command


def test_icpx_runtime_link_flags_embed_compiler_runtime(monkeypatch):
    compiler = "/opt/intel/oneapi/compiler/2026.1/bin/icpx"

    def fake_run(command, **kwargs):
        del kwargs
        assert command == [compiler, "-print-file-name=libsvml.so"]
        return type(
            "Result",
            (),
            {
                "returncode": 0,
                "stdout": "/opt/intel/oneapi/compiler/2026.1/lib/libsvml.so\n",
                "stderr": "",
            },
        )()

    monkeypatch.setattr(compiler_support.subprocess, "run", fake_run)

    assert compiler_support._compiler_runtime_link_flags(compiler) == [
        "-Wl,-rpath,/opt/intel/oneapi/compiler/2026.1/lib"
    ]


def test_install_icx_is_publicly_exported():
    from trading_dsl_engine.cpp_stream import install_icx

    assert install_icx is compiler_support.install_icx
