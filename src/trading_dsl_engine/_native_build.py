from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time


_LOCAL_INCLUDE = re.compile(rb'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)
_BUILD_ENV = (
    "CC",
    "CXX",
    "CFLAGS",
    "CXXFLAGS",
    "LDFLAGS",
    "TRADING_DSL_ENGINE_CPP_NATIVE",
    "TRADING_DSL_ENGINE_CPP_LTO",
    "TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS",
    "TRADING_DSL_ENGINE_CPP_EXTRA_LINK_FLAGS",
)
_EXTENSIONS = {
    "cpp_flat": (
        "src/trading_dsl_engine/jax_flat/engine.cpp",
        "src/trading_dsl_engine/jax_flat/_cpp_flat.build.json",
    ),
    "eigen_nnqp": (
        "src/trading_dsl_engine/jax_ffi/nnqp/eigen_nnqp.cc",
        "src/trading_dsl_engine/jax_ffi/nnqp/_eigen_nnqp.build.json",
    ),
}


def _resolve_local_include(source: Path, include: bytes, root: Path) -> Path | None:
    name = os.fsdecode(include)
    for candidate in (source.parent / name, root / name, root / "src" / name):
        candidate = candidate.resolve()
        if candidate.is_file() and (candidate == root or root in candidate.parents):
            return candidate
    return None


def _native_dependencies(root: Path, source_name: str) -> tuple[Path, ...]:
    """Return an extension's complete repository-local dependency closure."""
    root = root.resolve()
    pending = [root / source_name]
    seen: set[Path] = set()
    while pending:
        source = pending.pop().resolve()
        if source in seen:
            continue
        if not source.is_file():
            raise FileNotFoundError(f"native extension source is missing: {source}")
        seen.add(source)
        data = source.read_bytes()
        for include in _LOCAL_INCLUDE.findall(data):
            dependency = _resolve_local_include(source, include, root)
            if dependency is not None and dependency not in seen:
                pending.append(dependency)
    for build_input in (root / "setup.py", root / "pyproject.toml"):
        if build_input.is_file():
            seen.add(build_input.resolve())
    return tuple(sorted(seen))


def native_source_fingerprint(root: Path, extension_name: str = "cpp_flat") -> str:
    """Hash sources, transitive local includes, build settings, and the Python ABI."""
    root = root.resolve()
    digest = hashlib.sha256()
    source_name, _ = _EXTENSIONS[extension_name]
    for path in _native_dependencies(root, source_name):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    build_identity = {
        "abi": sys.implementation.cache_tag,
        "platform": platform.platform(),
        "env": {name: os.environ.get(name) for name in _BUILD_ENV},
    }
    digest.update(json.dumps(build_identity, sort_keys=True).encode())
    return digest.hexdigest()


def _read_fingerprint(stamp: Path) -> str | None:
    try:
        value = json.loads(stamp.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value.get("fingerprint") if isinstance(value, dict) else None


def _write_fingerprint(stamp: Path, fingerprint: str) -> None:
    temporary = stamp.with_name(f".{stamp.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps({"fingerprint": fingerprint}, sort_keys=True) + "\n")
    temporary.replace(stamp)


@contextmanager
def _build_lock(root: Path, timeout: float = 1800.0):
    """Serialize editable-tree builds without requiring a platform lock module."""
    lock = root / ".native-extension-build.lock"
    deadline = time.monotonic() + timeout
    while True:
        try:
            lock.mkdir()
            break
        except FileExistsError:
            try:
                stale = time.time() - lock.stat().st_mtime > timeout
            except FileNotFoundError:
                continue
            if stale:
                shutil.rmtree(lock, ignore_errors=True)
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for native extension build lock: {lock}")
            time.sleep(0.05)
    try:
        yield
    finally:
        shutil.rmtree(lock, ignore_errors=True)


def ensure_native_extension_current(root: Path, extension_name: str, extension: Path | None) -> None:
    """Rebuild an editable checkout's extension when any build input changed."""
    root = root.resolve()
    setup = root / "setup.py"
    if not setup.is_file():
        return  # Installed wheels do not carry build sources and are immutable.
    _, stamp_name = _EXTENSIONS[extension_name]
    stamp = root / stamp_name
    fingerprint = native_source_fingerprint(root, extension_name)
    if extension is not None and extension.is_file() and _read_fingerprint(stamp) == fingerprint:
        return
    with _build_lock(root):
        fingerprint = native_source_fingerprint(root, extension_name)
        if extension is not None and extension.is_file() and _read_fingerprint(stamp) == fingerprint:
            return
        if importlib.util.find_spec("setuptools") is None:
            subprocess.run([sys.executable, "-m", "pip", "install", "setuptools", "wheel"], check=True)
        subprocess.run(
            [sys.executable, "setup.py", "build_ext", "--inplace", "--force"],
            cwd=root,
            check=True,
        )
        # setup.py builds both native modules. Stamp both dependency closures so
        # importing the other module does not immediately repeat the same build.
        for built_name, (_, built_stamp) in _EXTENSIONS.items():
            _write_fingerprint(root / built_stamp, native_source_fingerprint(root, built_name))
