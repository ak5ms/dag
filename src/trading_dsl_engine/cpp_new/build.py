"""Concurrent-safe, content-addressed generated-source cache.

Compilation is intentionally a separate build-system concern.  A cache entry is
published by directory rename only after every artifact and its manifest exist.
"""
from __future__ import annotations
from dataclasses import dataclass
import contextlib, fcntl, json, os, pathlib, platform, shutil, tempfile
from trading_dsl_engine.cpp_new.codegen import emit_source


@dataclass(frozen=True)
class BuildArtifact:
    key: str
    directory: pathlib.Path
    source: pathlib.Path
    cached: bool


def materialize(ir, cache_dir=None) -> BuildArtifact:
    root = pathlib.Path(cache_dir or os.environ.get("TDE_CPP_NEW_CACHE", pathlib.Path.home()/".cache"/"trading_dsl_engine"/"cpp_new"))
    fingerprint = f"{ir.digest}:abi={ir.abi_version}:python={platform.python_version()}:machine={platform.machine()}"
    import hashlib
    key = hashlib.sha256(fingerprint.encode()).hexdigest()
    target, lock_path = root/key, root/f"{key}.lock"
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (target/"complete").exists(): return BuildArtifact(key, target, target/"formula.cpp", True)
        temporary = pathlib.Path(tempfile.mkdtemp(prefix=f".{key}.", dir=root))
        try:
            (temporary/"formula.cpp").write_text(emit_source(ir))
            (temporary/"manifest.json").write_text(json.dumps({"key": key, "fingerprint": fingerprint}, sort_keys=True))
            (temporary/"complete").write_text("ok\n")
            with contextlib.suppress(FileExistsError): os.rename(temporary, target)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return BuildArtifact(key, target, target/"formula.cpp", False)
