from __future__ import annotations

import json
from pathlib import Path
import os
import shutil
import subprocess
import sys

import pytest


CHUNK_SIZE = 4
OPERATOR_COUNT = 40
CHUNKS = (OPERATOR_COUNT + CHUNK_SIZE - 1) // CHUNK_SIZE
AUDIT_CHUNK = int(os.environ.get("RISKMINER_OPERATOR_TEST_CHUNK", "0"))


def test_riskminer_operator_chunk_lowers_and_executes_in_cpp_stream(
    tmp_path: Path,
):
    """Compile one small catalog slice in a fresh native process.

    Keeping each slice in its own process avoids retaining template/compiler or
    JAX runtime state across the complete 40-operator audit.
    """

    if sys.platform == "win32" or shutil.which("g++") is None:
        pytest.skip("cpp_stream native operator audit requires POSIX and g++")
    if not 0 <= AUDIT_CHUNK < CHUNKS:
        raise ValueError(
            f"RISKMINER_OPERATOR_TEST_CHUNK must be in [0, {CHUNKS})"
        )
    chunk = AUDIT_CHUNK
    root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": os.pathsep.join(
                filter(
                    None,
                    (str(root / "src"), os.environ.get("PYTHONPATH", "")),
                )
            ),
            "RISKMINER_OPERATOR_AUDIT_CHILD": "1",
            "RISKMINER_OPERATOR_AUDIT_CHUNK": str(chunk),
            "RISKMINER_OPERATOR_AUDIT_ROWS": "32",
            "RISKMINER_OPERATOR_AUDIT_INSTRUMENTS": "4",
            "RISKMINER_OPERATOR_AUDIT_CHUNK_SIZE": str(CHUNK_SIZE),
            "RISKMINER_OPERATOR_AUDIT_OUTPUT_DIR": str(tmp_path),
            "TRADING_DSL_ENGINE_CPP_STREAM_CACHE": str(
                tmp_path / "native_cache"
            ),
            # Correctness audit, not a throughput benchmark.
            "TRADING_DSL_ENGINE_CPP_LTO": "0",
            "TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS": "-O1",
        }
    )
    stdout_path = tmp_path / "stdout.log"
    stderr_path = tmp_path / "stderr.log"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        process = subprocess.run(
            [
                sys.executable,
                str(root / "scripts" / "audit_riskminer_operator_catalog.py"),
            ],
            cwd=root,
            env=environment,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
            timeout=90,
        )
    output = stdout_path.read_text()
    errors = stderr_path.read_text()
    if process.returncode != 0:
        raise AssertionError(
            f"operator audit chunk {chunk} failed\nSTDOUT:\n"
            + output
            + "\nSTDERR:\n"
            + errors
        )
    report = json.loads(output.strip().splitlines()[-1])
    assert report["chunk"] == chunk
    assert report["status"] == "passed"
    assert 1 <= len(report["operators"]) <= CHUNK_SIZE
