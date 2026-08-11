from __future__ import annotations

"""Compile and execute every RiskMiner operator through ``cpp_stream``.

Run from the repository root:

    PYTHONPATH=src python scripts/audit_riskminer_operator_catalog.py

The parent process launches one fresh child process per small operator group so
large template graphs do not accumulate compiler memory. Configuration is via
``RISKMINER_OPERATOR_AUDIT_*`` environment variables; no CLI is required.
"""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile

import numpy as np

from flows.riskminer import SearchShape, SemanticInfo, default_operator_catalog
from trading_dsl_engine.base.dsl import cat, gt, var
from trading_dsl_engine.cpp_stream import compile_formula


CHUNK_SIZE = int(os.environ.get("RISKMINER_OPERATOR_AUDIT_CHUNK_SIZE", "4"))
ROWS = int(os.environ.get("RISKMINER_OPERATOR_AUDIT_ROWS", "64"))
INSTRUMENTS = int(os.environ.get("RISKMINER_OPERATOR_AUDIT_INSTRUMENTS", "4"))
OUTPUT_DIR = Path(
    os.environ.get(
        "RISKMINER_OPERATOR_AUDIT_OUTPUT_DIR",
        "/tmp/riskminer-operator-audit",
    )
)
CHILD = os.environ.get("RISKMINER_OPERATOR_AUDIT_CHILD", "0") == "1"
CHUNK = int(os.environ.get("RISKMINER_OPERATOR_AUDIT_CHUNK", "0"))
CHILD_TIMEOUT_SECONDS = float(
    os.environ.get("RISKMINER_OPERATOR_AUDIT_CHILD_TIMEOUT", "60")
)
CHILD_RETRIES = int(
    os.environ.get("RISKMINER_OPERATOR_AUDIT_CHILD_RETRIES", "1")
)


def _expressions_for_chunk(chunk: int):
    rng = np.random.default_rng(202)
    sources = {
        "x": np.exp(rng.normal(scale=0.1, size=(ROWS, INSTRUMENTS))),
        "y": np.exp(rng.normal(scale=0.1, size=(ROWS, INSTRUMENTS))),
        "selector": rng.normal(size=(ROWS, INSTRUMENTS)),
    }
    row = SemanticInfo(
        frozenset({"numeric", "dimensionless"}),
        SearchShape.ROW,
        lower=0.01,
    )
    boolean_row = SemanticInfo(
        frozenset({"numeric", "dimensionless", "boolean"}),
        SearchShape.BOOLEAN_ROW,
        lower=0.0,
        upper=1.0,
        integer=True,
    )
    literal = SemanticInfo(
        frozenset(
            {"numeric", "dimensionless", "parameter", "compile_time"}
        ),
        SearchShape.SCALAR,
        lower=5.0,
        upper=5.0,
        integer=True,
        static=True,
        role="parameter",
    )
    expr = {
        "x": var("x"),
        "y": var("y"),
        "selector": var("selector"),
        "condition": gt(var("x"), var("y")),
    }
    schemas = default_operator_catalog()[
        chunk * CHUNK_SIZE : (chunk + 1) * CHUNK_SIZE
    ]
    outputs = []
    names = []
    for schema in schemas:
        if schema.name == "where":
            expressions = [expr["condition"], expr["x"], expr["y"]]
            semantics = [boolean_row, row, row]
            literals = [None, None, None]
        elif schema.family in {"temporal", "history", "rolling"}:
            expressions = [expr["x"], 5.0]
            semantics = [row, literal]
            literals = [None, 5.0]
        elif schema.family == "rolling_pair":
            expressions = [expr["x"], expr["y"], 5.0]
            semantics = [row, row, literal]
            literals = [None, None, 5.0]
        elif schema.family == "dynamic_temporal":
            expressions = [expr["x"], expr["selector"]]
            semantics = [row, row]
            literals = [None, None]
        elif schema.family == "dynamic_temporal_pair":
            expressions = [expr["x"], expr["y"], expr["selector"]]
            semantics = [row, row, row]
            literals = [None, None, None]
        elif schema.arity == 1:
            expressions = [expr["x"]]
            semantics = [row]
            literals = [None]
        elif schema.arity == 2:
            expressions = [expr["x"], expr["y"]]
            semantics = [row, row]
            literals = [None, None]
        else:
            raise RuntimeError(f"no audit fixture for {schema.name!r}")
        if not schema.validate(semantics):
            raise RuntimeError(f"audit semantics rejected {schema.name!r}")
        outputs.append(schema.build(expressions, literals))
        names.append(schema.name)
    return sources, outputs, names


def _run_child(chunk: int) -> dict[str, object]:
    sources, outputs, names = _expressions_for_chunk(chunk)
    if not outputs:
        return {"chunk": chunk, "operators": [], "status": "empty"}
    formula = outputs[0] if len(outputs) == 1 else cat(*outputs)
    runtime = compile_formula(
        formula,
        sources,
        n_instruments=INSTRUMENTS,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"operators_{chunk:02d}.bin"
    result = runtime.run(
        out_path=output_path,
        threads=1,
        async_writeback_mb=0,
    )
    values = np.fromfile(output_path, dtype=np.float64)
    expected = ROWS * INSTRUMENTS * len(names)
    if values.size != expected:
        raise RuntimeError(
            f"chunk {chunk} emitted {values.size} values; expected {expected}"
        )
    return {
        "chunk": chunk,
        "operators": names,
        "status": "passed",
        "output_shape": list(getattr(result, "output_shape", ())),
        "values": int(values.size),
        "finite": int(np.isfinite(values).sum()),
        "native_seconds": getattr(result, "seconds", None),
    }




def _invoke_child(environment: dict[str, str]) -> tuple[int, str, str]:
    # Use regular temporary files instead of PIPEs.  Compiler grandchildren
    # inherit pipe descriptors and can otherwise make communicate() wait after
    # the Python child has already been terminated.
    with (
        tempfile.TemporaryFile(mode="w+t") as stdout_file,
        tempfile.TemporaryFile(mode="w+t") as stderr_file,
    ):
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve())],
            env=environment,
            text=True,
            stdout=stdout_file,
            stderr=stderr_file,
            start_new_session=True,
        )
        try:
            returncode = int(process.wait(timeout=CHILD_TIMEOUT_SECONDS))
        except subprocess.TimeoutExpired:
            # Kill the whole process group so a compiler grandchild cannot
            # survive and poison later audit chunks.  A retry can reuse any
            # completed cache artifact from a teardown-only stall.
            os.killpg(process.pid, signal.SIGKILL)
            try:
                process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
            returncode = 124
        stdout_file.seek(0)
        stderr_file.seek(0)
        return returncode, stdout_file.read(), stderr_file.read()


def _run_parent() -> None:
    if CHUNK_SIZE <= 0 or ROWS <= 0 or INSTRUMENTS <= 0:
        raise ValueError("audit dimensions and chunk size must be positive")
    operator_count = len(default_operator_catalog())
    chunks = math.ceil(operator_count / CHUNK_SIZE)
    reports = []
    for chunk in range(chunks):
        environment = dict(os.environ)
        environment.update(
            {
                "RISKMINER_OPERATOR_AUDIT_CHILD": "1",
                "RISKMINER_OPERATOR_AUDIT_CHUNK": str(chunk),
            }
        )
        # This is a lowering/execution audit, not a throughput benchmark.
        # Large rolling-template groups can spend minutes in LTO while proving
        # nothing additional about correctness, so use a cold-cache-friendly
        # compile mode unless the caller explicitly overrides it.
        environment.setdefault("TRADING_DSL_ENGINE_CPP_LTO", "0")
        environment.setdefault(
            "TRADING_DSL_ENGINE_CPP_EXTRA_FLAGS", "-O1"
        )
        returncode = 124
        stdout = stderr = ""
        for attempt in range(CHILD_RETRIES + 1):
            returncode, stdout, stderr = _invoke_child(environment)
            if returncode == 0:
                break
            print(
                json.dumps(
                    {
                        "event": "audit_chunk_retry",
                        "chunk": chunk,
                        "attempt": attempt + 1,
                        "returncode": returncode,
                    }
                ),
                flush=True,
            )
        if stdout:
            print(stdout, end="", flush=True)
        if stderr:
            print(stderr, end="", file=sys.stderr, flush=True)
        if returncode != 0:
            raise RuntimeError(
                f"operator audit chunk {chunk} failed with "
                f"status {returncode}"
            )
        reports.append(json.loads(stdout.strip().splitlines()[-1]))
    audited = [name for report in reports for name in report["operators"]]
    if len(audited) != operator_count or len(set(audited)) != operator_count:
        raise RuntimeError("operator audit did not cover every catalog entry once")
    summary = {
        "operator_count": operator_count,
        "operators": audited,
        "chunks": reports,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / "operator_audit.json"
    report_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"event": "audit_done", **summary}), flush=True)
    print(f"report={report_path}", flush=True)


if __name__ == "__main__":
    if CHILD:
        print(json.dumps(_run_child(CHUNK), sort_keys=True), flush=True)
    else:
        _run_parent()
    # Importing the complete RiskMiner package also initializes JAX.  After
    # native compiler/runtime work, interpreter teardown can block in unrelated
    # third-party finalizers even though the audit report is complete.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
