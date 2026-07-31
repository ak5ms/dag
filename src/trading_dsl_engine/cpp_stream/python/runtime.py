from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trading_dsl_engine.cpp_stream.python.lowering import Plan
from trading_dsl_engine.ir.program import Program


@dataclass(frozen=True, slots=True)
class RunResult:
    output_path: Path
    rows: int
    seconds: float

    @property
    def rows_per_second(self) -> float:
        return float("inf") if self.seconds == 0.0 else self.rows / self.seconds


class CppStreamRuntime:
    def __init__(self, *, program: Program, plan: Plan, library_path: Path, generated_cpp: Path, n_instruments: int) -> None:
        self.program = program
        self.plan = plan
        self.library_path = Path(library_path)
        self.generated_cpp = Path(generated_cpp)
        self.n_instruments = int(n_instruments)
        self._library: ctypes.CDLL | None = None

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.program.input_names

    def _load(self) -> ctypes.CDLL:
        if self._library is None:
            lib = ctypes.CDLL(str(self.library_path))
            lib.cpp_stream_run_files.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.c_size_t, ctypes.c_char_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_double)]
            lib.cpp_stream_run_files.restype = ctypes.c_int
            lib.cpp_stream_last_error.argtypes = []
            lib.cpp_stream_last_error.restype = ctypes.c_char_p
            self._library = lib
        return self._library

    def run_files(self, data: Mapping[str, str | Path], *, out_path: str | Path, async_writeback_mb: int = 0) -> RunResult:
        missing = [name for name in self.input_names if name not in data]
        if missing:
            raise KeyError(f"missing cpp_stream input file(s): {missing}")
        extra = sorted(set(data) - set(self.input_names))
        if extra:
            raise KeyError(f"unexpected cpp_stream input file(s): {extra}")
        if async_writeback_mb < 0:
            raise ValueError("async_writeback_mb must be >= 0")
        encoded = [str(Path(data[name])).encode() for name in self.input_names]
        argv = (ctypes.c_char_p * len(encoded))(*encoded)
        rows = ctypes.c_size_t()
        seconds = ctypes.c_double()
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lib = self._load()
        code = lib.cpp_stream_run_files(argv, len(encoded), str(output).encode(), int(async_writeback_mb) * 1024 * 1024, ctypes.byref(rows), ctypes.byref(seconds))
        if code != 0:
            detail = lib.cpp_stream_last_error()
            message = detail.decode() if detail else ""
            meanings = {1: "input count/path validation failed", 2: "an input file is not a whole number of rows", 3: "input files have different row counts", 4: "group capacity exceeded or dense key fell outside its declared domain"}
            base = meanings.get(code, f"native runtime returned error code {code}")
            raise RuntimeError(f"cpp_stream: {base}" + (f": {message}" if message else ""))
        return RunResult(output_path=output, rows=int(rows.value), seconds=float(seconds.value))

    def explain(self) -> str:
        lines = [f"cpp_stream N={self.n_instruments}", f"inputs={self.input_names}", f"scratch_slots={self.plan.scratch_slots}"]
        for i, stage in enumerate(self.plan.stages):
            lines.append(f"{i}: {stage.kind} -> {'output' if stage.out.slot is None else f'slot {stage.out.slot}'}")
        return "\n".join(lines)
