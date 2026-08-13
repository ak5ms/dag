from __future__ import annotations

import ctypes
from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Mapping

import numpy as np

from trading_dsl_engine.cpp_stream.python.lowering import Plan
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.cpp_stream.python.parallel import ParallelPlan
from trading_dsl_engine.cpp_stream.python.sources import SourceValue, open_source_mapping
from trading_dsl_engine.ir.program import Program


_PARALLEL_MODES = {"serial": 0, "rows": 1, "lanes": 2}
_AUTO_TARGET_WORK_PER_THREAD = 262_144


def _close_memmap(array: np.memmap) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()


def _available_python_cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return max(1, os.cpu_count() or 1)


@dataclass(frozen=True, slots=True)
class RunResult:
    output_path: Path
    rows: int
    seconds: float
    output_rows: int
    output_shape: tuple[int, ...]
    output_mode: str
    cpu_seconds: float
    threads: int
    available_cpus: int
    parallel_mode: str
    output_dtype: str = "float64"
    data_offset: int = 0

    @property
    def rows_per_second(self) -> float:
        return float("inf") if self.seconds == 0.0 else self.rows / self.seconds

    @property
    def average_busy_cores(self) -> float:
        return 0.0 if self.seconds == 0.0 else self.cpu_seconds / self.seconds

    def load(self, *, mmap_mode: str | None = "r") -> np.ndarray:
        if self.output_path.suffix.lower() == ".npy":
            return np.load(self.output_path, mmap_mode=mmap_mode, allow_pickle=False)
        if mmap_mode is None:
            values = np.fromfile(self.output_path, dtype=self.output_dtype)
            return values.reshape(self.output_shape or ())
        return np.memmap(
            self.output_path,
            mode=mmap_mode,
            dtype=self.output_dtype,
            offset=self.data_offset,
            shape=self.output_shape or (),
            order="C",
        )


class CppStreamRuntime:
    def __init__(self, *, program: Program, plan: Plan, library_path: Path,
                 generated_cpp: Path, n_instruments: int,
                 input_types: tuple[InputTypeSpec, ...],
                 parallel_plan: ParallelPlan,
                 bound_sources: Mapping[str, SourceValue] | None = None) -> None:
        self.program = program
        self.plan = plan
        self.library_path = Path(library_path)
        self.generated_cpp = Path(generated_cpp)
        self.n_instruments = int(n_instruments)
        self.input_types = tuple(input_types)
        self.parallel_plan = parallel_plan
        self.bound_sources = None if bound_sources is None else dict(bound_sources)
        self._library: ctypes.CDLL | None = None

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.program.input_names

    def plot(self, backend: str = "pydot", *, show: bool = True,
             rankdir: str = "LR", figsize: tuple[float, float] | None = None):
        return self.program.plot(backend=backend, show=show, rankdir=rankdir, figsize=figsize)

    def _load(self) -> ctypes.CDLL:
        if self._library is None:
            lib = ctypes.CDLL(str(self.library_path))
            lib.cpp_stream_run_arrays.argtypes = [
                ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t), ctypes.c_size_t, ctypes.c_char_p,
                ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int,
                ctypes.c_bool, ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                ctypes.POINTER(ctypes.c_size_t), ctypes.POINTER(ctypes.c_size_t),
            ]
            lib.cpp_stream_run_arrays.restype = ctypes.c_int
            lib.cpp_stream_last_error.argtypes = []
            lib.cpp_stream_last_error.restype = ctypes.c_char_p
            self._library = lib
        return self._library

    def _validate_names(self, data: Mapping[str, object]) -> None:
        missing = [name for name in self.input_names if name not in data]
        if missing:
            raise KeyError(f"missing cpp_stream source(s): {missing}")

    @staticmethod
    def _validate_writeback(async_writeback_mb: int) -> int:
        if async_writeback_mb < 0:
            raise ValueError("async_writeback_mb must be >= 0")
        return int(async_writeback_mb) * 1024 * 1024

    @staticmethod
    def _validate_threads(threads: int) -> int:
        value = int(threads)
        if value < 0:
            raise ValueError("threads must be >= 0; zero selects automatic execution")
        return value

    def _resolved_request(self, threads: int, rows: int) -> int:
        value = self._validate_threads(threads)
        if value > 0:
            return value
        if not self.parallel_plan.auto_multicore or rows <= 0:
            return 1
        available = _available_python_cpus()
        maximum = self.n_instruments if self.parallel_plan.mode == "lanes" else rows
        total_work = rows * max(1, self.n_instruments) * max(1, self.parallel_plan.work_score)
        useful = max(1, min(available, max(1, maximum),
                            max(1, total_work // _AUTO_TARGET_WORK_PER_THREAD)))
        return useful if useful >= 2 else 1

    def _raise_native(self, code: int, lib: ctypes.CDLL) -> None:
        if code == 0:
            return
        detail = lib.cpp_stream_last_error()
        message = detail.decode() if detail else ""
        meanings = {
            1: "input count or pointer validation failed",
            2: "input row width validation failed",
            3: "input sources have different row counts",
            4: "group capacity exceeded or dense key fell outside its declared domain",
            5: "lane-parallel output shape is not partitionable by instrument",
            6: "output payload offset is not aligned for float64",
        }
        base = meanings.get(code, f"native runtime returned error code {code}")
        raise RuntimeError(f"cpp_stream: {base}" + (f": {message}" if message else ""))

    @staticmethod
    def _default_output_path() -> Path:
        descriptor, name = tempfile.mkstemp(prefix="cpp_stream_", suffix=".npy")
        os.close(descriptor)
        path = Path(name)
        path.unlink(missing_ok=True)
        return path

    @staticmethod
    def _prepare_output(output: Path, logical_shape: tuple[int, ...]) -> int:
        if output.suffix.lower() != ".npy":
            return 0
        if output.exists():
            try:
                existing = np.lib.format.open_memmap(output, mode="r+")
                try:
                    if existing.dtype == np.dtype(np.float64) and tuple(existing.shape) == logical_shape and existing.flags.c_contiguous:
                        return int(existing.offset)
                finally:
                    _close_memmap(existing)
            except (OSError, ValueError, TypeError):
                pass
        created = np.lib.format.open_memmap(output, mode="w+", dtype=np.float64,
                                             shape=logical_shape, fortran_order=False)
        try:
            return int(created.offset)
        finally:
            _close_memmap(created)

    def run(self, data: Mapping[str, SourceValue] | None = None, *,
            out_path: str | Path | None = None, async_writeback_mb: int = 0,
            threads: int = 0, pin_threads: bool = False) -> RunResult:
        """Run natively; omitted ``threads`` selects safe automatic execution."""
        selected = self.bound_sources if data is None else data
        if selected is None:
            raise ValueError("no sources are bound; pass a source mapping to run(...) or compile_formula(..., data)")
        self._validate_names(selected)
        writeback = self._validate_writeback(async_writeback_mb)
        prepared = open_source_mapping(selected, self.input_names, self.input_types)
        try:
            row_counts = {item.info.rows for item in prepared}
            if len(row_counts) != 1:
                details = {name: item.info.rows for name, item in zip(self.input_names, prepared)}
                raise ValueError(f"cpp_stream sources have different row counts: {details}")
            processed_input_rows = next(iter(row_counts))
            requested_threads = self._resolved_request(threads, processed_input_rows)
            logical_shape = self.plan.output_shape if self.plan.output_mode == "final" else (processed_input_rows,) + self.plan.output_shape
            output = self._default_output_path() if out_path is None else Path(out_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            output_offset = self._prepare_output(output, logical_shape)
            pointers = (ctypes.c_void_p * len(prepared))(*(ctypes.c_void_p(item.data_pointer) for item in prepared))
            input_rows = (ctypes.c_size_t * len(prepared))(*(item.info.rows for item in prepared))
            input_widths = (ctypes.c_size_t * len(prepared))(*(item.info.input_type.row_width for item in prepared))
            rows = ctypes.c_size_t()
            seconds = ctypes.c_double()
            cpu_seconds = ctypes.c_double()
            actual_threads = ctypes.c_size_t()
            available_cpus = ctypes.c_size_t()
            lib = self._load()
            code = lib.cpp_stream_run_arrays(
                pointers, input_rows, input_widths, len(prepared), str(output).encode(),
                output_offset, writeback, requested_threads,
                _PARALLEL_MODES[self.parallel_plan.mode], bool(pin_threads),
                ctypes.byref(rows), ctypes.byref(seconds), ctypes.byref(cpu_seconds),
                ctypes.byref(actual_threads), ctypes.byref(available_cpus),
            )
            self._raise_native(code, lib)
            processed_rows = int(rows.value)
            return RunResult(
                output_path=output, rows=processed_rows, seconds=float(seconds.value),
                output_rows=1 if self.plan.output_mode == "final" else processed_rows,
                output_shape=logical_shape, output_mode=self.plan.output_mode,
                cpu_seconds=float(cpu_seconds.value), threads=int(actual_threads.value),
                available_cpus=int(available_cpus.value),
                parallel_mode=self.parallel_plan.mode, output_dtype="float64",
                data_offset=output_offset,
            )
        finally:
            for item in reversed(prepared):
                item.close()

    def explain(self) -> str:
        lines = [
            f"cpp_stream N={self.n_instruments}", f"inputs={self.input_names}",
            f"input_types={self.input_types}", f"sources_bound={self.bound_sources is not None}",
            f"parallel_mode={self.parallel_plan.mode}",
            f"parallel_reason={self.parallel_plan.reason}",
            f"parallel_auto_multicore={self.parallel_plan.auto_multicore}",
            f"parallel_work_score={self.parallel_plan.work_score}",
            "default_threads=automatic (threads=0)",
            f"scratch_slots={self.plan.scratch_slots}", f"output_mode={self.plan.output_mode}",
            f"output_shape={self.plan.output_shape}",
            "default_output=.npy (direct payload mmap; no conversion copy)",
        ]
        for i, stage in enumerate(self.plan.stages):
            lines.append(f"{i}: {stage.kind} -> {'output' if stage.out.slot is None else f'slot {stage.out.slot}'}")
        return "\n".join(lines)


__all__ = ["CppStreamRuntime", "RunResult"]
