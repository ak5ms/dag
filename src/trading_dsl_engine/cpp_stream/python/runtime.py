from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trading_dsl_engine.cpp_stream.python.lowering import Plan
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.cpp_stream.python.sources import (
    SourceValue,
    open_source_mapping,
)
from trading_dsl_engine.ir.program import Program


@dataclass(frozen=True, slots=True)
class RunResult:
    output_path: Path
    rows: int
    seconds: float
    output_rows: int
    output_shape: tuple[int, ...]
    output_mode: str

    @property
    def rows_per_second(self) -> float:
        return float("inf") if self.seconds == 0.0 else self.rows / self.seconds


class CppStreamRuntime:
    def __init__(
        self,
        *,
        program: Program,
        plan: Plan,
        library_path: Path,
        generated_cpp: Path,
        n_instruments: int,
        input_types: tuple[InputTypeSpec, ...],
        bound_sources: Mapping[str, SourceValue] | None = None,
    ) -> None:
        self.program = program
        self.plan = plan
        self.library_path = Path(library_path)
        self.generated_cpp = Path(generated_cpp)
        self.n_instruments = int(n_instruments)
        self.input_types = tuple(input_types)
        self.bound_sources = (
            None if bound_sources is None else dict(bound_sources)
        )
        self._library: ctypes.CDLL | None = None

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.program.input_names

    def _load(self) -> ctypes.CDLL:
        if self._library is None:
            lib = ctypes.CDLL(str(self.library_path))
            lib.cpp_stream_run_arrays.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_double),
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
        extra = sorted(set(data) - set(self.input_names))
        if extra:
            raise KeyError(f"unexpected cpp_stream source(s): {extra}")

    @staticmethod
    def _validate_writeback(async_writeback_mb: int) -> int:
        if async_writeback_mb < 0:
            raise ValueError("async_writeback_mb must be >= 0")
        return int(async_writeback_mb) * 1024 * 1024

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
        }
        base = meanings.get(code, f"native runtime returned error code {code}")
        raise RuntimeError(f"cpp_stream: {base}" + (f": {message}" if message else ""))

    def run(
        self,
        data: Mapping[str, SourceValue] | None = None,
        *,
        out_path: str | Path,
        async_writeback_mb: int = 0,
    ) -> RunResult:
        """Execute compatible heterogeneous sources through one native call.

        Every input is independently dispatched to a source adapter. A single run
        may therefore mix `.npy`, headerless raw files, in-memory arrays, and custom
        registered source types. Sources bound by ``compile_formula(..., data)`` are
        used when ``data`` is omitted.
        """
        selected = self.bound_sources if data is None else data
        if selected is None:
            raise ValueError(
                "no sources are bound; pass a source mapping to run(...) or "
                "compile_formula(..., data)"
            )
        self._validate_names(selected)
        writeback = self._validate_writeback(async_writeback_mb)
        prepared = open_source_mapping(
            selected,
            self.input_names,
            self.input_types,
        )
        try:
            row_counts = {item.info.rows for item in prepared}
            if len(row_counts) != 1:
                details = {
                    name: item.info.rows
                    for name, item in zip(self.input_names, prepared)
                }
                raise ValueError(
                    f"cpp_stream sources have different row counts: {details}"
                )
            pointers = (ctypes.c_void_p * len(prepared))(
                *(ctypes.c_void_p(item.data_pointer) for item in prepared)
            )
            input_rows = (ctypes.c_size_t * len(prepared))(
                *(item.info.rows for item in prepared)
            )
            input_widths = (ctypes.c_size_t * len(prepared))(
                *(item.info.input_type.row_width for item in prepared)
            )
            rows = ctypes.c_size_t()
            seconds = ctypes.c_double()
            output = Path(out_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            lib = self._load()
            code = lib.cpp_stream_run_arrays(
                pointers,
                input_rows,
                input_widths,
                len(prepared),
                str(output).encode(),
                writeback,
                ctypes.byref(rows),
                ctypes.byref(seconds),
            )
            self._raise_native(code, lib)
            processed_rows = int(rows.value)
            logical_shape = (
                self.plan.output_shape
                if self.plan.output_mode == "final"
                else (processed_rows,) + self.plan.output_shape
            )
            return RunResult(
                output_path=output,
                rows=processed_rows,
                seconds=float(seconds.value),
                output_rows=1 if self.plan.output_mode == "final" else processed_rows,
                output_shape=logical_shape,
                output_mode=self.plan.output_mode,
            )
        finally:
            for item in reversed(prepared):
                item.close()

    def explain(self) -> str:
        lines = [
            f"cpp_stream N={self.n_instruments}",
            f"inputs={self.input_names}",
            f"input_types={self.input_types}",
            f"sources_bound={self.bound_sources is not None}",
            f"scratch_slots={self.plan.scratch_slots}",
            f"output_mode={self.plan.output_mode}",
            f"output_shape={self.plan.output_shape}",
        ]
        for i, stage in enumerate(self.plan.stages):
            lines.append(
                f"{i}: {stage.kind} -> "
                f"{'output' if stage.out.slot is None else f'slot {stage.out.slot}'}"
            )
        return "\n".join(lines)


__all__ = ["CppStreamRuntime", "RunResult"]
