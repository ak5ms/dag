from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from trading_dsl_engine.cpp_stream.python.lowering import Plan
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec, mmap_npy
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
    def __init__(
        self,
        *,
        program: Program,
        plan: Plan,
        library_path: Path,
        generated_cpp: Path,
        n_instruments: int,
        input_types: tuple[InputTypeSpec, ...],
    ) -> None:
        self.program = program
        self.plan = plan
        self.library_path = Path(library_path)
        self.generated_cpp = Path(generated_cpp)
        self.n_instruments = int(n_instruments)
        self.input_types = tuple(input_types)
        self._library: ctypes.CDLL | None = None

    @property
    def input_names(self) -> tuple[str, ...]:
        return self.program.input_names

    def _load(self) -> ctypes.CDLL:
        if self._library is None:
            lib = ctypes.CDLL(str(self.library_path))
            common_tail = [
                ctypes.c_char_p,
                ctypes.c_size_t,
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_double),
            ]
            lib.cpp_stream_run_files.argtypes = [
                ctypes.POINTER(ctypes.c_char_p),
                ctypes.c_size_t,
                *common_tail,
            ]
            lib.cpp_stream_run_files.restype = ctypes.c_int
            lib.cpp_stream_run_arrays.argtypes = [
                ctypes.POINTER(ctypes.c_void_p),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.POINTER(ctypes.c_size_t),
                ctypes.c_size_t,
                *common_tail,
            ]
            lib.cpp_stream_run_arrays.restype = ctypes.c_int
            lib.cpp_stream_last_error.argtypes = []
            lib.cpp_stream_last_error.restype = ctypes.c_char_p
            self._library = lib
        return self._library

    def _validate_names(self, data: Mapping[str, object]) -> None:
        missing = [name for name in self.input_names if name not in data]
        if missing:
            raise KeyError(f"missing cpp_stream input file(s): {missing}")
        extra = sorted(set(data) - set(self.input_names))
        if extra:
            raise KeyError(f"unexpected cpp_stream input file(s): {extra}")

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
            1: "input count/path/pointer validation failed",
            2: "input row width or raw row-byte validation failed",
            3: "input arrays have different row counts",
            4: "group capacity exceeded or dense key fell outside its declared domain",
        }
        base = meanings.get(code, f"native runtime returned error code {code}")
        raise RuntimeError(f"cpp_stream: {base}" + (f": {message}" if message else ""))

    def run_files(
        self,
        data: Mapping[str, str | Path],
        *,
        out_path: str | Path,
        async_writeback_mb: int = 0,
    ) -> RunResult:
        """Run headerless raw files using the compile-time dtype/row-width specs."""
        self._validate_names(data)
        writeback = self._validate_writeback(async_writeback_mb)
        encoded = [str(Path(data[name])).encode() for name in self.input_names]
        argv = (ctypes.c_char_p * len(encoded))(*encoded)
        rows = ctypes.c_size_t()
        seconds = ctypes.c_double()
        output = Path(out_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        lib = self._load()
        code = lib.cpp_stream_run_files(
            argv,
            len(encoded),
            str(output).encode(),
            writeback,
            ctypes.byref(rows),
            ctypes.byref(seconds),
        )
        self._raise_native(code, lib)
        return RunResult(output_path=output, rows=int(rows.value), seconds=float(seconds.value))

    def run_npy_files(
        self,
        data: Mapping[str, str | Path],
        *,
        out_path: str | Path,
        async_writeback_mb: int = 0,
    ) -> RunResult:
        """Mmap C-order .npy payloads and pass typed pointers to native code."""
        self._validate_names(data)
        writeback = self._validate_writeback(async_writeback_mb)
        mapped = [mmap_npy(data[name]) for name in self.input_names]
        try:
            for name, item, expected in zip(self.input_names, mapped, self.input_types):
                actual = item.info.input_type
                if actual != expected:
                    raise TypeError(
                        f"cpp_stream input {name!r} was compiled as "
                        f"dtype={expected.dtype}, row_width={expected.row_width}, but "
                        f"the .npy file has dtype={actual.dtype}, row_width={actual.row_width}"
                    )
            pointers = (ctypes.c_void_p * len(mapped))(
                *(ctypes.c_void_p(item.data_pointer) for item in mapped)
            )
            input_rows = (ctypes.c_size_t * len(mapped))(
                *(item.info.rows for item in mapped)
            )
            input_widths = (ctypes.c_size_t * len(mapped))(
                *(item.info.row_width for item in mapped)
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
                len(mapped),
                str(output).encode(),
                writeback,
                ctypes.byref(rows),
                ctypes.byref(seconds),
            )
            self._raise_native(code, lib)
            return RunResult(
                output_path=output,
                rows=int(rows.value),
                seconds=float(seconds.value),
            )
        finally:
            # Keep every memmap alive until the native call returns, then close
            # its underlying mmap deterministically when NumPy exposes it.
            for item in mapped:
                mapping = getattr(item.array, "_mmap", None)
                if mapping is not None:
                    mapping.close()

    def explain(self) -> str:
        lines = [
            f"cpp_stream N={self.n_instruments}",
            f"inputs={self.input_names}",
            f"input_types={self.input_types}",
            f"scratch_slots={self.plan.scratch_slots}",
        ]
        for i, stage in enumerate(self.plan.stages):
            lines.append(
                f"{i}: {stage.kind} -> "
                f"{'output' if stage.out.slot is None else f'slot {stage.out.slot}'}"
            )
        return "\n".join(lines)
