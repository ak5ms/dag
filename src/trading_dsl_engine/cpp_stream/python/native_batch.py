from __future__ import annotations

from collections.abc import Mapping, Sequence
import ctypes
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time

from trading_dsl_engine.cpp_stream.python.compiler_support import build_shared
from trading_dsl_engine.cpp_stream.python.runtime import (
    _PARALLEL_MODES,
    CppStreamRuntime,
    RunResult,
)
from trading_dsl_engine.cpp_stream.python.sources import (
    PreparedSource,
    SourceValue,
    open_source_mapping,
)


_DISPATCH_SOURCE = r'''// Generic native task scheduler for independent cpp_stream runners.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <exception>
#include <string>
#include <thread>
#include <vector>

#include "stackdsl/runtime.hpp"

namespace {
thread_local std::string g_last_error;

using RunFunction = int (*)(
    const void* const*,
    const std::size_t*,
    const std::size_t*,
    std::size_t,
    const char*,
    std::size_t,
    std::size_t,
    std::size_t,
    int,
    bool,
    std::size_t*,
    double*,
    double*,
    std::size_t*,
    std::size_t*);

struct NativeTask {
    RunFunction function;
    const void* const* input_data;
    const std::size_t* input_rows;
    const std::size_t* input_widths;
    std::size_t input_count;
    const char* output_path;
    std::size_t output_offset_bytes;
    std::size_t async_writeback_bytes;
    std::size_t requested_threads;
    int parallel_mode;
    std::uint8_t pin_threads;

    std::size_t rows_out;
    double seconds_out;
    double cpu_seconds_out;
    std::size_t threads_out;
    std::size_t available_cpus_out;
    int code;
};

void run_task(NativeTask& task) noexcept {
    task.code = task.function(
        task.input_data,
        task.input_rows,
        task.input_widths,
        task.input_count,
        task.output_path,
        task.output_offset_bytes,
        task.async_writeback_bytes,
        task.requested_threads,
        task.parallel_mode,
        task.pin_threads != 0,
        &task.rows_out,
        &task.seconds_out,
        &task.cpu_seconds_out,
        &task.threads_out,
        &task.available_cpus_out);
}

void attribute_batch_cpu(
    NativeTask* tasks,
    std::size_t task_count,
    double batch_cpu_seconds) noexcept {
    // Each generated runner measures process CPU time. Those clocks overlap
    // when runners execute concurrently, so summing them would double count.
    // Preserve an additive per-task field by attributing the one aggregate
    // process measurement in proportion to each task's native wall time.
    double weight_sum = 0.0;
    for (std::size_t index = 0; index < task_count; ++index) {
        if (tasks[index].seconds_out > 0.0) {
            weight_sum += tasks[index].seconds_out;
        }
    }
    if (weight_sum > 0.0) {
        for (std::size_t index = 0; index < task_count; ++index) {
            const double weight = std::max(0.0, tasks[index].seconds_out);
            tasks[index].cpu_seconds_out =
                batch_cpu_seconds * weight / weight_sum;
        }
        return;
    }
    for (std::size_t index = 0; index < task_count; ++index) {
        tasks[index].cpu_seconds_out = index == 0 ? batch_cpu_seconds : 0.0;
    }
}
}  // namespace

extern "C" const char* cpp_stream_batch_last_error() noexcept {
    return g_last_error.c_str();
}

extern "C" int cpp_stream_run_many(
    NativeTask* tasks,
    std::size_t task_count,
    std::size_t requested_workers,
    bool pin_workers,
    std::size_t* workers_out) noexcept {
    try {
        g_last_error.clear();
        if (tasks == nullptr || task_count == 0) return 1;
        const std::size_t available = stackdsl::available_cpu_count();
        const std::size_t requested = requested_workers == 0
            ? available
            : requested_workers;
        const std::size_t workers = std::max<std::size_t>(
            1,
            std::min(task_count, std::min(requested, available)));
        if (workers_out != nullptr) *workers_out = workers;

        const std::clock_t batch_cpu_started = std::clock();
        if (workers == 1) {
            for (std::size_t index = 0; index < task_count; ++index) {
                run_task(tasks[index]);
            }
        } else {
            std::atomic<std::size_t> next{0};
            std::vector<std::thread> pool;
            pool.reserve(workers);
            for (std::size_t worker = 0; worker < workers; ++worker) {
                pool.emplace_back([&, worker] {
                    if (pin_workers) stackdsl::pin_current_thread(worker);
                    while (true) {
                        const std::size_t index = next.fetch_add(
                            1, std::memory_order_relaxed);
                        if (index >= task_count) break;
                        run_task(tasks[index]);
                    }
                });
            }
            for (auto& thread : pool) thread.join();
        }
        const std::clock_t batch_cpu_finished = std::clock();
        const double batch_cpu_seconds = static_cast<double>(
            batch_cpu_finished - batch_cpu_started) / CLOCKS_PER_SEC;
        attribute_batch_cpu(tasks, task_count, batch_cpu_seconds);
        return 0;
    } catch (const std::exception& exc) {
        g_last_error = exc.what();
        return 100;
    } catch (...) {
        g_last_error = "unknown native batch scheduler exception";
        return 101;
    }
}
'''


class _NativeTask(ctypes.Structure):
    _fields_ = [
        ("function", ctypes.c_void_p),
        ("input_data", ctypes.POINTER(ctypes.c_void_p)),
        ("input_rows", ctypes.POINTER(ctypes.c_size_t)),
        ("input_widths", ctypes.POINTER(ctypes.c_size_t)),
        ("input_count", ctypes.c_size_t),
        ("output_path", ctypes.c_char_p),
        ("output_offset_bytes", ctypes.c_size_t),
        ("async_writeback_bytes", ctypes.c_size_t),
        ("requested_threads", ctypes.c_size_t),
        ("parallel_mode", ctypes.c_int),
        ("pin_threads", ctypes.c_uint8),
        ("rows_out", ctypes.c_size_t),
        ("seconds_out", ctypes.c_double),
        ("cpu_seconds_out", ctypes.c_double),
        ("threads_out", ctypes.c_size_t),
        ("available_cpus_out", ctypes.c_size_t),
        ("code", ctypes.c_int),
    ]


@dataclass(frozen=True, slots=True)
class NativeBatchResult:
    """Results from one native task-pool invocation."""

    results: tuple[RunResult, ...]
    wall_seconds: float
    workers: int

    @property
    def native_seconds_sum(self) -> float:
        return float(sum(result.seconds for result in self.results))

    @property
    def effective_concurrency(self) -> float:
        if self.wall_seconds <= 0.0:
            return 0.0
        return self.native_seconds_sum / self.wall_seconds


@dataclass(slots=True)
class _PreparedTask:
    runtime: CppStreamRuntime
    prepared_sources: list[PreparedSource]
    pointers: ctypes.Array
    input_rows: ctypes.Array
    input_widths: ctypes.Array
    output_path: Path
    output_path_bytes: bytes
    output_offset: int
    logical_shape: tuple[int, ...]
    processed_input_rows: int
    task: _NativeTask

    def close(self) -> None:
        for item in reversed(self.prepared_sources):
            item.close()

    def result(self, task: _NativeTask) -> RunResult:
        processed_rows = int(task.rows_out)
        return RunResult(
            output_path=self.output_path,
            rows=processed_rows,
            seconds=float(task.seconds_out),
            output_rows=(
                1
                if self.runtime.output_layout.mode == "final"
                else processed_rows
            ),
            output_shape=self.logical_shape,
            output_mode=self.runtime.output_layout.mode,
            cpu_seconds=float(task.cpu_seconds_out),
            threads=int(task.threads_out),
            available_cpus=int(task.available_cpus_out),
            parallel_mode=self.runtime.parallel_plan.mode,
            formula_outputs=self.runtime.output_layout.outputs,
            row_output_width=self.runtime.output_layout.row_width,
            final_output_width=self.runtime.output_layout.final_width,
            return_multiple=self.runtime.return_multiple,
            output_dtype="float64",
            data_offset=self.output_offset,
        )


@lru_cache(maxsize=1)
def _load_dispatcher() -> ctypes.CDLL:
    library_path, _ = build_shared(_DISPATCH_SOURCE)
    library = ctypes.CDLL(str(library_path))
    library.cpp_stream_run_many.argtypes = [
        ctypes.POINTER(_NativeTask),
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_size_t),
    ]
    library.cpp_stream_run_many.restype = ctypes.c_int
    library.cpp_stream_batch_last_error.argtypes = []
    library.cpp_stream_batch_last_error.restype = ctypes.c_char_p
    return library


def _logical_output_shape(runtime: CppStreamRuntime, rows: int) -> tuple[int, ...]:
    if len(runtime.output_layout.outputs) == 1:
        public = runtime.output_layout.outputs[0]
        return public.shape if public.mode == "final" else (rows,) + public.shape
    return (runtime.output_layout.storage_size(rows),)


def _prepare_task(
    runtime: CppStreamRuntime,
    data: Mapping[str, SourceValue] | None,
    *,
    out_path: str | Path | None,
    async_writeback_mb: int,
    threads: int,
    pin_threads: bool,
) -> _PreparedTask:
    selected = runtime.bound_sources if data is None else data
    if selected is None:
        raise ValueError(
            "no sources are bound; pass data to run_many(...) or compile each "
            "runtime with a source mapping"
        )
    runtime._validate_names(selected)
    writeback = runtime._validate_writeback(async_writeback_mb)
    requested_threads = runtime._resolved_request(threads)
    prepared = open_source_mapping(
        selected,
        runtime.input_names,
        runtime.input_types,
    )
    try:
        row_counts = {item.info.rows for item in prepared}
        if len(row_counts) != 1:
            details = {
                name: item.info.rows
                for name, item in zip(runtime.input_names, prepared)
            }
            raise ValueError(
                f"cpp_stream sources have different row counts: {details}"
            )
        rows = int(next(iter(row_counts)))
        logical_shape = _logical_output_shape(runtime, rows)
        output_path = (
            runtime._default_output_path()
            if out_path is None
            else Path(out_path)
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_offset = runtime._prepare_output(output_path, logical_shape)

        pointers = (ctypes.c_void_p * len(prepared))(
            *(ctypes.c_void_p(item.data_pointer) for item in prepared)
        )
        input_rows = (ctypes.c_size_t * len(prepared))(
            *(item.info.rows for item in prepared)
        )
        input_widths = (ctypes.c_size_t * len(prepared))(
            *(item.info.input_type.row_width for item in prepared)
        )
        output_path_bytes = str(output_path).encode()
        library = runtime._load()
        function_address = ctypes.cast(
            library.cpp_stream_run_arrays,
            ctypes.c_void_p,
        ).value
        if function_address is None:
            raise RuntimeError("could not resolve cpp_stream_run_arrays")

        task = _NativeTask(
            ctypes.c_void_p(function_address),
            pointers,
            input_rows,
            input_widths,
            len(prepared),
            output_path_bytes,
            output_offset,
            writeback,
            requested_threads,
            _PARALLEL_MODES[runtime.parallel_plan.mode],
            int(bool(pin_threads)),
            0,
            0.0,
            0.0,
            0,
            0,
            0,
        )
        return _PreparedTask(
            runtime=runtime,
            prepared_sources=prepared,
            pointers=pointers,
            input_rows=input_rows,
            input_widths=input_widths,
            output_path=output_path,
            output_path_bytes=output_path_bytes,
            output_offset=output_offset,
            logical_shape=logical_shape,
            processed_input_rows=rows,
            task=task,
        )
    except Exception:
        for item in reversed(prepared):
            item.close()
        raise


def run_many(
    runtimes: Sequence[CppStreamRuntime],
    data: Mapping[str, SourceValue] | None = None,
    *,
    out_paths: Sequence[str | Path | None] | None = None,
    workers: int = 0,
    threads_per_runtime: int = 1,
    async_writeback_mb: int = 0,
    pin_workers: bool = False,
    pin_runtime_threads: bool = False,
) -> NativeBatchResult:
    """Execute independent compiled runtimes on one native C++ task pool.

    Each runtime remains an ordinary generated cpp_stream DAG.  This helper is
    intended for collections of independent final-reduction DAGs, such as GP
    candidate microbatches, that cannot safely shard one temporal accumulator.
    Python prepares the input/output descriptors once; all scheduling and hot
    execution occur in C++ without Python worker threads.
    """

    runtimes = tuple(runtimes)
    if not runtimes:
        raise ValueError("run_many requires at least one runtime")
    if workers < 0:
        raise ValueError("workers must be >= 0; zero selects available CPUs")
    if threads_per_runtime < 0:
        raise ValueError("threads_per_runtime must be >= 0")
    if out_paths is None:
        resolved_paths: tuple[str | Path | None, ...] = (None,) * len(runtimes)
    else:
        resolved_paths = tuple(out_paths)
        if len(resolved_paths) != len(runtimes):
            raise ValueError(
                "out_paths length must match runtimes: "
                f"{len(resolved_paths)} != {len(runtimes)}"
            )
        explicit_paths = [
            Path(path).expanduser().resolve()
            for path in resolved_paths
            if path is not None
        ]
        if len(set(explicit_paths)) != len(explicit_paths):
            raise ValueError(
                "out_paths must be distinct when runtimes may execute concurrently"
            )

    # Avoid nested oversubscription.  A caller may still request a multithreaded
    # child DAG when there is only one outer task.
    child_threads = int(threads_per_runtime)
    if len(runtimes) > 1 and (workers == 0 or workers > 1):
        child_threads = 1

    prepared: list[_PreparedTask] = []
    try:
        for runtime, path in zip(runtimes, resolved_paths):
            prepared.append(
                _prepare_task(
                    runtime,
                    data,
                    out_path=path,
                    async_writeback_mb=async_writeback_mb,
                    threads=child_threads,
                    pin_threads=pin_runtime_threads,
                )
            )

        tasks = (_NativeTask * len(prepared))(
            *(item.task for item in prepared)
        )
        dispatcher = _load_dispatcher()
        actual_workers = ctypes.c_size_t()
        started = time.perf_counter()
        code = dispatcher.cpp_stream_run_many(
            tasks,
            len(prepared),
            int(workers),
            bool(pin_workers),
            ctypes.byref(actual_workers),
        )
        wall_seconds = time.perf_counter() - started
        if code != 0:
            detail = dispatcher.cpp_stream_batch_last_error()
            message = detail.decode() if detail else ""
            base = {
                1: "native batch task validation failed",
                100: "native batch scheduler raised an exception",
                101: "native batch scheduler raised an unknown exception",
            }.get(code, f"native batch scheduler returned error code {code}")
            raise RuntimeError(base + (f": {message}" if message else ""))

        results: list[RunResult] = []
        for index, item in enumerate(prepared):
            task = tasks[index]
            if task.code != 0:
                # The generated runner's diagnostic string is thread-local to
                # the native worker, so the stable code-to-meaning mapping is the
                # useful error detail once control returns to Python.
                item.runtime._raise_native(int(task.code), item.runtime._load())
            results.append(item.result(task))
        return NativeBatchResult(
            results=tuple(results),
            wall_seconds=float(wall_seconds),
            workers=int(actual_workers.value),
        )
    finally:
        for item in reversed(prepared):
            item.close()


__all__ = ["NativeBatchResult", "run_many"]
