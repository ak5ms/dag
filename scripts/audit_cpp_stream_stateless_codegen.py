from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import includeigen

from trading_dsl_engine.cpp_stream.python.codegen import render_translation_unit
from trading_dsl_engine.cpp_stream.python.lowering import lower_program
from trading_dsl_engine.cpp_stream.python.npy import InputTypeSpec
from trading_dsl_engine.ir import compile_ir


FORMULA = os.environ.get(
    "CPP_STREAM_STATELESS_AUDIT_FORMULA",
    "xs_rank((x + 5 + y) * 3)",
)
N_INSTRUMENTS = int(
    os.environ.get("CPP_STREAM_STATELESS_AUDIT_INSTRUMENTS", "9")
)


def _stage_types(source: str) -> list[tuple[int, str]]:
    result = [
        (int(match.group(2)), match.group(1))
        for match in re.finditer(
            r"^    (stackdsl::.+) s(\d+)\{\};$",
            source,
            re.MULTILINE,
        )
    ]
    result.sort()
    return result


def _row_loop(source: str, stage_count: int) -> str:
    start = source.index("    for (std::size_t t = 0; t < rows; ++t) {")
    last_call = source.index(
        f"        s{stage_count - 1}.on_data(ctx);",
        start,
    )
    end = source.index("\n    }\n", last_call) + len("\n    }\n")
    return source[start:end]


def _kernel_source(
    stage_types: list[tuple[int, str]],
    *,
    scratch_slots: int,
    matrix_scratch_slots: int,
    matrix_scratch_width: int,
) -> str:
    lines = [
        '#include "stackdsl/runtime.hpp"',
        "struct AuditKernel {",
    ]
    for index, cpp_type in stage_types:
        lines.append(f"    {cpp_type} s{index}{{}};")
    lines.append("    void setup() noexcept {")
    for index, _ in stage_types:
        lines.append(f"        s{index}.setup();")
    lines.extend(
        [
            "    }",
            "    __attribute__((noinline)) void run(",
            "        const double* x,",
            "        const double* y,",
            "        double* out",
            "    ) noexcept {",
            "        stackdsl::RowContext<",
            f"            {N_INSTRUMENTS}, 2, {scratch_slots},",
            f"            {matrix_scratch_slots}, {matrix_scratch_width}",
            "        > ctx{};",
            "        ctx.inputs[0] = x;",
            "        ctx.inputs[1] = y;",
            "        ctx.output = out;",
        ]
    )
    for index, _ in stage_types:
        lines.append(f"        s{index}.on_data(ctx);")
    lines.extend(
        [
            "    }",
            "};",
            "extern \"C\" __attribute__((noinline)) void stateless_kernel(",
            "    AuditKernel* kernel,",
            "    const double* x,",
            "    const double* y,",
            "    double* out",
            ") noexcept {",
            "    kernel->run(x, y, out);",
            "}",
        ]
    )
    return "\n".join(lines) + "\n"


def _function_assembly(assembly: str) -> str:
    labels = list(
        re.finditer(
            r"(?m)^([.$A-Za-z_][.$A-Za-z0-9_]*AuditKernel3run[.$A-Za-z0-9_]*):$",
            assembly,
        )
    )
    if not labels:
        raise AssertionError("could not find AuditKernel::run in generated assembly")
    start = labels[0].start()
    size = re.search(r"(?m)^\s*\.size\s+[^,]+,", assembly[start:])
    if size is None:
        raise AssertionError("could not find AuditKernel::run assembly terminator")
    return assembly[start : start + size.end()]


def main() -> None:
    if N_INSTRUMENTS <= 0:
        raise ValueError("CPP_STREAM_STATELESS_AUDIT_INSTRUMENTS must be positive")
    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if compiler is None:
        raise RuntimeError("stateless codegen audit requires a C++ compiler")

    program = compile_ir(FORMULA)
    if program.input_names != ("x", "y"):
        raise AssertionError(
            f"audit formula must consume x then y, got {program.input_names!r}"
        )
    input_types = (
        InputTypeSpec("float64", N_INSTRUMENTS),
        InputTypeSpec("float64", N_INSTRUMENTS),
    )
    plan = lower_program(
        program,
        n_instruments=N_INSTRUMENTS,
        input_dtypes=tuple(spec.dtype for spec in input_types),
    )
    generated = render_translation_unit(
        plan,
        n_instruments=N_INSTRUMENTS,
        prefetch_rows=16,
        input_types=input_types,
    ).text
    stages = _stage_types(generated)
    if len(stages) != 4:
        raise AssertionError(f"expected three arithmetic stages plus rank, got {stages!r}")

    row_loop = _row_loop(generated, len(stages))
    stage_calls = re.findall(
        r"^        s(\d+)\.on_data\(ctx\);$",
        row_loop,
        re.MULTILINE,
    )
    expected_calls = [str(index) for index in range(len(stages))]
    if stage_calls != expected_calls:
        raise AssertionError(
            f"stateless stages are not adjacent: {stage_calls!r} != {expected_calls!r}"
        )

    cpp_root = Path(__file__).resolve().parents[1] / (
        "src/trading_dsl_engine/cpp_stream/cpp"
    )
    eigen_include = Path(includeigen.get_include()).resolve()
    with tempfile.TemporaryDirectory(prefix="cpp-stream-stateless-audit-") as tmp:
        tmp_path = Path(tmp)
        kernel_path = tmp_path / "kernel.cpp"
        assembly_path = tmp_path / "kernel.s"
        vector_report_path = tmp_path / "vector.txt"
        kernel_path.write_text(
            _kernel_source(
                stages,
                scratch_slots=plan.scratch_slots,
                matrix_scratch_slots=plan.matrix_scratch_slots,
                matrix_scratch_width=plan.matrix_scratch_width,
            )
        )
        command = [
            compiler,
            "-std=c++20",
            "-O3",
            "-DNDEBUG",
            "-fno-math-errno",
            "-funroll-loops",
            "-march=native",
            "-mtune=native",
            "-masm=intel",
            "-S",
            f"-I{cpp_root}",
            f"-I{eigen_include}",
            f"-fopt-info-vec-optimized={vector_report_path}",
            str(kernel_path),
            "-o",
            str(assembly_path),
        ]
        subprocess.run(command, check=True)
        assembly = _function_assembly(assembly_path.read_text())
        vector_report = (
            vector_report_path.read_text()
            if vector_report_path.exists()
            else ""
        )

    call_count = len(re.findall(r"(?m)^\s*call\s", assembly))
    if call_count != 0:
        raise AssertionError(
            "stateless hot kernel contains out-of-line calls:\n" + assembly
        )

    packed_add = re.search(r"\bv(?:f?m)?add[^\s]*pd\b|\bvaddpd\b", assembly)
    packed_mul = re.search(r"\bvmul[^\s]*pd\b|\bvfm(?:add|sub)[^\s]*pd\b", assembly)
    scalar_add = re.search(r"\bvaddsd\b|\baddsd\b", assembly)
    scalar_mul = re.search(r"\bvmulsd\b|\bmulsd\b", assembly)
    add_match = packed_add or scalar_add
    mul_match = packed_mul or scalar_mul
    if add_match is None or mul_match is None:
        raise AssertionError(
            "optimized kernel is missing expected floating add/multiply instructions:\n"
            + assembly
        )
    if abs(add_match.start() - mul_match.start()) > 4000:
        raise AssertionError(
            "arithmetic instructions are unexpectedly far apart in the hot kernel"
        )

    print(f"formula: {FORMULA}")
    print(f"instruments: {N_INSTRUMENTS}")
    print(f"scratch slots in logical plan: {plan.scratch_slots}")
    print(f"adjacent generated stage calls: {stage_calls}")
    print(f"out-of-line calls in optimized row kernel: {call_count}")
    print("vectorization report:")
    print(vector_report.strip() or "<compiler emitted no vectorization remarks>")
    print("optimized AuditKernel::run assembly:")
    print(assembly)


if __name__ == "__main__":
    main()
