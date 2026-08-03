from __future__ import annotations

import os
from pathlib import Path
from statistics import median
import subprocess
import tempfile

import includeigen


ROWS = int(os.environ.get("CPP_STREAM_EIGEN_ROWS", "5000000"))
N = int(os.environ.get("CPP_STREAM_EIGEN_INSTRUMENTS", "9"))
RUNS = int(os.environ.get("CPP_STREAM_EIGEN_RUNS", "10"))
WARMUPS = int(os.environ.get("CPP_STREAM_EIGEN_WARMUPS", "1"))
AUDIT_ROWS = int(os.environ.get("CPP_STREAM_EIGEN_AUDIT_ROWS", "10000"))


SOURCE = r'''
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

#include "stackdsl/ops/eigen_solvers.hpp"

namespace {
constexpr std::size_t K = 3;
constexpr std::size_t N = __N__;

bool custom_cholesky(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    std::array<double, K>& solution
) noexcept {
    std::array<double, K * K> lower{};
    for (std::size_t i = 0; i < K; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = system[i * K + j];
            for (std::size_t k = 0; k < j; ++k) {
                value = std::fma(
                    -lower[i * K + k],
                    lower[j * K + k],
                    value
                );
            }
            if (i == j) {
                if (!(value > 1e-15)) return false;
                lower[i * K + j] = std::sqrt(value);
            } else {
                lower[i * K + j] = value / lower[j * K + j];
            }
        }
    }
    std::array<double, K> intermediate{};
    for (std::size_t i = 0; i < K; ++i) {
        double value = rhs[i];
        for (std::size_t j = 0; j < i; ++j) {
            value = std::fma(-lower[i * K + j], intermediate[j], value);
        }
        intermediate[i] = value / lower[i * K + i];
    }
    for (std::size_t reverse = 0; reverse < K; ++reverse) {
        const std::size_t i = K - 1 - reverse;
        double value = intermediate[i];
        for (std::size_t j = i + 1; j < K; ++j) {
            value = std::fma(-lower[j * K + i], solution[j], value);
        }
        solution[i] = value / lower[i * K + i];
    }
    return true;
}

void build_row(
    std::size_t row,
    std::array<double, K * K>& system,
    std::array<double, K>& rhs
) noexcept {
    system.fill(0.0);
    rhs.fill(0.0);
    for (std::size_t lane = 0; lane < N; ++lane) {
        const double base = static_cast<double>((row * 17 + lane * 13) % 257) / 257.0;
        const std::array<double, K> x{
            0.5 + base,
            0.25 + static_cast<double>((row + lane * 3) % 127) / 127.0,
            0.75 + static_cast<double>((row * 5 + lane) % 193) / 193.0,
        };
        const double y = 0.2 + 0.7 * x[0] - 0.3 * x[1] + 0.5 * x[2];
        for (std::size_t j = 0; j < K; ++j) {
            rhs[j] = std::fma(x[j], y, rhs[j]);
            for (std::size_t k = 0; k < K; ++k) {
                system[j * K + k] = std::fma(
                    x[j], x[k], system[j * K + k]
                );
            }
        }
    }
    for (std::size_t j = 0; j < K; ++j) {
        system[j * K + j] *= 1.1;
    }
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const std::string mode = argv[1];
    const std::size_t rows = static_cast<std::size_t>(std::strtoull(argv[2], nullptr, 10));
    double checksum = 0.0;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t row = 0; row < rows; ++row) {
        std::array<double, K * K> system{};
        std::array<double, K> rhs{};
        std::array<double, K> solution{};
        build_row(row, system, rhs);
        bool solved = false;
        if (mode == "custom") {
            solved = custom_cholesky(system, rhs, solution);
        } else if (mode == "eigen") {
            solved = stackdsl::eigen_detail::solve_unconstrained<K>(
                system, rhs, solution
            );
        } else {
            return 3;
        }
        if (!solved) return 4;
        checksum += solution[0] + 2.0 * solution[1] + 3.0 * solution[2];
    }
    const auto ended = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(ended - started).count();
    std::cout << std::setprecision(17)
              << "seconds=" << seconds << "\n"
              << "checksum=" << checksum << "\n";
    return 0;
}
}
'''.replace("__N__", str(N))


def compile_binary(root: Path, *, audit: bool) -> Path:
    source = root / ("audit.cpp" if audit else "benchmark.cpp")
    binary = root / ("audit" if audit else "benchmark")
    source.write_text(SOURCE)
    cpp_root = Path(__file__).resolve().parents[1] / "src" / "trading_dsl_engine" / "cpp_stream" / "cpp"
    flags = [
        "g++",
        "-std=c++20",
        "-O3",
        "-march=native",
        "-mtune=native",
        "-DEIGEN_DONT_PARALLELIZE",
        "-DEIGEN_MPL2_ONLY",
        f"-I{cpp_root}",
        f"-I{Path(includeigen.get_include()).resolve()}",
        str(source),
        "-o",
        str(binary),
    ]
    if audit:
        flags[1:1] = [
            "-DEIGEN_RUNTIME_NO_MALLOC",
            "-DSTACKDSL_EIGEN_RUNTIME_NO_MALLOC",
        ]
    else:
        flags[1:1] = ["-DNDEBUG", "-DEIGEN_NO_DEBUG", "-flto"]
    subprocess.run(flags, check=True)
    return binary


def execute(binary: Path, mode: str, rows: int) -> tuple[float, float]:
    completed = subprocess.run(
        [str(binary), mode, str(rows)],
        check=True,
        capture_output=True,
        text=True,
    )
    values = dict(
        line.split("=", 1)
        for line in completed.stdout.splitlines()
        if "=" in line
    )
    return float(values["seconds"]), float(values["checksum"])


def main() -> None:
    print(
        f"rows={ROWS:,} instruments={N} warmups={WARMUPS} runs={RUNS} "
        "system=3x3 SPD solve per row"
    )
    print(
        "eigen_flags=-O3 -DNDEBUG -DEIGEN_NO_DEBUG "
        "-DEIGEN_DONT_PARALLELIZE -march=native -flto"
    )
    print(
        "allocation_audit=-DEIGEN_RUNTIME_NO_MALLOC "
        "-DSTACKDSL_EIGEN_RUNTIME_NO_MALLOC (without EIGEN_NO_DEBUG)"
    )

    with tempfile.TemporaryDirectory(prefix="cpp_stream_eigen_") as temporary:
        root = Path(temporary)
        benchmark = compile_binary(root, audit=False)
        audit = compile_binary(root, audit=True)

        # The allocation-audit binary aborts inside Eigen if a dynamic allocation
        # occurs while the fixed-size solver is active.
        audit_seconds, audit_checksum = execute(audit, "eigen", AUDIT_ROWS)
        print(
            f"allocation_audit_rows={AUDIT_ROWS:,} status=passed "
            f"seconds={audit_seconds:.6f} checksum={audit_checksum:.12g}"
        )

        for _ in range(WARMUPS):
            execute(benchmark, "custom", ROWS)
            execute(benchmark, "eigen", ROWS)

        timings = {"custom": [], "eigen": []}
        checksums = {"custom": [], "eigen": []}
        for repetition in range(RUNS):
            order = ("custom", "eigen") if repetition % 2 == 0 else ("eigen", "custom")
            for mode in order:
                seconds, checksum = execute(benchmark, mode, ROWS)
                timings[mode].append(seconds)
                checksums[mode].append(checksum)

        custom_checksum = median(checksums["custom"])
        eigen_checksum = median(checksums["eigen"])
        tolerance = 1e-10 * max(1.0, abs(custom_checksum))
        if abs(custom_checksum - eigen_checksum) > tolerance:
            raise RuntimeError(
                f"solver checksum mismatch: custom={custom_checksum} eigen={eigen_checksum}"
            )

        custom = median(timings["custom"])
        eigen = median(timings["eigen"])
        print("custom_runs=" + ", ".join(f"{value:.6f}" for value in timings["custom"]))
        print("eigen_runs=" + ", ".join(f"{value:.6f}" for value in timings["eigen"]))
        print(f"custom_median_seconds={custom:.6f}")
        print(f"eigen_median_seconds={eigen:.6f}")
        print(f"custom_rows_per_second={ROWS / custom:.6f}")
        print(f"eigen_rows_per_second={ROWS / eigen:.6f}")
        print(f"eigen_vs_custom_speedup={custom / eigen:.6f}x")
        print(f"checksum={custom_checksum:.12g}")


if __name__ == "__main__":
    main()
