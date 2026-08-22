from __future__ import annotations

import ctypes
import json
import os
from pathlib import Path
import subprocess
import textwrap

import cvxpy as cp
import numpy as np
import pytest

from trading_dsl_engine.cpp_stream.optimizer import (
    ClarabelNativePaths,
    generate_clarabel_program,
)
from trading_dsl_engine.cpp_stream.python.compiler_support import build_shared


def _clarabel_native() -> ClarabelNativePaths:
    include = os.environ.get("CLARABEL_INCLUDE_DIR")
    library = os.environ.get("CLARABEL_STATIC_LIBRARY")
    if not include or not library:
        pytest.skip(
            "set CLARABEL_INCLUDE_DIR and CLARABEL_STATIC_LIBRARY for native optimizer tests"
        )
    return ClarabelNativePaths(Path(include), Path(library))


def _mpo_problem(n_assets: int = 3, n_horizons: int = 2) -> cp.Problem:
    weights = cp.Variable((n_horizons, n_assets), name="weights")
    turnover = cp.Variable((n_horizons, n_assets), name="turnover")
    expected_returns = cp.Parameter(
        (n_horizons, n_assets), name="expected_returns"
    )
    half_spread = cp.Parameter(
        (n_horizons, n_assets), nonneg=True, name="half_spread"
    )
    current_weights = cp.Parameter(n_assets, name="current_weights")
    risk_radius = cp.Parameter(n_horizons, nonneg=True, name="risk_radius")
    risk_factor = cp.Parameter((n_assets, n_assets), name="risk_factor")
    previous = cp.vstack([current_weights, weights[:-1]])
    delta = weights - previous
    constraints = [turnover >= delta, turnover >= -delta]
    constraints.extend(
        cp.SOC(risk_radius[horizon], risk_factor @ weights[horizon])
        for horizon in range(n_horizons)
    )
    return cp.Problem(
        cp.Minimize(
            -cp.sum(cp.multiply(expected_returns, weights))
            + cp.sum(cp.multiply(half_spread, turnover))
        ),
        constraints,
    )


def _generate(tmp_path: Path):
    return generate_clarabel_program(
        _mpo_problem(),
        code_dir=tmp_path / "generated",
        clarabel=_clarabel_native(),
        class_name="GeneratedMpo",
        prefix="mpo_",
    )


def _generate_qp(tmp_path: Path):
    x = cp.Variable(3, name="x")
    target = cp.Parameter(3, name="target")
    lower = cp.Parameter(3, name="lower")
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - target)), [x >= lower])
    return generate_clarabel_program(
        problem,
        code_dir=tmp_path / "generated-qp",
        clarabel=_clarabel_native(),
        class_name="GeneratedQp",
        prefix="qp_",
    )


def test_generated_class_is_persistent_and_instance_owned(tmp_path: Path):
    artifact = _generate(tmp_path)
    source = artifact.instance_header.read_text()
    assert "class GeneratedMpo final" in source
    assert "clarabel_DefaultSolver_update_A" in source
    assert "clarabel_DefaultSolver_update_q" in source
    assert "clarabel_DefaultSolver_update_b" in source
    assert "clarabel_DefaultSolver_free" in source
    assert "GeneratedMpo(const GeneratedMpo&) = delete" in source
    assert "inline static cpg_int" in source
    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["instance_owned"] is True
    assert manifest["persistent_solver"] is True
    assert manifest["cvxpygen_version"] == "1.0.0"
    assert [item["name"] for item in manifest["parameters"]] == [
        "expected_returns",
        "half_spread",
        "current_weights",
        "risk_radius",
        "risk_factor",
    ]
    assert manifest["parameters"][-1]["dirty_blocks"] == ["A"]


def test_two_generated_instances_solve_concurrently(tmp_path: Path):
    artifact = _generate(tmp_path)
    driver = tmp_path / "driver.cpp"
    binary = tmp_path / "driver"
    driver.write_text(
        textwrap.dedent(
            """
            #include "cpg_instance.hpp"
            #include <array>
            #include <cmath>
            #include <cstdio>
            #include <thread>

            double run(double scale) {
                GeneratedMpo solver;
                std::array<double, 6> er{};
                std::array<double, 6> hs{};
                std::array<double, 3> current{};
                std::array<double, 2> radius{};
                std::array<double, 9> factor{};
                radius.fill(0.1);
                factor[0] = factor[4] = factor[8] = 1.0;
                for (int iteration = 0; iteration < 20; ++iteration) {
                    for (int i = 0; i < 6; ++i) {
                        er[i] = scale * 1e-4 * (i + 1) * (1.0 + 1e-4 * iteration);
                        hs[i] = 1e-4;
                    }
                    solver.set_expected_returns(er);
                    solver.set_half_spread(hs);
                    solver.set_current_weights(current);
                    solver.set_risk_radius(radius);
                    solver.set_risk_factor(factor);
                    solver.solve();
                }
                return solver.primal_weights()[0];
            }

            int main() {
                double first = 0.0;
                double second = 0.0;
                std::thread a([&] { first = run(1.0); });
                std::thread b([&] { second = run(2.0); });
                a.join();
                b.join();
                std::printf("%.17g %.17g\\n", first, second);
                return !(std::isfinite(first) && std::isfinite(second) && first != second);
            }
            """
        )
    )
    command = [
        os.environ.get("CXX", "g++"),
        "-std=gnu++20",
        "-O3",
        *(f"-I{path}" for path in artifact.include_dirs),
        str(driver),
        str(artifact.clarabel.static_library),
        "-ldl",
        "-lpthread",
        "-lm",
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    values = [float(value) for value in completed.stdout.split()]
    assert len(values) == 2
    assert values[0] != values[1]


def test_repeated_solves_do_not_leak_one_solver_per_call(tmp_path: Path):
    artifact = _generate(tmp_path)
    driver = tmp_path / "rss.cpp"
    binary = tmp_path / "rss"
    driver.write_text(
        textwrap.dedent(
            """
            #include "cpg_instance.hpp"
            #include <array>
            #include <cstdio>

            long rss_kb() {
                FILE* file = std::fopen("/proc/self/status", "r");
                if (!file) return -1;
                char line[256];
                long result = -1;
                while (std::fgets(line, sizeof(line), file)) {
                    if (std::sscanf(line, "VmRSS: %ld kB", &result) == 1) break;
                }
                std::fclose(file);
                return result;
            }

            int main() {
                GeneratedMpo solver;
                std::array<double, 6> er{};
                std::array<double, 6> hs{};
                std::array<double, 3> current{};
                std::array<double, 2> radius{};
                std::array<double, 9> factor{};
                radius.fill(0.1);
                factor[0] = factor[4] = factor[8] = 1.0;
                auto solve = [&](int iteration) {
                    for (int i = 0; i < 6; ++i) {
                        er[i] = 1e-4 * (i + 1) * (1.0 + 1e-5 * iteration);
                        hs[i] = 1e-4;
                    }
                    solver.set_expected_returns(er);
                    solver.set_half_spread(hs);
                    solver.set_current_weights(current);
                    solver.set_risk_radius(radius);
                    solver.set_risk_factor(factor);
                    solver.solve();
                };
                for (int i = 0; i < 5; ++i) solve(i);
                const long before = rss_kb();
                for (int i = 0; i < 100; ++i) solve(i + 5);
                const long after = rss_kb();
                std::printf("%ld\\n", after - before);
                return 0;
            }
            """
        )
    )
    command = [
        os.environ.get("CXX", "g++"),
        "-std=gnu++20",
        "-O3",
        *(f"-I{path}" for path in artifact.include_dirs),
        str(driver),
        str(artifact.clarabel.static_library),
        "-ldl",
        "-lpthread",
        "-lm",
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    completed = subprocess.run(
        [str(binary)], check=True, capture_output=True, text=True
    )
    # Allocator warm-up can retain a small amount, but growth must not scale by
    # one complete solver workspace per invocation.
    assert int(completed.stdout.strip()) < 4096


def test_cpp_stream_native_builder_links_generated_instance(tmp_path: Path, monkeypatch):
    artifact = _generate(tmp_path)
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_STREAM_CACHE", str(tmp_path / "native-cache")
    )
    source = textwrap.dedent(
        """
        #include "cpg_instance.hpp"
        #include <array>

        extern "C" double solve_once(double scale) {
            GeneratedMpo solver;
            std::array<double, 6> er{};
            std::array<double, 6> hs{};
            std::array<double, 3> current{};
            std::array<double, 2> radius{};
            std::array<double, 9> factor{};
            radius.fill(0.1);
            factor[0] = factor[4] = factor[8] = 1.0;
            for (int i = 0; i < 6; ++i) {
                er[i] = scale * 1e-4 * (i + 1);
                hs[i] = 1e-4;
            }
            solver.set_expected_returns(er);
            solver.set_half_spread(hs);
            solver.set_current_weights(current);
            solver.set_risk_radius(radius);
            solver.set_risk_factor(factor);
            solver.solve();
            return solver.primal_weights()[0];
        }
        """
    )
    shared, _ = build_shared(source, **artifact.build_shared_kwargs())
    library = ctypes.CDLL(str(shared))
    library.solve_once.argtypes = [ctypes.c_double]
    library.solve_once.restype = ctypes.c_double
    first = library.solve_once(1.0)
    second = library.solve_once(2.0)
    assert np.isfinite(first)
    assert np.isfinite(second)
    assert first != second


def test_generated_instance_supports_quadratic_objective(tmp_path: Path, monkeypatch):
    artifact = _generate_qp(tmp_path)
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_STREAM_CACHE", str(tmp_path / "qp-native-cache")
    )
    source = textwrap.dedent(
        """
        #include "cpg_instance.hpp"
        #include <array>

        extern "C" double solve_qp(double first_target) {
            GeneratedQp solver;
            std::array<double, 3> target{first_target, 2.0, 3.0};
            std::array<double, 3> lower{0.0, 0.0, 0.0};
            solver.set_target(target);
            solver.set_lower(lower);
            solver.solve();
            return solver.primal_x()[0];
        }
        """
    )
    shared, _ = build_shared(source, **artifact.build_shared_kwargs())
    library = ctypes.CDLL(str(shared))
    library.solve_qp.argtypes = [ctypes.c_double]
    library.solve_qp.restype = ctypes.c_double
    assert library.solve_qp(1.5) == pytest.approx(1.5, abs=1e-7)
    assert library.solve_qp(-1.5) == pytest.approx(0.0, abs=1e-7)
