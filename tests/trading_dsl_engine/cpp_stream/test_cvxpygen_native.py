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


def test_full_ridge_riskmodel_mpo_pipeline_has_one_time_loop(
    tmp_path: Path, monkeypatch
):
    from flows.riskmodel import risk_covariance
    from trading_dsl_engine.base.dsl import (
        Ridge,
        cat,
        ewm,
        get_preds,
        psd_factor,
        shift,
        var,
    )
    from trading_dsl_engine.cpp_stream import compile_formula
    from trading_dsl_engine.cpp_stream.optimizer import bind_program, get_field

    n_assets, n_horizons, rows = 3, 2, 24
    weights = cp.Variable((n_horizons, n_assets), name="weights")
    turnover = cp.Variable((n_horizons, n_assets), name="turnover")
    expected_returns = cp.Parameter(
        (n_horizons, n_assets), name="expected_returns"
    )
    half_spread_bps = cp.Parameter(
        n_assets, nonneg=True, name="half_spread_bps"
    )
    current_weights = cp.Parameter(n_assets, name="current_weights")
    risk_factor = cp.Parameter((n_assets, n_assets), name="risk_factor")
    risk_radius = cp.Parameter(nonneg=True, name="risk_radius")
    previous = cp.vstack([current_weights, weights[:-1]])
    delta = weights - previous
    constraints = [turnover >= delta, turnover >= -delta]
    constraints.extend(
        cp.SOC(risk_radius, risk_factor @ weights[horizon])
        for horizon in range(n_horizons)
    )
    problem = cp.Problem(
        cp.Minimize(
            -cp.sum(cp.multiply(expected_returns, weights))
            + cp.sum(cp.multiply(half_spread_bps * 1e-4, turnover))
        ),
        constraints,
    )
    artifact = generate_clarabel_program(
        problem,
        code_dir=tmp_path / "fused-generated",
        clarabel=_clarabel_native(),
        class_name="GeneratedFusedMpo",
        prefix="fused_mpo_",
        instrument_count=n_assets,
    )

    returns = var("returns")
    lagged = shift(returns, 1, 1)
    fast = ewm(returns, 4, min_periods=2)
    forecasts = cat(
        get_preds(Ridge(lagged, fast, y=returns, hl=4, lambda_=0.1)),
        get_preds(Ridge(lagged, fast, y=returns, hl=16, lambda_=0.1)),
    )
    factor = psd_factor(
        risk_covariance(returns, span=8, min_periods=2),
        eigenvalue_floor=1e-8,
    )
    mpo = bind_program(
        artifact,
        expected_returns=forecasts,
        half_spread_bps=var("half_spread_bps"),
        current_weights=var("current_weights"),
        risk_factor=factor,
        risk_radius=0.08,
    )
    next_weights = get_field(mpo, "weights[0]")
    first_turnover = get_field(mpo, "turnover[0]")
    pnl = shift(next_weights, 1, 1) * returns

    rng = np.random.default_rng(9)
    data = {
        "returns": rng.normal(scale=0.01, size=(rows, n_assets)),
        "half_spread_bps": np.ones((rows, n_assets)),
        "current_weights": np.zeros((rows, n_assets)),
    }
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_STREAM_CACHE", str(tmp_path / "fused-cache")
    )
    runtime = compile_formula([pnl, next_weights, first_turnover], data)
    generated = runtime.generated_cpp.read_text()
    row_loop = "for (std::size_t t = row_begin; t < row_end; ++t)"
    assert generated.count(row_loop) == 1
    assert generated.count("stackdsl::CvxpygenNode<") == 1
    assert "stackdsl::PsdFactorNode<" in generated
    cvxpygen_stages = [
        stage
        for stage in runtime.plan.stages
        if stage.kind in {"cvxpygen", "cvxpygen_bundle"}
    ]
    assert len(cvxpygen_stages) == 1
    assert cvxpygen_stages[0].kind == "cvxpygen_bundle"
    assert len(cvxpygen_stages[0].members) == 2

    result = runtime.run(out_path=tmp_path / "fused.npy")
    pnl_values, weight_values, turnover_values = result.load(mmap_mode=None)
    assert pnl_values.shape == (rows, n_assets)
    assert weight_values.shape == (rows, n_assets)
    assert turnover_values.shape == (rows, n_assets)
    assert np.isfinite(weight_values[-1]).all()
    assert np.isfinite(turnover_values[-1]).all()


def test_cpp_stream_risk_covariance_and_psd_factor_handle_missing_rows(tmp_path: Path):
    from flows.riskmodel import risk_covariance
    from trading_dsl_engine.base.dsl import psd_factor, var
    from trading_dsl_engine.cpp_stream import compile_formula

    rng = np.random.default_rng(123)
    rows, n_assets = 80, 4
    returns = rng.normal(scale=0.01, size=(rows, n_assets))
    returns[rng.random(returns.shape) < 0.08] = np.nan
    returns[rng.random(returns.shape) < 0.04] = 0.0
    covariance = risk_covariance(returns=var("returns"), span=12, min_periods=4)
    factor = psd_factor(covariance, eigenvalue_floor=1e-9)
    runtime = compile_formula([covariance, factor], {"returns": returns})
    result = runtime.run(out_path=tmp_path / "risk.npy")
    covariance_values, factor_values = result.load(mmap_mode=None)

    final_covariance = covariance_values[-1]
    final_factor = factor_values[-1]
    assert np.isfinite(final_covariance).all()
    assert np.isfinite(final_factor).all()
    reconstructed = final_factor @ final_factor.T
    np.testing.assert_allclose(
        reconstructed,
        0.5 * (final_covariance + final_covariance.T),
        rtol=1e-8,
        atol=1e-10,
    )
    assert np.linalg.eigvalsh(reconstructed).min() > 0.0
