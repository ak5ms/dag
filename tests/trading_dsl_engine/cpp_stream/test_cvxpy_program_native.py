from __future__ import annotations

import ctypes
from importlib.util import find_spec
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
    cvxpy_program,
    get_field,
    previous_solution,
)
from trading_dsl_engine.cpp_stream.optimizer.direct_clarabel import (
    generate_clarabel_artifact,
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


def test_optimizer_public_surface_has_no_legacy_codegen_path():
    from trading_dsl_engine.cpp_stream import optimizer

    assert set(optimizer.__all__) == {
        "ClarabelNativePaths",
        "build_current_clarabel",
        "cvxpy_program",
        "get_field",
        "previous_solution",
    }
    assert find_spec(
        "trading_dsl_engine.cpp_stream.optimizer.cvxpygen_native"
    ) is None


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
    return generate_clarabel_artifact(
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
    return generate_clarabel_artifact(
        problem,
        code_dir=tmp_path / "generated-qp",
        clarabel=_clarabel_native(),
        class_name="GeneratedQp",
        prefix="qp_",
    )


def test_sharded_maps_reproduce_cvxpy_canonical_data():
    from scipy import sparse

    from trading_dsl_engine.cpp_stream.optimizer.direct_clarabel import (
        compile_sharded_canonical_program,
    )

    problem = _mpo_problem()
    rng = np.random.default_rng(128)
    for parameter in problem.parameters():
        value = (
            rng.uniform(0.01, 1.0, size=parameter.shape or ())
            if parameter.is_nonneg()
            else rng.normal(size=parameter.shape or ())
        )
        parameter.value = value
    compiled = compile_sharded_canonical_program(
        problem,
        parameter_shard_size=4,
    )
    parameter_vector = np.empty(
        sum(parameter.size for parameter in problem.parameters()) + 1
    )
    for parameter, offset in zip(
        problem.parameters(), compiled.parameter_offsets
    ):
        parameter_vector[offset : offset + parameter.size] = np.asarray(
            parameter.value
        ).ravel(order="F")
    parameter_vector[-1] = 1.0
    canonical = {
        name: np.asarray(
            sparse.csr_matrix(
                (mapping.values, mapping.columns, mapping.row_ptr),
                shape=(mapping.rows, parameter_vector.size),
            )
            @ parameter_vector
        ).ravel()
        for name, mapping in compiled.parameter_maps.items()
    }
    actual_A = sparse.csc_matrix(
        (
            canonical["A"],
            compiled.A.row_indices,
            compiled.A.column_ptr,
        ),
        shape=(compiled.A.rows, compiled.A.columns),
    )
    expected, _, _ = problem.get_problem_data(
        cp.CLARABEL,
        enforce_dpp=True,
        canon_backend="COO",
    )
    np.testing.assert_array_equal(canonical["q"], expected["c"])
    np.testing.assert_array_equal(canonical["b"], expected["b"])
    difference = actual_A - expected["A"]
    assert difference.nnz == 0


def test_sharded_maps_match_cvxpy_for_mixed_cones():
    from scipy import sparse

    from trading_dsl_engine.cpp_stream.optimizer.direct_clarabel import (
        compile_sharded_canonical_program,
    )

    x = cp.Variable(5, name="x")
    objective_parameter = cp.Parameter(5, name="objective_parameter")
    psd_expression = cp.bmat(
        [[x[0] + 2.0, x[1]], [x[1], x[2] + 2.0]]
    )
    problem = cp.Problem(
        cp.Minimize(cp.sum_squares(x) + objective_parameter @ x + 2.0),
        [
            x[4] == 0.0,
            x[3] >= -2.0,
            cp.SOC(x[0] + 3.0, x[1:3]),
            psd_expression >> 0,
            cp.ExpCone(x[0], x[3] + 3.0, x[4] + 4.0),
            cp.PowCone3D(x[0] + 3.0, x[1] + 3.0, x[2], 0.4),
        ],
    )
    objective_parameter.value = np.array([0.1, -0.2, 0.3, 0.4, -0.5])
    compiled = compile_sharded_canonical_program(
        problem,
        parameter_shard_size=2,
    )
    parameter_vector = np.concatenate(
        [objective_parameter.value.ravel(order="F"), np.ones(1)]
    )

    def evaluate_map(name: str) -> np.ndarray:
        mapping = compiled.parameter_maps[name]
        return np.asarray(
            sparse.csr_matrix(
                (mapping.values, mapping.columns, mapping.row_ptr),
                shape=(mapping.rows, parameter_vector.size),
            )
            @ parameter_vector
        ).ravel()

    canonical = {
        name: evaluate_map(name) for name in ("P", "q", "A", "b", "d")
    }
    actual_P = sparse.csc_matrix(
        (canonical["P"], compiled.P.row_indices, compiled.P.column_ptr),
        shape=(compiled.P.rows, compiled.P.columns),
    )
    actual_A = sparse.csc_matrix(
        (canonical["A"], compiled.A.row_indices, compiled.A.column_ptr),
        shape=(compiled.A.rows, compiled.A.columns),
    )
    expected, _, inverse_data = problem.get_problem_data(
        cp.CLARABEL,
        enforce_dpp=True,
        canon_backend="COO",
    )
    np.testing.assert_array_equal(actual_P.toarray(), expected["P"].toarray())
    np.testing.assert_array_equal(canonical["q"], expected["c"])
    np.testing.assert_array_equal(actual_A.toarray(), expected["A"].toarray())
    np.testing.assert_array_equal(canonical["b"], expected["b"])
    np.testing.assert_array_equal(canonical["d"], [inverse_data[-1]["offset"]])
    assert compiled.cone_initializers == (
        "ClarabelZeroConeT(1)",
        "ClarabelNonnegativeConeT(1)",
        "ClarabelSecondOrderConeT(3)",
        "ClarabelPSDTriangleConeT(2)",
        "ClarabelExponentialConeT()",
        "ClarabelPowerConeT(0.40000000000000002)",
    )


def test_generated_class_is_persistent_and_instance_owned(tmp_path: Path):
    artifact = _generate(tmp_path)
    source = artifact.instance_header.read_text()
    assert "class alignas(64) GeneratedMpo final" in source
    assert "clarabel_DefaultSolver_update_A" in source
    assert "clarabel_DefaultSolver_update_q" in source
    assert "clarabel_DefaultSolver_update_b" in source
    assert "clarabel_DefaultSolver_free" in source
    assert "GeneratedMpo(const GeneratedMpo&) = delete" in source
    assert "inline static constexpr std::array" in source
    assert "cvxpygen" not in source.lower()
    manifest = json.loads(artifact.manifest_path.read_text())
    assert manifest["instance_owned"] is True
    assert manifest["persistent_solver"] is True
    assert manifest["backend"] == "cvxpy-direct-clarabel"
    assert manifest["canonicalization_backend"] == "COO"
    assert manifest["parameter_shard_size"] == 512
    assert [item["name"] for item in manifest["parameters"]] == [
        "expected_returns",
        "half_spread",
        "current_weights",
        "risk_radius",
        "risk_factor",
    ]
    assert manifest["parameters"][-1]["dirty_blocks"] == ["A"]
    assert manifest["schema_version"] == 5
    assert len(manifest["duals"]) == len(_mpo_problem().constraints)


def test_two_generated_instances_solve_concurrently(tmp_path: Path):
    artifact = _generate(tmp_path)
    driver = tmp_path / "driver.cpp"
    binary = tmp_path / "driver"
    driver.write_text(
        textwrap.dedent(
            """
            #include "mpo_instance.hpp"
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
            #include "mpo_instance.hpp"
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
        #include "mpo_instance.hpp"
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
    shared, _ = build_shared(
        source,
        extra_include_dirs=artifact.include_dirs,
        extra_link_files=artifact.link_files,
        extra_fingerprint_files=artifact.fingerprint_files,
    )
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
        #include "qp_instance.hpp"
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
    shared, _ = build_shared(
        source,
        extra_include_dirs=artifact.include_dirs,
        extra_link_files=artifact.link_files,
        extra_fingerprint_files=artifact.fingerprint_files,
    )
    library = ctypes.CDLL(str(shared))
    library.solve_qp.argtypes = [ctypes.c_double]
    library.solve_qp.restype = ctypes.c_double
    assert library.solve_qp(1.5) == pytest.approx(1.5, abs=1e-7)
    assert library.solve_qp(-1.5) == pytest.approx(0.0, abs=1e-7)


@pytest.mark.skipif(os.name != "posix", reason="GNU linker allocation wrappers")
def test_warm_generated_solver_hot_path_has_zero_allocations(tmp_path: Path):
    artifact = _generate_qp(tmp_path)
    driver = tmp_path / "allocation_audit.cpp"
    binary = tmp_path / "allocation_audit"
    driver.write_text(
        textwrap.dedent(
            """
            #include "qp_instance.hpp"
            #include "stackdsl/ops/clarabel_program.hpp"
            #include <array>
            #include <atomic>
            #include <cstddef>
            #include <cstdio>
            #include <cstdlib>

            static std::atomic<unsigned long long> allocations{0};
            static std::atomic<bool> count_allocations{false};

            extern "C" void* __real_malloc(std::size_t);
            extern "C" void* __real_calloc(std::size_t, std::size_t);
            extern "C" void* __real_realloc(void*, std::size_t);
            extern "C" void* __real_aligned_alloc(std::size_t, std::size_t);

            extern "C" void* __wrap_malloc(std::size_t size) {
                if (count_allocations.load(std::memory_order_relaxed)) {
                    allocations.fetch_add(1, std::memory_order_relaxed);
                }
                return __real_malloc(size);
            }
            extern "C" void* __wrap_calloc(std::size_t count, std::size_t size) {
                if (count_allocations.load(std::memory_order_relaxed)) {
                    allocations.fetch_add(1, std::memory_order_relaxed);
                }
                return __real_calloc(count, size);
            }
            extern "C" void* __wrap_realloc(void* pointer, std::size_t size) {
                if (count_allocations.load(std::memory_order_relaxed)) {
                    allocations.fetch_add(1, std::memory_order_relaxed);
                }
                return __real_realloc(pointer, size);
            }
            extern "C" void* __wrap_aligned_alloc(
                std::size_t alignment, std::size_t size
            ) {
                if (count_allocations.load(std::memory_order_relaxed)) {
                    allocations.fetch_add(1, std::memory_order_relaxed);
                }
                return __real_aligned_alloc(alignment, size);
            }

            struct InitialSource {
                using shape = stackdsl::TensorShape<3>;
                template <class Context>
                static double read_flat(
                    const Context& ctx, std::size_t offset
                ) noexcept {
                    return ctx.initial[offset];
                }
                template <class Context>
                static void load_contiguous(
                    const Context& ctx,
                    std::size_t base,
                    std::size_t count,
                    double* out
                ) noexcept {
                    for (std::size_t index = 0; index < count; ++index) {
                        out[index] = ctx.initial[base + index];
                    }
                }
            };

            struct LowerSource {
                using shape = stackdsl::TensorShape<3>;
                template <class Context>
                static double read_flat(
                    const Context& ctx, std::size_t offset
                ) noexcept {
                    return ctx.lower[offset];
                }
                template <class Context>
                static void load_contiguous(
                    const Context& ctx,
                    std::size_t base,
                    std::size_t count,
                    double* out
                ) noexcept {
                    for (std::size_t index = 0; index < count; ++index) {
                        out[index] = ctx.lower[base + index];
                    }
                }
            };

            struct Output {};
            struct Context {
                std::array<double, 3> initial{1.0, 2.0, 3.0};
                std::array<double, 3> lower{0.0, 0.0, 0.0};
                std::array<double, 3> output{};

                template <class>
                double* write_ptr() noexcept { return output.data(); }
            };

            using Node = stackdsl::ClarabelNode<
                GeneratedQp,
                stackdsl::ClarabelParameterList<
                    stackdsl::ClarabelPreviousPrimalBinding<
                        0, 0, 0, 3, 1, InitialSource>,
                    stackdsl::ClarabelParameterBinding<1, LowerSource>
                >,
                stackdsl::ClarabelProjectionList<
                    stackdsl::ClarabelPrimalProjection<0, 0, 3, 1, Output>
                >
            >;

            int main() {
                Node node;
                Context context;
                node.setup();
                auto solve = [&](int iteration) {
                    context.lower[0] = 1e-6 * iteration;
                    node.on_data(context);
                    return context.output[0];
                };
                volatile double checksum = 0.0;
                for (int iteration = 0; iteration < 10; ++iteration) {
                    checksum += solve(iteration);
                }
                allocations.store(0, std::memory_order_relaxed);
                count_allocations.store(true, std::memory_order_relaxed);
                for (int iteration = 0; iteration < 100; ++iteration) {
                    checksum += solve(iteration + 10);
                }
                count_allocations.store(false, std::memory_order_relaxed);
                const auto count = allocations.load(std::memory_order_relaxed);
                std::printf("%llu %.17g\\n", count, static_cast<double>(checksum));
                return count == 0 ? 0 : 1;
            }
            """
        )
    )
    command = [
        os.environ.get("CXX", "g++"),
        "-std=gnu++20",
        "-O3",
        "-Isrc/trading_dsl_engine/cpp_stream/cpp",
        *(f"-I{path}" for path in artifact.include_dirs),
        str(driver),
        str(artifact.clarabel.static_library),
        "-Wl,--wrap=malloc",
        "-Wl,--wrap=calloc",
        "-Wl,--wrap=realloc",
        "-Wl,--wrap=aligned_alloc",
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
    allocations, checksum = completed.stdout.split()
    assert int(allocations) == 0
    assert np.isfinite(float(checksum))


def test_decorated_problem_factory_caches_and_projects_all_result_kinds(
    tmp_path: Path, monkeypatch
):
    from trading_dsl_engine.base.dsl import var
    from trading_dsl_engine.cpp_stream import compile_formula

    native = _clarabel_native()

    @cvxpy_program(
        cache_dir=tmp_path / "program-cache",
        clarabel=native,
    )
    def MPO(
        expected_returns,
        half_spread_bps,
        current_weights,
        risk_factor,
        risk_radius=0.08,
    ) -> cp.Problem:
        n_horizons, n_assets = expected_returns.shape
        expected_returns = cp.Parameter(
            expected_returns.shape, name="expected_returns"
        )
        half_spread_bps = cp.Parameter(
            half_spread_bps.shape,
            name="half_spread_bps",
            nonneg=True,
        )
        current_weights = cp.Parameter(
            current_weights.shape, name="current_weights"
        )
        risk_factor = cp.Parameter(risk_factor.shape, name="risk_factor")
        risk_radius = cp.Parameter(name="risk_radius", nonneg=True)
        weights = cp.Variable((n_horizons, n_assets), name="weights")
        turnover = cp.Variable((n_horizons, n_assets), name="turnover")
        delta = weights - cp.vstack([current_weights, weights[:-1]])
        turnover_up = turnover >= delta
        turnover_up.set_label("turnover_up")
        turnover_down = turnover >= -delta
        turnover_down.set_label("turnover_down")
        constraints = [turnover_up, turnover_down]
        for horizon in range(n_horizons):
            risk = cp.SOC(risk_radius, risk_factor @ weights[horizon])
            risk.set_label(f"risk_{horizon}")
            constraints.append(risk)
        return cp.Problem(
            cp.Minimize(
                -cp.sum(cp.multiply(expected_returns, weights))
                + cp.sum(
                    cp.multiply(half_spread_bps * 1e-4, turnover)
                )
            ),
            constraints,
        )

    rows, n_assets, n_horizons = 12, 3, 2
    rng = np.random.default_rng(44)
    data = {
        "expected_returns": rng.normal(
            scale=1e-4, size=(rows, n_assets, n_horizons)
        ),
        "half_spread_bps": np.broadcast_to(
            np.linspace(0.5, 1.0, n_assets), (rows, n_assets)
        ).copy(),
        "current_weights": np.zeros((rows, n_assets)),
        "risk_factor": np.broadcast_to(
            np.eye(n_assets), (rows, n_assets, n_assets)
        ).copy(),
    }
    mpo = MPO(
        expected_returns=var("expected_returns"),
        half_spread_bps=var("half_spread_bps"),
        current_weights=var("current_weights"),
        risk_factor=var("risk_factor"),
    )
    fields = [
        get_field(mpo, "weights[0]"),
        get_field(mpo, "turnover_up.dual[0]"),
        get_field(mpo, "risk_0.dual"),
        get_field(mpo, "risk_0.value"),
        get_field(mpo, "objective"),
        get_field(mpo, "iterations"),
        get_field(mpo, "status"),
        get_field(mpo, "primal_residual"),
        get_field(mpo, "dual_residual"),
    ]
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_STREAM_CACHE", str(tmp_path / "native-cache")
    )
    runtime = compile_formula(fields, data)
    generated = runtime.generated_cpp.read_text()
    row_loop = "for (std::size_t t = row_begin; t < row_end; ++t)"
    assert generated.count(row_loop) == 1
    assert generated.count("stackdsl::ClarabelNode<") == 1
    assert "ClarabelResultKind::Dual" in generated
    assert "ClarabelResultKind::Info" in generated
    optimizer_stages = [
        stage
        for stage in runtime.plan.stages
        if stage.kind in {"clarabel", "clarabel_bundle"}
    ]
    assert len(optimizer_stages) == 1
    assert optimizer_stages[0].kind == "clarabel_bundle"
    assert len(optimizer_stages[0].members) == len(fields)

    result = runtime.run(out_path=tmp_path / "decorated-mpo.npy")
    (
        weight_values,
        turnover_duals,
        risk_duals,
        risk_values,
        objectives,
        iterations,
        statuses,
        primal_residuals,
        dual_residuals,
    ) = result.load(mmap_mode=None)
    assert weight_values.shape == (rows, n_assets)
    assert turnover_duals.shape == (rows, n_assets)
    assert risk_duals.shape == (rows, n_assets + 1)
    assert risk_values.shape == (rows, n_assets + 1)
    for values in (
        weight_values,
        turnover_duals,
        risk_duals,
        risk_values,
        objectives,
        iterations,
        statuses,
        primal_residuals,
        dual_residuals,
    ):
        assert np.isfinite(values).all()

    reference = MPO.factory(
        data["expected_returns"][-1].T,
        data["half_spread_bps"][-1],
        data["current_weights"][-1],
        data["risk_factor"][-1],
        0.08,
    )
    reference_values = {
        "expected_returns": data["expected_returns"][-1].T,
        "half_spread_bps": data["half_spread_bps"][-1],
        "current_weights": data["current_weights"][-1],
        "risk_factor": data["risk_factor"][-1],
        "risk_radius": 0.08,
    }
    for parameter in reference.parameters():
        parameter.value = reference_values[parameter.name()]
    reference.solve(solver=cp.CLARABEL, presolve_enable=False)
    reference_weights = next(
        variable for variable in reference.variables() if variable.name() == "weights"
    )
    np.testing.assert_allclose(
        weight_values[-1], reference_weights.value[0], rtol=2e-5, atol=2e-7
    )
    np.testing.assert_allclose(
        turnover_duals[-1],
        reference.constraints[0].dual_value[0],
        rtol=2e-4,
        atol=2e-7,
    )
    reference_risk_dual = np.concatenate(
        [
            np.asarray(part, dtype=np.float64).reshape(-1)
            for part in reference.constraints[2].dual_value
        ]
    )
    np.testing.assert_allclose(
        risk_duals[-1], reference_risk_dual, rtol=2e-4, atol=2e-7
    )
    expected_risk_value = np.concatenate(
        ([0.08], data["risk_factor"][-1] @ reference_weights.value[0])
    )
    np.testing.assert_allclose(
        risk_values[-1], expected_risk_value, rtol=2e-5, atol=2e-7
    )
    assert objectives[-1] == pytest.approx(reference.value, rel=2e-5, abs=2e-7)

    # The exact factory/shape/request set owns one generated sub-program and is
    # reused by a second full-DAG compile rather than canonicalizing again.
    manifests = tuple((tmp_path / "program-cache").rglob("clarabel_program_manifest.json"))
    assert len(manifests) == 1
    manifest_mtime = manifests[0].stat().st_mtime_ns
    second_runtime = compile_formula(fields, data)
    assert manifests[0].stat().st_mtime_ns == manifest_mtime
    assert second_runtime.generated_cpp.read_text().count(
        "stackdsl::ClarabelNode<"
    ) == 1

    alternate_mpo = MPO(
        expected_returns=var("expected_returns"),
        half_spread_bps=var("half_spread_bps"),
        current_weights=var("current_weights"),
        risk_factor=var("risk_factor"),
    )
    alternate_fields = [
        get_field(alternate_mpo, "weights[1]"),
        get_field(alternate_mpo, "risk_0.value"),
    ]
    compile_formula(alternate_fields, data)
    assert tuple(
        (tmp_path / "program-cache").rglob("clarabel_program_manifest.json")
    ) == manifests
    assert manifests[0].stat().st_mtime_ns == manifest_mtime

    invalid = MPO(
        expected_returns=var("expected_returns"),
        half_spread_bps=var("half_spread_bps"),
        current_weights=var("current_weights"),
        risk_factor=var("risk_factor"),
    )
    with pytest.raises(KeyError, match="unknown generated field"):
        compile_formula(get_field(invalid, "not_a_solver_field"), data)


def test_previous_solution_feedback_is_initialized_and_forces_sequential_rows(
    tmp_path: Path, monkeypatch
):
    from trading_dsl_engine.base.dsl import ewm, var
    from trading_dsl_engine.cpp_stream import compile_formula

    @cvxpy_program(
        cache_dir=tmp_path / "feedback-program-cache",
        clarabel=_clarabel_native(),
    )
    def FollowTarget(target, current_weights) -> cp.Problem:
        shape = target.shape
        target = cp.Parameter(shape, name="target")
        # This parameter's shape comes from the target declaration, allowing a
        # scalar first-row initializer to broadcast without weakening shape
        # validation of ordinary direct bindings.
        current_weights = cp.Parameter(shape, name="current_weights")
        weights = cp.Variable(shape, name="weights")
        return cp.Problem(
            cp.Minimize(
                cp.sum_squares(weights - target)
                + cp.sum_squares(weights - current_weights)
            )
        )

    rows, n_assets = 9, 4
    rng = np.random.default_rng(707)
    targets = rng.normal(scale=0.1, size=(rows, n_assets))
    mpo = FollowTarget(
        target=var("target"),
        current_weights=previous_solution("weights", initial=0.0),
    )
    weights = get_field(mpo, "weights")
    monkeypatch.setenv(
        "TRADING_DSL_ENGINE_CPP_STREAM_CACHE",
        str(tmp_path / "feedback-native-cache"),
    )
    runtime = compile_formula(weights, {"target": targets})
    assert runtime.parallel_plan.mode == "serial"
    assert "prior-solve state" in runtime.parallel_plan.reason
    generated = runtime.generated_cpp.read_text()
    assert "stackdsl::ClarabelPreviousPrimalBinding" in generated

    actual = runtime.run(
        out_path=tmp_path / "feedback.npy", threads=8
    ).load(mmap_mode=None)
    expected = np.empty_like(targets)
    previous = np.zeros(n_assets)
    for row in range(rows):
        expected[row] = 0.5 * (targets[row] + previous)
        previous = expected[row]
    np.testing.assert_allclose(actual, expected, rtol=5e-5, atol=2e-7)

    initial_weights = rng.normal(scale=0.05, size=n_assets)
    vector_initialized = FollowTarget(
        target=var("target"),
        current_weights=previous_solution(
            "weights", initial=var("initial_weights")
        ),
    )
    vector_runtime = compile_formula(
        get_field(vector_initialized, "weights"),
        {
            "target": targets,
            "initial_weights": np.broadcast_to(
                initial_weights, targets.shape
            ).copy(),
        },
    )
    vector_actual = vector_runtime.run(
        out_path=tmp_path / "feedback-vector-initial.npy"
    ).load(mmap_mode=None)
    vector_expected = np.empty_like(targets)
    previous = initial_weights
    for row in range(rows):
        vector_expected[row] = 0.5 * (targets[row] + previous)
        previous = vector_expected[row]
    np.testing.assert_allclose(
        vector_actual, vector_expected, rtol=5e-5, atol=2e-7
    )

    independent = FollowTarget(
        target=var("target"),
        current_weights=var("current_weights"),
    )
    independent_runtime = compile_formula(
        get_field(independent, "weights"),
        {
            "target": targets,
            "current_weights": np.zeros_like(targets),
        },
    )
    assert independent_runtime.parallel_plan.mode == "rows"

    IndependentHint = cvxpy_program(
        FollowTarget.factory,
        cache_dir=tmp_path / "feedback-program-cache",
        clarabel=_clarabel_native(),
        sequential=False,
    )
    stateful_input = IndependentHint(
        target=ewm(var("target"), 3),
        current_weights=var("current_weights"),
    )
    stateful_runtime = compile_formula(
        get_field(stateful_input, "weights"),
        {
            "target": targets,
            "current_weights": np.zeros_like(targets),
        },
    )
    assert stateful_runtime.parallel_plan.mode == "serial"

    OrderedTarget = cvxpy_program(
        FollowTarget.factory,
        cache_dir=tmp_path / "feedback-program-cache",
        clarabel=_clarabel_native(),
        sequential=True,
    )
    ordered = OrderedTarget(
        target=var("target"),
        current_weights=var("current_weights"),
    )
    ordered_runtime = compile_formula(
        get_field(ordered, "weights"),
        {
            "target": targets,
            "current_weights": np.zeros_like(targets),
        },
    )
    assert ordered_runtime.parallel_plan.mode == "serial"
    assert len(
        tuple(
            (tmp_path / "feedback-program-cache").rglob(
                "clarabel_program_manifest.json"
            )
        )
    ) == 1

    with pytest.raises(ValueError, match="temporal dependency"):
        IndependentHint(
            target=var("target"),
            current_weights=previous_solution("weights", initial=0.0),
        )


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

    n_assets, n_horizons, rows = 3, 2, 24

    @cvxpy_program(
        cache_dir=tmp_path / "fused-program-cache",
        clarabel=_clarabel_native(),
    )
    def MPO(
        expected_returns,
        half_spread_bps,
        current_weights,
        risk_factor,
        risk_radius=0.08,
    ) -> cp.Problem:
        n_horizons, n_assets = expected_returns.shape
        expected_returns = cp.Parameter(
            expected_returns.shape, name="expected_returns"
        )
        half_spread_bps = cp.Parameter(
            half_spread_bps.shape,
            nonneg=True,
            name="half_spread_bps",
        )
        current_weights = cp.Parameter(
            current_weights.shape, name="current_weights"
        )
        risk_factor = cp.Parameter(risk_factor.shape, name="risk_factor")
        risk_radius = cp.Parameter(nonneg=True, name="risk_radius")
        weights = cp.Variable((n_horizons, n_assets), name="weights")
        turnover = cp.Variable((n_horizons, n_assets), name="turnover")
        previous = cp.vstack([current_weights, weights[:-1]])
        delta = weights - previous
        constraints = [turnover >= delta, turnover >= -delta]
        constraints.extend(
            cp.SOC(risk_radius, risk_factor @ weights[horizon])
            for horizon in range(n_horizons)
        )
        return cp.Problem(
            cp.Minimize(
                -cp.sum(cp.multiply(expected_returns, weights))
                + cp.sum(cp.multiply(half_spread_bps * 1e-4, turnover))
            ),
            constraints,
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
    mpo = MPO(
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
    assert generated.count("stackdsl::ClarabelNode<") == 1
    assert "stackdsl::PsdFactorNode<" in generated
    clarabel_stages = [
        stage
        for stage in runtime.plan.stages
        if stage.kind in {"clarabel", "clarabel_bundle"}
    ]
    assert len(clarabel_stages) == 1
    assert clarabel_stages[0].kind == "clarabel_bundle"
    assert len(clarabel_stages[0].members) == 2

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


def test_failed_feedback_solve_cannot_publish_or_carry_infeasibility_certificate(tmp_path):
    from trading_dsl_engine.base.dsl import var
    from trading_dsl_engine.cpp_stream import compile_formula

    @cvxpy_program(cache_dir=tmp_path / 'failed-feedback', clarabel=_clarabel_native())
    def Bound(lower, previous):
        lower = cp.Parameter(lower.shape, name='lower')
        previous = cp.Parameter(lower.shape, name='previous')
        x = cp.Variable(lower.shape, name='x')
        return cp.Problem(cp.Minimize(cp.sum_squares(x - previous)), [x >= lower, x <= 1])

    model = Bound(var('lower'), previous_solution('x', initial=0.))
    runtime = compile_formula({'x': get_field(model, 'x'), 'status': get_field(model, 'status')},
                              {'lower': np.array([[.5], [1.5], [.2]])}, n_instruments=1)
    got = runtime.run(out_path=tmp_path / 'must-not-publish.npy').load()
    np.testing.assert_allclose(got['x'][0], .5, atol=1e-7)
    assert np.isnan(got['x'][1:]).all()
    np.testing.assert_array_equal(got['status'], [1., 2., 2.])
