#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>

#include "stackdsl/engine.hpp"
#include "stackdsl/ops/cat.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

struct RidgePredsProjection {};
struct RidgeBetaProjection {};

namespace ridge_detail {

template <std::size_t K>
STACKDSL_HOT bool finite_vector(const std::array<double, K>& x) noexcept {
    for (double value : x) {
        if (!std::isfinite(value)) return false;
    }
    return true;
}

template <std::size_t K>
STACKDSL_HOT double dot(
    const std::array<double, K>& x,
    const double* STACKDSL_RESTRICT beta
) noexcept {
    double result = 0.0;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
    for (std::size_t j = 0; j < K; ++j) {
        result = std::fma(x[j], beta[j], result);
    }
    return result;
}

template <std::size_t K>
STACKDSL_HOT bool cholesky_solve(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    std::array<double, K>& solution
) noexcept {
    std::array<double, K * K> lower{};
    double scale = 0.0;
    for (std::size_t i = 0; i < K; ++i) {
        const double diagonal = system[i * K + i];
        if (!std::isfinite(diagonal)) return false;
        scale = std::max(scale, std::abs(diagonal));
    }
    const double tolerance = std::max(1e-15, scale * 1e-14);
    for (std::size_t i = 0; i < K; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            double value = system[i * K + j];
            if (!std::isfinite(value)) return false;
            for (std::size_t k = 0; k < j; ++k) {
                value = std::fma(-lower[i * K + k], lower[j * K + k], value);
            }
            if (i == j) {
                if (!(value > tolerance)) return false;
                lower[i * K + j] = std::sqrt(value);
            } else {
                lower[i * K + j] = value / lower[j * K + j];
            }
        }
    }
    std::array<double, K> intermediate{};
    for (std::size_t i = 0; i < K; ++i) {
        double value = rhs[i];
        if (!std::isfinite(value)) return false;
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
        if (!std::isfinite(solution[i])) return false;
    }
    return true;
}

template <std::size_t K>
STACKDSL_HOT bool gaussian_solve(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    std::array<double, K>& solution
) noexcept {
    std::array<double, K * K> a = system;
    std::array<double, K> b = rhs;
    double scale = 0.0;
    for (double value : a) {
        if (!std::isfinite(value)) return false;
        scale = std::max(scale, std::abs(value));
    }
    for (double value : b) {
        if (!std::isfinite(value)) return false;
    }
    const double tolerance = std::max(1e-15, scale * 1e-12);

    for (std::size_t column = 0; column < K; ++column) {
        std::size_t pivot = column;
        double pivot_abs = std::abs(a[column * K + column]);
        for (std::size_t row = column + 1; row < K; ++row) {
            const double candidate = std::abs(a[row * K + column]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (!(pivot_abs > tolerance)) return false;
        if (pivot != column) {
            for (std::size_t j = column; j < K; ++j) {
                std::swap(a[column * K + j], a[pivot * K + j]);
            }
            std::swap(b[column], b[pivot]);
        }
        const double diagonal = a[column * K + column];
        for (std::size_t row = column + 1; row < K; ++row) {
            const double factor = a[row * K + column] / diagonal;
            a[row * K + column] = 0.0;
            for (std::size_t j = column + 1; j < K; ++j) {
                a[row * K + j] = std::fma(-factor, a[column * K + j], a[row * K + j]);
            }
            b[row] = std::fma(-factor, b[column], b[row]);
        }
    }

    for (std::size_t reverse = 0; reverse < K; ++reverse) {
        const std::size_t row = K - 1 - reverse;
        double residual = b[row];
        for (std::size_t j = row + 1; j < K; ++j) {
            residual = std::fma(-a[row * K + j], solution[j], residual);
        }
        const double diagonal = a[row * K + row];
        if (!(std::abs(diagonal) > tolerance)) return false;
        solution[row] = residual / diagonal;
        if (!std::isfinite(solution[row])) return false;
    }
    return true;
}

template <std::size_t K>
bool pseudo_inverse_solve(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    std::array<double, K>& solution
) noexcept {
    std::array<double, K * K> a = system;
    std::array<double, K * K> eigenvectors{};
    for (std::size_t i = 0; i < K; ++i) eigenvectors[i * K + i] = 1.0;

    double scale = 0.0;
    for (double value : a) {
        if (!std::isfinite(value)) return false;
        scale = std::max(scale, std::abs(value));
    }
    for (double value : rhs) {
        if (!std::isfinite(value)) return false;
    }
    const double off_tolerance = std::max(1e-15, scale * 1e-13);
    constexpr std::size_t max_rotations = 32 * (K > 1 ? K * K : 1);

    for (std::size_t rotation = 0; rotation < max_rotations; ++rotation) {
        std::size_t p = 0;
        std::size_t q = 0;
        double largest = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            for (std::size_t j = i + 1; j < K; ++j) {
                const double magnitude = std::abs(a[i * K + j]);
                if (magnitude > largest) {
                    largest = magnitude;
                    p = i;
                    q = j;
                }
            }
        }
        if (!(largest > off_tolerance)) break;

        const double app = a[p * K + p];
        const double aqq = a[q * K + q];
        const double apq = a[p * K + q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double t = std::copysign(1.0, tau) /
            (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double c = 1.0 / std::sqrt(1.0 + t * t);
        const double s = t * c;

        for (std::size_t k = 0; k < K; ++k) {
            if (k == p || k == q) continue;
            const double akp = a[k * K + p];
            const double akq = a[k * K + q];
            const double next_kp = c * akp - s * akq;
            const double next_kq = s * akp + c * akq;
            a[k * K + p] = next_kp;
            a[p * K + k] = next_kp;
            a[k * K + q] = next_kq;
            a[q * K + k] = next_kq;
        }
        a[p * K + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q * K + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p * K + q] = 0.0;
        a[q * K + p] = 0.0;

        for (std::size_t k = 0; k < K; ++k) {
            const double vkp = eigenvectors[k * K + p];
            const double vkq = eigenvectors[k * K + q];
            eigenvectors[k * K + p] = c * vkp - s * vkq;
            eigenvectors[k * K + q] = s * vkp + c * vkq;
        }
    }

    double max_eigenvalue = 0.0;
    for (std::size_t i = 0; i < K; ++i) {
        max_eigenvalue = std::max(max_eigenvalue, std::abs(a[i * K + i]));
    }
    const double eigen_tolerance = std::max(1e-15, max_eigenvalue * 1e-12);
    std::array<double, K> projected{};
    for (std::size_t eigen = 0; eigen < K; ++eigen) {
        double value = 0.0;
        for (std::size_t row = 0; row < K; ++row) {
            value = std::fma(eigenvectors[row * K + eigen], rhs[row], value);
        }
        const double lambda = a[eigen * K + eigen];
        projected[eigen] = std::abs(lambda) > eigen_tolerance ? value / lambda : 0.0;
    }
    for (std::size_t row = 0; row < K; ++row) {
        double value = 0.0;
        for (std::size_t eigen = 0; eigen < K; ++eigen) {
            value = std::fma(eigenvectors[row * K + eigen], projected[eigen], value);
        }
        if (!std::isfinite(value)) return false;
        solution[row] = value;
    }
    return true;
}

template <std::size_t K>
STACKDSL_HOT bool unconstrained_solve(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    std::array<double, K>& solution
) noexcept {
    solution.fill(0.0);
    if (cholesky_solve(system, rhs, solution)) return true;
    solution.fill(0.0);
    if (gaussian_solve(system, rhs, solution)) return true;
    solution.fill(0.0);
    return pseudo_inverse_solve(system, rhs, solution);
}

template <std::size_t K>
STACKDSL_HOT bool nonnegative_solve(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    const std::array<double, K>& fallback,
    std::array<double, K>& solution
) noexcept {
    for (std::size_t j = 0; j < K; ++j) {
        solution[j] = std::max(0.0, fallback[j]);
    }
    // Generic cyclic coordinate descent over the compile-time-sized quadratic.
    // No Ridge shape or group-specific implementation is selected.
    constexpr std::size_t max_sweeps = 64;
    for (std::size_t sweep = 0; sweep < max_sweeps; ++sweep) {
        double max_change = 0.0;
        for (std::size_t j = 0; j < K; ++j) {
            const double diagonal = system[j * K + j];
            if (!(diagonal > 1e-18) || !std::isfinite(diagonal)) continue;
            double residual = rhs[j];
            for (std::size_t k = 0; k < K; ++k) {
                if (k != j) residual = std::fma(-system[j * K + k], solution[k], residual);
            }
            const double next = std::max(0.0, residual / diagonal);
            max_change = std::max(max_change, std::abs(next - solution[j]));
            solution[j] = next;
        }
        if (max_change <= 1e-12) break;
    }
    return finite_vector(solution);
}

template <std::size_t Groups, std::size_t K, bool Stateful>
struct RidgeState;

template <std::size_t Groups, std::size_t K>
struct RidgeState<Groups, K, true> {
    std::array<double, Groups * K * K> xx{};
    std::array<double, Groups * K> xy{};
    std::array<std::uint8_t, Groups * K * K> has_xx{};
    std::array<std::uint8_t, Groups * K> has_xy{};
    std::array<std::uint64_t, Groups * K * K> last_xx{};
    std::array<std::uint64_t, Groups * K> last_xy{};
    std::array<double, Groups * K> beta{};
    std::uint64_t t = 0;
};

template <std::size_t Groups, std::size_t K>
struct RidgeState<Groups, K, false> {
    // Stateless/current-row Ridge needs no persistent moment or coefficient
    // storage. This empty policy is selected at compile time and optimized away.
};

}  // namespace ridge_detail

template <
    std::size_t N,
    class Features,
    class Y,
    class Weights,
    class Out,
    std::uint64_t HalfLifeBits,
    std::uint64_t LambdaBits,
    bool Nonnegative,
    bool Stateful,
    class Projection,
    class Execution = DirectExecution<N>
>
struct RidgeNode;

template <
    std::size_t N,
    class Y,
    class Weights,
    class Out,
    std::uint64_t HalfLifeBits,
    std::uint64_t LambdaBits,
    bool Nonnegative,
    bool Stateful,
    class Projection,
    class Execution,
    class... FeatureSources
>
struct RidgeNode<
    N,
    FeatureList<FeatureSources...>,
    Y,
    Weights,
    Out,
    HalfLifeBits,
    LambdaBits,
    Nonnegative,
    Stateful,
    Projection,
    Execution
> {
    static constexpr std::size_t K = sizeof...(FeatureSources);
    static constexpr std::size_t Groups = Execution::cross_state_size;
    static_assert(K > 0, "Ridge requires at least one feature");
    static_assert(Groups > 0);

    ridge_detail::RidgeState<Groups, K, Stateful> state{};

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        constexpr double half_life = std::bit_cast<double>(HalfLifeBits);
        constexpr double ridge_lambda_raw = std::bit_cast<double>(LambdaBits);
        constexpr bool instant = !Stateful;
        constexpr double ridge_lambda =
            std::isnan(ridge_lambda_raw) || ridge_lambda_raw < 0.0 ? 0.0 : ridge_lambda_raw;
        const double rho = instant || !(half_life > 0.0)
            ? 0.0
            : std::exp(std::log(0.5) / half_life);
        const double alpha = std::clamp(1.0 - rho, 0.0, 1.0);

        std::array<std::uint32_t, N> lane_groups{};
        std::array<std::uint32_t, N> active_groups{};
        std::array<std::uint8_t, N> active_index_by_lane{};
        std::size_t active_count = 0;
        for (std::size_t lane = 0; lane < N; ++lane) {
            const auto group = static_cast<std::uint32_t>(Execution::cross_group(ctx, lane));
            lane_groups[lane] = group;
            std::size_t active_index = 0;
            for (; active_index < active_count; ++active_index) {
                if (active_groups[active_index] == group) break;
            }
            if (active_index == active_count) active_groups[active_count++] = group;
            active_index_by_lane[lane] = static_cast<std::uint8_t>(active_index);
        }

        std::array<double, N * K * K> xx_new{};
        std::array<double, N * K> xy_new{};
        std::array<std::uint8_t, N * K * K> xx_valid{};
        std::array<std::uint8_t, N * K> xy_valid{};
        std::array<std::array<double, K>, N> features_by_lane{};
        std::array<std::uint8_t, N> prediction_valid{};

        for (std::size_t lane = 0; lane < N; ++lane) {
            auto& features = features_by_lane[lane];
            load_features(ctx, lane, features, FeatureList<FeatureSources...>{});
            const double y = ctx.template read<Y>(lane);
            const double weight = ctx.template read<Weights>(lane);
            prediction_valid[lane] = static_cast<std::uint8_t>(
                std::isfinite(y) && ridge_detail::finite_vector(features));
            if (!std::isfinite(weight)) continue;
            const std::size_t active = active_index_by_lane[lane];
            const std::size_t matrix_base = active * K * K;
            const std::size_t vector_base = active * K;
            for (std::size_t j = 0; j < K; ++j) {
                const double xj = features[j];
                if (!std::isfinite(xj)) continue;
                if (std::isfinite(y)) {
                    xy_new[vector_base + j] = std::fma(xj * weight, y, xy_new[vector_base + j]);
                    xy_valid[vector_base + j] = 1;
                }
                for (std::size_t k = j; k < K; ++k) {
                    const double xk = features[k];
                    if (!std::isfinite(xk)) continue;
                    const double contribution = xj * weight * xk;
                    xx_new[matrix_base + j * K + k] += contribution;
                    xx_valid[matrix_base + j * K + k] = 1;
                    if (j != k) {
                        xx_new[matrix_base + k * K + j] += contribution;
                        xx_valid[matrix_base + k * K + j] = 1;
                    }
                }
            }
        }

        if constexpr (Stateful && std::is_same_v<Projection, RidgePredsProjection>) {
            auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
            for (std::size_t lane = 0; lane < N; ++lane) {
                if (!prediction_valid[lane]) {
                    out[lane] = kNaN;
                    continue;
                }
                const std::size_t group = lane_groups[lane];
                out[lane] = ridge_detail::dot(
                    features_by_lane[lane],
                    state.beta.data() + group * K);
            }
        }

        std::array<std::array<double, K>, N> solved_betas{};
        for (std::size_t active = 0; active < active_count; ++active) {
            const std::size_t group = active_groups[active];
            const std::size_t matrix_base = active * K * K;
            const std::size_t vector_base = active * K;
            std::array<double, K * K> xx{};
            std::array<double, K> xy{};
            std::array<double, K> fallback{};

            if constexpr (Stateful) {
                const std::size_t state_matrix_base = group * K * K;
                const std::size_t state_vector_base = group * K;
                for (std::size_t j = 0; j < K; ++j) {
                    fallback[j] = state.beta[state_vector_base + j];
                    const std::size_t local_j = vector_base + j;
                    const std::size_t state_j = state_vector_base + j;
                    if (xy_valid[local_j]) {
                        if (state.has_xy[state_j]) {
                            const auto gap = state.t - state.last_xy[state_j];
                            const double a = gap == 1
                                ? alpha
                                : std::pow(alpha, static_cast<double>(gap));
                            state.xy[state_j] = std::fma(
                                a,
                                xy_new[local_j] - state.xy[state_j],
                                state.xy[state_j]);
                        } else {
                            state.xy[state_j] = xy_new[local_j];
                        }
                        state.has_xy[state_j] = 1;
                        state.last_xy[state_j] = state.t;
                    }
                    xy[j] = state.xy[state_j];
                    for (std::size_t k = 0; k < K; ++k) {
                        const std::size_t local_jk = matrix_base + j * K + k;
                        const std::size_t state_jk = state_matrix_base + j * K + k;
                        if (xx_valid[local_jk]) {
                            if (state.has_xx[state_jk]) {
                                const auto gap = state.t - state.last_xx[state_jk];
                                const double a = gap == 1
                                    ? alpha
                                    : std::pow(alpha, static_cast<double>(gap));
                                state.xx[state_jk] = std::fma(
                                    a,
                                    xx_new[local_jk] - state.xx[state_jk],
                                    state.xx[state_jk]);
                            } else {
                                state.xx[state_jk] = xx_new[local_jk];
                            }
                            state.has_xx[state_jk] = 1;
                            state.last_xx[state_jk] = state.t;
                        }
                        xx[j * K + k] = state.xx[state_jk];
                    }
                }
            } else {
                for (std::size_t j = 0; j < K; ++j) {
                    xy[j] = xy_valid[vector_base + j] ? xy_new[vector_base + j] : 0.0;
                    for (std::size_t k = 0; k < K; ++k) {
                        const std::size_t index = matrix_base + j * K + k;
                        xx[j * K + k] = xx_valid[index] ? xx_new[index] : 0.0;
                    }
                }
            }

            std::array<double, K * K> system = xx;
            for (std::size_t j = 0; j < K; ++j) {
                system[j * K + j] = std::fma(
                    ridge_lambda,
                    xx[j * K + j],
                    xx[j * K + j]);
            }
            auto& beta = solved_betas[active];
            bool solved = false;
            if constexpr (Nonnegative) {
                solved = ridge_detail::nonnegative_solve(system, xy, fallback, beta);
            } else {
                solved = ridge_detail::unconstrained_solve(system, xy, beta);
            }
            if (!solved) beta = fallback;
            if constexpr (Stateful) {
                const std::size_t state_vector_base = group * K;
                for (std::size_t j = 0; j < K; ++j) {
                    state.beta[state_vector_base + j] = beta[j];
                }
            }
        }

        if constexpr (std::is_same_v<Projection, RidgePredsProjection>) {
            if constexpr (!Stateful) {
                auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
                for (std::size_t lane = 0; lane < N; ++lane) {
                    if (!prediction_valid[lane]) {
                        out[lane] = kNaN;
                        continue;
                    }
                    out[lane] = ridge_detail::dot(
                        features_by_lane[lane],
                        solved_betas[active_index_by_lane[lane]].data());
                }
            }
        } else {
            static_assert(std::is_same_v<Projection, RidgeBetaProjection>);
            static_assert(
                std::is_same_v<Out, OutputDst>,
                "Ridge beta projection is currently materialized at a program/groupby root");
            if constexpr (Execution::cross_state_size == 1) {
                for (std::size_t j = 0; j < K; ++j) ctx.output[j] = solved_betas[0][j];
            } else {
                // In grouped execution beta is a per-instrument K-wide matrix:
                // each lane receives the coefficient vector of its current
                // static/dynamic cross-sectional group.
                for (std::size_t lane = 0; lane < N; ++lane) {
                    const auto& beta = solved_betas[active_index_by_lane[lane]];
                    for (std::size_t j = 0; j < K; ++j) {
                        ctx.output[lane * K + j] = beta[j];
                    }
                }
            }
        }

        if constexpr (Stateful) ++state.t;
    }
};

}  // namespace stackdsl
