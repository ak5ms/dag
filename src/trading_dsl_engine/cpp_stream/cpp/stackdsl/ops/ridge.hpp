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
#include "stackdsl/ops/eigen_solvers.hpp"
#include "stackdsl/ops/nnqp.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

struct RidgePredsProjection {};
struct RidgeBetaProjection {};
struct RidgeResidualsProjection {};
template <std::size_t Component> struct RidgeCoefficientProjection {};
struct RidgeStandardErrorsProjection {};
template <std::size_t Component> struct RidgeStandardErrorProjection {};
struct RidgeTStatsProjection {};
template <std::size_t Component> struct RidgeTStatProjection {};
struct RidgeSseProjection {};
struct RidgeSstProjection {};
struct RidgeR2Projection {};
struct RidgeResidualVarianceProjection {};
struct RidgeEffectiveDfProjection {};
struct RidgeEffectiveNProjection {};

template <class Out, class Projection>
struct RidgeProjectionBinding {
    using output_type = Out;
    using projection_type = Projection;
};

template <class... Bindings>
struct RidgeProjectionBundle {
    static_assert(sizeof...(Bindings) > 1);
};

namespace ridge_detail {

template <std::size_t K>
STACKDSL_HOT bool finite_vector(const std::array<double, K>& values) noexcept {
    for (double value : values) if (!std::isfinite(value)) return false;
    return true;
}

template <std::size_t K>
STACKDSL_HOT double dot(
    const std::array<double, K>& values,
    const double* STACKDSL_RESTRICT beta
) noexcept {
    double result = 0.0;
    for (std::size_t j = 0; j < K; ++j) {
        result = std::fma(values[j], beta[j], result);
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
                value = std::fma(
                    -lower[i * K + k],
                    lower[j * K + k],
                    value
                );
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
    std::array<double, K * K> matrix = system;
    std::array<double, K> values = rhs;
    double scale = 0.0;
    for (double value : matrix) {
        if (!std::isfinite(value)) return false;
        scale = std::max(scale, std::abs(value));
    }
    for (double value : values) if (!std::isfinite(value)) return false;
    const double tolerance = std::max(1e-15, scale * 1e-12);
    for (std::size_t column = 0; column < K; ++column) {
        std::size_t pivot = column;
        double pivot_abs = std::abs(matrix[column * K + column]);
        for (std::size_t row = column + 1; row < K; ++row) {
            const double candidate = std::abs(matrix[row * K + column]);
            if (candidate > pivot_abs) {
                pivot = row;
                pivot_abs = candidate;
            }
        }
        if (!(pivot_abs > tolerance)) return false;
        if (pivot != column) {
            for (std::size_t j = column; j < K; ++j) {
                std::swap(matrix[column * K + j], matrix[pivot * K + j]);
            }
            std::swap(values[column], values[pivot]);
        }
        const double diagonal = matrix[column * K + column];
        for (std::size_t row = column + 1; row < K; ++row) {
            const double factor = matrix[row * K + column] / diagonal;
            matrix[row * K + column] = 0.0;
            for (std::size_t j = column + 1; j < K; ++j) {
                matrix[row * K + j] = std::fma(
                    -factor,
                    matrix[column * K + j],
                    matrix[row * K + j]
                );
            }
            values[row] = std::fma(-factor, values[column], values[row]);
        }
    }
    for (std::size_t reverse = 0; reverse < K; ++reverse) {
        const std::size_t row = K - 1 - reverse;
        double residual = values[row];
        for (std::size_t j = row + 1; j < K; ++j) {
            residual = std::fma(-matrix[row * K + j], solution[j], residual);
        }
        const double diagonal = matrix[row * K + row];
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
    std::array<double, K * K> matrix = system;
    std::array<double, K * K> eigenvectors{};
    for (std::size_t i = 0; i < K; ++i) eigenvectors[i * K + i] = 1.0;
    double scale = 0.0;
    for (double value : matrix) {
        if (!std::isfinite(value)) return false;
        scale = std::max(scale, std::abs(value));
    }
    for (double value : rhs) if (!std::isfinite(value)) return false;
    const double off_tolerance = std::max(1e-15, scale * 1e-13);
    constexpr std::size_t max_rotations = 32 * (K > 1 ? K * K : 1);
    for (std::size_t rotation = 0; rotation < max_rotations; ++rotation) {
        std::size_t p = 0;
        std::size_t q = 0;
        double largest = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            for (std::size_t j = i + 1; j < K; ++j) {
                const double magnitude = std::abs(matrix[i * K + j]);
                if (magnitude > largest) {
                    largest = magnitude;
                    p = i;
                    q = j;
                }
            }
        }
        if (!(largest > off_tolerance)) break;
        const double app = matrix[p * K + p];
        const double aqq = matrix[q * K + q];
        const double apq = matrix[p * K + q];
        const double tau = (aqq - app) / (2.0 * apq);
        const double tangent = std::copysign(1.0, tau) /
            (std::abs(tau) + std::sqrt(1.0 + tau * tau));
        const double cosine = 1.0 / std::sqrt(1.0 + tangent * tangent);
        const double sine = tangent * cosine;
        for (std::size_t k = 0; k < K; ++k) {
            if (k == p || k == q) continue;
            const double akp = matrix[k * K + p];
            const double akq = matrix[k * K + q];
            matrix[k * K + p] = matrix[p * K + k] = cosine * akp - sine * akq;
            matrix[k * K + q] = matrix[q * K + k] = sine * akp + cosine * akq;
        }
        matrix[p * K + p] = cosine * cosine * app -
            2.0 * sine * cosine * apq + sine * sine * aqq;
        matrix[q * K + q] = sine * sine * app +
            2.0 * sine * cosine * apq + cosine * cosine * aqq;
        matrix[p * K + q] = matrix[q * K + p] = 0.0;
        for (std::size_t k = 0; k < K; ++k) {
            const double vkp = eigenvectors[k * K + p];
            const double vkq = eigenvectors[k * K + q];
            eigenvectors[k * K + p] = cosine * vkp - sine * vkq;
            eigenvectors[k * K + q] = sine * vkp + cosine * vkq;
        }
    }
    double max_eigenvalue = 0.0;
    for (std::size_t i = 0; i < K; ++i) {
        max_eigenvalue = std::max(
            max_eigenvalue,
            std::abs(matrix[i * K + i])
        );
    }
    const double eigen_tolerance = std::max(1e-15, max_eigenvalue * 1e-12);
    std::array<double, K> projected{};
    for (std::size_t eigen = 0; eigen < K; ++eigen) {
        double value = 0.0;
        for (std::size_t row = 0; row < K; ++row) {
            value = std::fma(
                eigenvectors[row * K + eigen],
                rhs[row],
                value
            );
        }
        const double lambda = matrix[eigen * K + eigen];
        projected[eigen] = std::abs(lambda) > eigen_tolerance
            ? value / lambda
            : 0.0;
    }
    for (std::size_t row = 0; row < K; ++row) {
        double value = 0.0;
        for (std::size_t eigen = 0; eigen < K; ++eigen) {
            value = std::fma(
                eigenvectors[row * K + eigen],
                projected[eigen],
                value
            );
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
STACKDSL_HOT bool coordinate_nonnegative_solve(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    const std::array<double, K>& fallback,
    std::array<double, K>& solution
) noexcept {
    for (std::size_t j = 0; j < K; ++j) {
        solution[j] = std::max(0.0, fallback[j]);
    }
    for (std::size_t sweep = 0; sweep < 64; ++sweep) {
        double max_change = 0.0;
        for (std::size_t j = 0; j < K; ++j) {
            const double diagonal = system[j * K + j];
            if (!(diagonal > 1e-18) || !std::isfinite(diagonal)) continue;
            double residual = rhs[j];
            for (std::size_t k = 0; k < K; ++k) {
                if (k != j) {
                    residual = std::fma(
                        -system[j * K + k],
                        solution[k],
                        residual
                    );
                }
            }
            const double next = std::max(0.0, residual / diagonal);
            max_change = std::max(max_change, std::abs(next - solution[j]));
            solution[j] = next;
        }
        if (max_change <= 1e-12) break;
    }
    return finite_vector(solution);
}

template <std::size_t K>
STACKDSL_HOT bool nnqp_nonnegative_solve(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    const std::array<double, K>& fallback,
    std::array<double, K>& solution
) noexcept {
    solution.fill(0.0);
    return nnqp::solve<K>(system, rhs, fallback, solution);
}

template <std::size_t K>
STACKDSL_HOT bool inverse(
    const std::array<double, K * K>& system,
    std::array<double, K * K>& result
) noexcept {
    for (std::size_t column = 0; column < K; ++column) {
        std::array<double, K> rhs{};
        std::array<double, K> solution{};
        rhs[column] = 1.0;
        if (!unconstrained_solve(system, rhs, solution)) return false;
        for (std::size_t row = 0; row < K; ++row) {
            result[row * K + column] = solution[row];
        }
    }
    return true;
}

template <std::size_t Groups, std::size_t K, bool Stateful> struct RidgeState;
template <std::size_t Groups, std::size_t K> struct RidgeState<Groups, K, true> {
    std::array<double, Groups * K * K> xx{};
    std::array<double, Groups * K> xy{};
    std::array<std::uint8_t, Groups * K * K> has_xx{};
    std::array<std::uint8_t, Groups * K> has_xy{};
    std::array<std::uint64_t, Groups * K * K> last_xx{};
    std::array<std::uint64_t, Groups * K> last_xy{};
    std::array<double, Groups * K> beta{};
    std::array<std::uint8_t, Groups> full_synced{};
    std::array<std::uint64_t, Groups> last_full{};
    std::uint64_t t = 0;
};
template <std::size_t Groups, std::size_t K> struct RidgeState<Groups, K, false> {};

template <std::size_t Groups, std::size_t K, bool Enabled>
struct RidgeMetricState {};
template <std::size_t Groups, std::size_t K>
struct RidgeMetricState<Groups, K, true> {
    std::array<double, Groups * K * K> xx{};
    std::array<double, Groups * K> xy{};
    std::array<double, Groups> ywy{};
    std::array<double, Groups> wy{};
    std::array<double, Groups> weight{};
    std::array<double, Groups> weight_square{};
    std::array<std::uint8_t, Groups> initialized{};
    std::array<std::uint64_t, Groups> last_update{};
};

template <
    std::size_t Groups,
    std::size_t K,
    bool NeedsMetrics,
    bool NeedsInference,
    bool Enabled
>
struct RidgeResultCache {};

template <
    std::size_t Groups,
    std::size_t K,
    bool NeedsMetrics,
    bool NeedsInference
>
struct RidgeResultCache<Groups, K, NeedsMetrics, NeedsInference, true> {
    std::array<double, Groups * K> beta{};
    std::array<std::uint8_t, Groups> initialized{};
    std::array<double, NeedsInference ? Groups * K : 0> standard_errors{};
    std::array<double, NeedsInference ? Groups * K : 0> tstats{};
    std::array<double, NeedsMetrics ? Groups : 0> sse{};
    std::array<double, NeedsMetrics ? Groups : 0> sst{};
    std::array<double, NeedsMetrics ? Groups : 0> r2{};
    std::array<double, NeedsInference ? Groups : 0> residual_variance{};
    std::array<double, NeedsInference ? Groups : 0> effective_df{};
    std::array<double, NeedsMetrics ? Groups : 0> effective_n{};

    STACKDSL_HOT void load(
        std::size_t group,
        std::array<double, K>& beta_out,
        std::array<double, K>& standard_errors_out,
        std::array<double, K>& tstats_out,
        double& sse_out,
        double& sst_out,
        double& r2_out,
        double& residual_variance_out,
        double& effective_df_out,
        double& effective_n_out
    ) const noexcept {
        const std::size_t vector_base = group * K;
        for (std::size_t j = 0; j < K; ++j) {
            beta_out[j] = beta[vector_base + j];
        }
        if constexpr (NeedsInference) {
            for (std::size_t j = 0; j < K; ++j) {
                standard_errors_out[j] =
                    standard_errors[vector_base + j];
                tstats_out[j] = tstats[vector_base + j];
            }
            residual_variance_out = residual_variance[group];
            effective_df_out = effective_df[group];
        }
        if constexpr (NeedsMetrics) {
            sse_out = sse[group];
            sst_out = sst[group];
            r2_out = r2[group];
            effective_n_out = effective_n[group];
        }
    }

    STACKDSL_HOT void store(
        std::size_t group,
        const std::array<double, K>& beta_in,
        const std::array<double, K>& standard_errors_in,
        const std::array<double, K>& tstats_in,
        double sse_in,
        double sst_in,
        double r2_in,
        double residual_variance_in,
        double effective_df_in,
        double effective_n_in
    ) noexcept {
        const std::size_t vector_base = group * K;
        for (std::size_t j = 0; j < K; ++j) {
            beta[vector_base + j] = beta_in[j];
        }
        if constexpr (NeedsInference) {
            for (std::size_t j = 0; j < K; ++j) {
                standard_errors[vector_base + j] =
                    standard_errors_in[j];
                tstats[vector_base + j] = tstats_in[j];
            }
            residual_variance[group] = residual_variance_in;
            effective_df[group] = effective_df_in;
        }
        if constexpr (NeedsMetrics) {
            sse[group] = sse_in;
            sst[group] = sst_in;
            r2[group] = r2_in;
            effective_n[group] = effective_n_in;
        }
        initialized[group] = 1;
    }
};

template <class Projection> struct projection_component {
    static constexpr std::size_t value = std::numeric_limits<std::size_t>::max();
};
template <std::size_t Component>
struct projection_component<RidgeCoefficientProjection<Component>> {
    static constexpr std::size_t value = Component;
};
template <std::size_t Component>
struct projection_component<RidgeStandardErrorProjection<Component>> {
    static constexpr std::size_t value = Component;
};
template <std::size_t Component>
struct projection_component<RidgeTStatProjection<Component>> {
    static constexpr std::size_t value = Component;
};
template <class Projection> struct is_coefficient_projection : std::false_type {};
template <std::size_t Component>
struct is_coefficient_projection<RidgeCoefficientProjection<Component>> : std::true_type {};
template <class Projection> struct is_standard_error_projection : std::false_type {};
template <std::size_t Component>
struct is_standard_error_projection<RidgeStandardErrorProjection<Component>> : std::true_type {};
template <class Projection> struct is_tstat_projection : std::false_type {};
template <std::size_t Component>
struct is_tstat_projection<RidgeTStatProjection<Component>> : std::true_type {};

template <class Projection>
struct projection_traits {
    static constexpr bool predicts =
        std::is_same_v<Projection, RidgePredsProjection>
        || std::is_same_v<Projection, RidgeResidualsProjection>;
    static constexpr bool full_coefficients =
        std::is_same_v<Projection, RidgeBetaProjection>
        || std::is_same_v<Projection, RidgeStandardErrorsProjection>
        || std::is_same_v<Projection, RidgeTStatsProjection>;
    static constexpr bool needs_inference =
        std::is_same_v<Projection, RidgeStandardErrorsProjection>
        || std::is_same_v<Projection, RidgeTStatsProjection>
        || std::is_same_v<Projection, RidgeResidualVarianceProjection>
        || std::is_same_v<Projection, RidgeEffectiveDfProjection>
        || is_standard_error_projection<Projection>::value
        || is_tstat_projection<Projection>::value;
    static constexpr bool needs_metrics =
        needs_inference
        || std::is_same_v<Projection, RidgeSseProjection>
        || std::is_same_v<Projection, RidgeSstProjection>
        || std::is_same_v<Projection, RidgeR2Projection>
        || std::is_same_v<Projection, RidgeEffectiveNProjection>;
};

template <class ProjectionSpec, class DefaultOut>
struct projection_set {
    static constexpr bool predicts = projection_traits<ProjectionSpec>::predicts;
    static constexpr bool needs_inference =
        projection_traits<ProjectionSpec>::needs_inference;
    static constexpr bool needs_metrics =
        projection_traits<ProjectionSpec>::needs_metrics;

    template <class Function>
    STACKDSL_HOT static void for_each(Function&& function) noexcept {
        function.template operator()<DefaultOut, ProjectionSpec>();
    }
};

template <class DefaultOut, class... Bindings>
struct projection_set<RidgeProjectionBundle<Bindings...>, DefaultOut> {
    static constexpr bool predicts =
        (projection_traits<typename Bindings::projection_type>::predicts || ...);
    static constexpr bool needs_inference =
        (projection_traits<typename Bindings::projection_type>::needs_inference || ...);
    static constexpr bool needs_metrics =
        (projection_traits<typename Bindings::projection_type>::needs_metrics || ...);

    template <class Function>
    STACKDSL_HOT static void for_each(Function&& function) noexcept {
        (function.template operator()<
            typename Bindings::output_type,
            typename Bindings::projection_type
        >(), ...);
    }
};

}  // namespace ridge_detail

template <
    std::size_t N,
    class Features,
    class Y,
    class Weights,
    class Out,
    std::uint64_t AlphaBits,
    std::uint64_t LambdaBits,
    bool Nonnegative,
    bool Stateful,
    class Projection,
    class Execution = DirectExecution<N>,
    std::size_t RecomputeEvery = 1
>
struct RidgeNode;

template <
    std::size_t N,
    class Y,
    class Weights,
    class Out,
    std::uint64_t AlphaBits,
    std::uint64_t LambdaBits,
    bool Nonnegative,
    bool Stateful,
    class Projection,
    class Execution,
    std::size_t RecomputeEvery,
    class... FeatureSources
>
struct RidgeNode<
    N,
    FeatureList<FeatureSources...>,
    Y,
    Weights,
    Out,
    AlphaBits,
    LambdaBits,
    Nonnegative,
    Stateful,
    Projection,
    Execution,
    RecomputeEvery
> {
    static constexpr std::size_t K = FeatureList<FeatureSources...>::width;
    static constexpr std::size_t Groups = Execution::cross_state_size;
    static constexpr std::size_t MaxActiveGroups = Groups < N ? Groups : N;
    static_assert(K > 0 && Groups > 0);
    using Projections = ridge_detail::projection_set<Projection, Out>;
    static constexpr bool PredProjection = Projections::predicts;
    static constexpr bool NeedsInference = Projections::needs_inference;
    static constexpr bool NeedsMetrics = Projections::needs_metrics;
    static_assert(RecomputeEvery > 0, "Ridge recompute interval must be > 0");
    ridge_detail::RidgeState<Groups, K, Stateful> state{};
    ridge_detail::RidgeMetricState<Groups, K, Stateful && NeedsMetrics> metrics{};
    PeriodicRecompute<RecomputeEvery, Groups> recompute_schedule{};
    ridge_detail::RidgeResultCache<
        Groups,
        K,
        NeedsMetrics,
        NeedsInference,
        (RecomputeEvery > 1)
    > result_cache{};
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        constexpr double alpha = std::bit_cast<double>(AlphaBits);
        constexpr double lambda_raw = std::bit_cast<double>(LambdaBits);
        constexpr double ridge_lambda = std::isnan(lambda_raw) || lambda_raw < 0.0 ? 0.0 : lambda_raw;
        std::array<std::uint32_t, N> lane_groups{};
        std::array<std::uint32_t, MaxActiveGroups> active_groups{};
        std::array<std::uint8_t, N> active_index_by_lane{};
        std::size_t active_count = 1;
        if constexpr (Groups == 1) active_groups[0] = 0;
        else {
            active_count = 0;
            for (std::size_t lane = 0; lane < N; ++lane) {
                const auto group = static_cast<std::uint32_t>(Execution::cross_group(ctx, lane));
                lane_groups[lane] = group;
                std::size_t active = 0;
                for (; active < active_count; ++active) if (active_groups[active] == group) break;
                if (active == active_count) active_groups[active_count++] = group;
                active_index_by_lane[lane] = static_cast<std::uint8_t>(active);
            }
        }
        std::array<std::array<double, K>, N> features_by_lane{};
        std::array<double, N> y_by_lane{}, weights_by_lane{};
        std::array<std::uint8_t, N> prediction_valid{};
        bool all_finite = true;
        for (std::size_t lane = 0; lane < N; ++lane) {
            auto& features = features_by_lane[lane];
            load_features(ctx, lane, features, FeatureList<FeatureSources...>{});
            y_by_lane[lane] = ctx.template read<Y>(lane);
            weights_by_lane[lane] = ctx.template read<Weights>(lane);
            const bool features_finite = ridge_detail::finite_vector(features);
            prediction_valid[lane] = static_cast<std::uint8_t>(std::isfinite(y_by_lane[lane]) && features_finite);
            all_finite = all_finite && features_finite && std::isfinite(y_by_lane[lane]) && std::isfinite(weights_by_lane[lane]);
        }
        std::array<double, MaxActiveGroups * K * K> xx_new{};
        std::array<double, MaxActiveGroups * K> xy_new{};
        std::array<std::uint8_t, MaxActiveGroups * K * K> xx_valid{};
        std::array<std::uint8_t, MaxActiveGroups * K> xy_valid{};
        if (all_finite) {
            for (std::size_t lane = 0; lane < N; ++lane) {
                const std::size_t active = active_index_by_lane[lane], matrix_base = active * K * K, vector_base = active * K;
                const auto& features = features_by_lane[lane];
                const double weighted_y = weights_by_lane[lane] * y_by_lane[lane];
                for (std::size_t j = 0; j < K; ++j) {
                    const double weighted_xj = weights_by_lane[lane] * features[j];
                    xy_new[vector_base + j] = std::fma(features[j], weighted_y, xy_new[vector_base + j]);
                    for (std::size_t k = j; k < K; ++k) {
                        const double contribution = weighted_xj * features[k];
                        xx_new[matrix_base + j * K + k] += contribution;
                        if (j != k) xx_new[matrix_base + k * K + j] += contribution;
                    }
                }
            }
        } else {
            for (std::size_t lane = 0; lane < N; ++lane) {
                const double weight = weights_by_lane[lane];
                if (!std::isfinite(weight)) continue;
                const std::size_t active = active_index_by_lane[lane], matrix_base = active * K * K, vector_base = active * K;
                const auto& features = features_by_lane[lane];
                for (std::size_t j = 0; j < K; ++j) {
                    const double xj = features[j];
                    if (!std::isfinite(xj)) continue;
                    if (std::isfinite(y_by_lane[lane])) {
                        xy_new[vector_base + j] = std::fma(xj * weight, y_by_lane[lane], xy_new[vector_base + j]);
                        xy_valid[vector_base + j] = 1;
                    }
                    for (std::size_t k = j; k < K; ++k) {
                        const double xk = features[k];
                        if (!std::isfinite(xk)) continue;
                        const double contribution = xj * weight * xk;
                        xx_new[matrix_base + j * K + k] += contribution;
                        xx_valid[matrix_base + j * K + k] = 1;
                        if (j != k) { xx_new[matrix_base + k * K + j] += contribution; xx_valid[matrix_base + k * K + j] = 1; }
                    }
                }
            }
        }
        std::array<double, MaxActiveGroups * K * K> metric_xx_new{};
        std::array<double, MaxActiveGroups * K> metric_xy_new{};
        std::array<double, MaxActiveGroups> metric_ywy_new{};
        std::array<double, MaxActiveGroups> metric_wy_new{};
        std::array<double, MaxActiveGroups> metric_weight_new{};
        std::array<double, MaxActiveGroups> metric_weight_square_new{};
        std::array<std::uint8_t, MaxActiveGroups> metric_valid{};
        if constexpr (NeedsMetrics) {
            for (std::size_t lane = 0; lane < N; ++lane) {
                const double weight = weights_by_lane[lane];
                const double y = y_by_lane[lane];
                const auto& features = features_by_lane[lane];
                if (
                    !(weight > 0.0) || !std::isfinite(weight) ||
                    !std::isfinite(y) || !ridge_detail::finite_vector(features)
                ) continue;
                const std::size_t active = active_index_by_lane[lane];
                const std::size_t matrix_base = active * K * K;
                const std::size_t vector_base = active * K;
                metric_valid[active] = 1;
                metric_weight_new[active] += weight;
                metric_weight_square_new[active] = std::fma(
                    weight, weight, metric_weight_square_new[active]
                );
                metric_wy_new[active] = std::fma(
                    weight, y, metric_wy_new[active]
                );
                metric_ywy_new[active] = std::fma(
                    weight * y, y, metric_ywy_new[active]
                );
                for (std::size_t j = 0; j < K; ++j) {
                    metric_xy_new[vector_base + j] = std::fma(
                        weight * features[j], y,
                        metric_xy_new[vector_base + j]
                    );
                    for (std::size_t k = 0; k < K; ++k) {
                        metric_xx_new[matrix_base + j * K + k] = std::fma(
                            weight * features[j], features[k],
                            metric_xx_new[matrix_base + j * K + k]
                        );
                    }
                }
            }
        }
        if constexpr (Stateful && PredProjection) {
            Projections::for_each(
                [&]<class ProjectionOut, class ProjectionType>() noexcept {
                    if constexpr (
                        ridge_detail::projection_traits<
                            ProjectionType
                        >::predicts
                    ) {
                        auto* projection_out =
                            ctx.template write_ptr<ProjectionOut>();
                        for (std::size_t lane = 0; lane < N; ++lane) {
                            if (!prediction_valid[lane]) {
                                projection_out[lane] = kNaN;
                                continue;
                            }
                            const double prediction = ridge_detail::dot(
                                features_by_lane[lane],
                                state.beta.data()
                                    + static_cast<std::size_t>(
                                        lane_groups[lane]
                                    ) * K
                            );
                            projection_out[lane] = std::is_same_v<
                                ProjectionType, RidgeResidualsProjection
                            > ? y_by_lane[lane] - prediction : prediction;
                        }
                    }
                }
            );
        }
        std::array<std::array<double, K>, MaxActiveGroups> solved_betas{};
        std::array<std::array<double, K>, MaxActiveGroups> standard_errors{};
        std::array<std::array<double, K>, MaxActiveGroups> tstats{};
        std::array<double, MaxActiveGroups> sse_values{};
        std::array<double, MaxActiveGroups> sst_values{};
        std::array<double, MaxActiveGroups> r2_values{};
        std::array<double, MaxActiveGroups> residual_variances{};
        std::array<double, MaxActiveGroups> effective_df_values{};
        std::array<double, MaxActiveGroups> effective_n_values{};
        if constexpr (NeedsMetrics) {
            for (auto& values : standard_errors) values.fill(kNaN);
            for (auto& values : tstats) values.fill(kNaN);
            sse_values.fill(kNaN);
            sst_values.fill(kNaN);
            r2_values.fill(kNaN);
            residual_variances.fill(kNaN);
            effective_df_values.fill(kNaN);
            effective_n_values.fill(kNaN);
        }
        for (std::size_t active = 0; active < active_count; ++active) {
            const std::size_t group = active_groups[active];
            const std::size_t local_matrix = active * K * K;
            const std::size_t local_vector = active * K;
            const bool recompute = recompute_schedule.due(group);
            std::array<double, K * K> xx{};
            std::array<double, K> xy{}, fallback{};
            if constexpr (Stateful) {
                const std::size_t state_matrix = group * K * K, state_vector = group * K;
                for (std::size_t j = 0; j < K; ++j) fallback[j] = state.beta[state_vector + j];
                if (all_finite && state.full_synced[group]) {
                    // Missing samples freeze each statistic's observation clock.
                    const double decay = alpha;
                    for (std::size_t j = 0; j < K; ++j) {
                        const std::size_t sj = state_vector + j;
                        state.xy[sj] = std::fma(decay, xy_new[local_vector + j] - state.xy[sj], state.xy[sj]);
                        state.last_xy[sj] = state.t;
                        xy[j] = state.xy[sj];
                        for (std::size_t k = 0; k < K; ++k) {
                            const std::size_t sjk = state_matrix + j * K + k;
                            state.xx[sjk] = std::fma(decay, xx_new[local_matrix + j * K + k] - state.xx[sjk], state.xx[sjk]);
                            state.last_xx[sjk] = state.t;
                            xx[j * K + k] = state.xx[sjk];
                        }
                    }
                } else {
                    for (std::size_t j = 0; j < K; ++j) {
                        const std::size_t lj = local_vector + j, sj = state_vector + j;
                        if (all_finite || xy_valid[lj]) {
                            if (state.has_xy[sj]) { const double decay = alpha; state.xy[sj] = std::fma(decay, xy_new[lj] - state.xy[sj], state.xy[sj]); }
                            else state.xy[sj] = xy_new[lj];
                            state.has_xy[sj] = 1; state.last_xy[sj] = state.t;
                        }
                        xy[j] = state.xy[sj];
                        for (std::size_t k = 0; k < K; ++k) {
                            const std::size_t ljk = local_matrix + j * K + k, sjk = state_matrix + j * K + k;
                            if (all_finite || xx_valid[ljk]) {
                                if (state.has_xx[sjk]) { const double decay = alpha; state.xx[sjk] = std::fma(decay, xx_new[ljk] - state.xx[sjk], state.xx[sjk]); }
                                else state.xx[sjk] = xx_new[ljk];
                                state.has_xx[sjk] = 1; state.last_xx[sjk] = state.t;
                            }
                            xx[j * K + k] = state.xx[sjk];
                        }
                    }
                }
                if (all_finite) { state.full_synced[group] = 1; state.last_full[group] = state.t; }
                else state.full_synced[group] = 0;
            } else {
                for (std::size_t j = 0; j < K; ++j) {
                    xy[j] = all_finite || xy_valid[local_vector + j] ? xy_new[local_vector + j] : 0.0;
                    for (std::size_t k = 0; k < K; ++k) { const std::size_t index = local_matrix + j * K + k; xx[j * K + k] = all_finite || xx_valid[index] ? xx_new[index] : 0.0; }
                }
            }
            std::array<double, K * K> metric_xx{};
            std::array<double, K> metric_xy{};
            double metric_ywy = 0.0;
            double metric_wy = 0.0;
            double metric_weight = 0.0;
            double metric_weight_square = 0.0;
            bool metric_ready = false;
            if constexpr (NeedsMetrics) {
                if constexpr (Stateful) {
                    const std::size_t state_matrix = group * K * K;
                    const std::size_t state_vector = group * K;
                    if (metric_valid[active]) {
                        if (metrics.initialized[group]) {
                            const double update = alpha;
                            const double old_factor = 1.0 - update;
                            for (std::size_t j = 0; j < K; ++j) {
                                metrics.xy[state_vector + j] = std::fma(
                                    update,
                                    metric_xy_new[local_vector + j]
                                        - metrics.xy[state_vector + j],
                                    metrics.xy[state_vector + j]
                                );
                                for (std::size_t k = 0; k < K; ++k) {
                                    const std::size_t state_index = state_matrix + j * K + k;
                                    metrics.xx[state_index] = std::fma(
                                        update,
                                        metric_xx_new[local_matrix + j * K + k]
                                            - metrics.xx[state_index],
                                        metrics.xx[state_index]
                                    );
                                }
                            }
                            metrics.ywy[group] = std::fma(
                                update,
                                metric_ywy_new[active] - metrics.ywy[group],
                                metrics.ywy[group]
                            );
                            metrics.wy[group] = std::fma(
                                update,
                                metric_wy_new[active] - metrics.wy[group],
                                metrics.wy[group]
                            );
                            metrics.weight[group] = std::fma(
                                update,
                                metric_weight_new[active] - metrics.weight[group],
                                metrics.weight[group]
                            );
                            metrics.weight_square[group] =
                                old_factor * old_factor * metrics.weight_square[group]
                                + update * update * metric_weight_square_new[active];
                        } else {
                            for (std::size_t j = 0; j < K; ++j) {
                                metrics.xy[state_vector + j] = metric_xy_new[local_vector + j];
                                for (std::size_t k = 0; k < K; ++k) {
                                    metrics.xx[state_matrix + j * K + k] =
                                        metric_xx_new[local_matrix + j * K + k];
                                }
                            }
                            metrics.ywy[group] = metric_ywy_new[active];
                            metrics.wy[group] = metric_wy_new[active];
                            metrics.weight[group] = metric_weight_new[active];
                            metrics.weight_square[group] = metric_weight_square_new[active];
                            metrics.initialized[group] = 1;
                        }
                        metrics.last_update[group] = state.t;
                    }
                    metric_ready = metrics.initialized[group];
                    if (metric_ready) {
                        for (std::size_t j = 0; j < K; ++j) {
                            metric_xy[j] = metrics.xy[state_vector + j];
                            for (std::size_t k = 0; k < K; ++k) {
                                metric_xx[j * K + k] =
                                    metrics.xx[state_matrix + j * K + k];
                            }
                        }
                        metric_ywy = metrics.ywy[group];
                        metric_wy = metrics.wy[group];
                        metric_weight = metrics.weight[group];
                        metric_weight_square = metrics.weight_square[group];
                    }
                } else if (metric_valid[active]) {
                    metric_ready = true;
                    for (std::size_t j = 0; j < K; ++j) {
                        metric_xy[j] = metric_xy_new[local_vector + j];
                        for (std::size_t k = 0; k < K; ++k) {
                            metric_xx[j * K + k] =
                                metric_xx_new[local_matrix + j * K + k];
                        }
                    }
                    metric_ywy = metric_ywy_new[active];
                    metric_wy = metric_wy_new[active];
                    metric_weight = metric_weight_new[active];
                    metric_weight_square = metric_weight_square_new[active];
                }
            }
            if constexpr (RecomputeEvery > 1) {
                if (!recompute && result_cache.initialized[group]) {
                    result_cache.load(
                        group,
                        solved_betas[active],
                        standard_errors[active],
                        tstats[active],
                        sse_values[active],
                        sst_values[active],
                        r2_values[active],
                        residual_variances[active],
                        effective_df_values[active],
                        effective_n_values[active]
                    );
                    continue;
                }
            }
            std::array<double, K * K> system = xx;
            for (std::size_t j = 0; j < K; ++j) {
                system[j * K + j] = std::fma(
                    ridge_lambda,
                    xx[j * K + j],
                    xx[j * K + j]
                );
            }
            auto& beta = solved_betas[active];
            bool solved = false;
            if constexpr (Nonnegative) {
                if constexpr (Stateful) {
                    solved = ridge_detail::coordinate_nonnegative_solve(
                        system, xy, fallback, beta
                    );
                } else {
                    solved = ridge_detail::nnqp_nonnegative_solve(
                        system, xy, fallback, beta
                    );
                }
            } else {
                solved = ridge_detail::unconstrained_solve(
                    system, xy, beta
                );
            }
            if (!solved) beta = fallback;
            if constexpr (Stateful) for (std::size_t j = 0; j < K; ++j) state.beta[group * K + j] = beta[j];
            if constexpr (NeedsMetrics) {
                if (metric_ready && metric_weight > 0.0 && metric_weight_square > 0.0) {
                    double beta_xy = 0.0;
                    double beta_xx_beta = 0.0;
                    for (std::size_t j = 0; j < K; ++j) {
                        beta_xy = std::fma(beta[j], metric_xy[j], beta_xy);
                        double row_value = 0.0;
                        for (std::size_t k = 0; k < K; ++k) {
                            row_value = std::fma(
                                metric_xx[j * K + k], beta[k], row_value
                            );
                        }
                        beta_xx_beta = std::fma(beta[j], row_value, beta_xx_beta);
                    }
                    const double raw_sse = metric_ywy - 2.0 * beta_xy + beta_xx_beta;
                    const double sse = raw_sse > 0.0 ? raw_sse : 0.0;
                    const double sst = metric_ywy - metric_wy * metric_wy / metric_weight;
                    const double effective_n = metric_weight * metric_weight /
                        metric_weight_square;
                    sse_values[active] = sse;
                    sst_values[active] = sst > 0.0 ? sst : 0.0;
                    r2_values[active] = sst > 0.0 ? 1.0 - sse / sst : kNaN;
                    effective_n_values[active] = effective_n;

                    if constexpr (NeedsInference && !Nonnegative) {
                        std::array<double, K * K> inverse{};
                        if (ridge_detail::inverse<K>(system, inverse)) {
                            std::array<double, K * K> hat_core{};
                            for (std::size_t row = 0; row < K; ++row) {
                                for (std::size_t column = 0; column < K; ++column) {
                                    double value = 0.0;
                                    for (std::size_t inner = 0; inner < K; ++inner) {
                                        value = std::fma(
                                            inverse[row * K + inner],
                                            metric_xx[inner * K + column],
                                            value
                                        );
                                    }
                                    hat_core[row * K + column] = value;
                                }
                            }
                            double effective_df = 0.0;
                            double hat_square_trace = 0.0;
                            for (std::size_t row = 0; row < K; ++row) {
                                effective_df += hat_core[row * K + row];
                                for (std::size_t column = 0; column < K; ++column) {
                                    hat_square_trace = std::fma(
                                        hat_core[row * K + column],
                                        hat_core[column * K + row],
                                        hat_square_trace
                                    );
                                }
                            }
                            effective_df_values[active] = effective_df;
                            const double residual_df = effective_n
                                - 2.0 * effective_df + hat_square_trace;
                            const double residual_variance = residual_df > 0.0
                                ? sse / residual_df
                                : kNaN;
                            residual_variances[active] = residual_variance;
                            if (std::isfinite(residual_variance)) {
                                for (std::size_t coefficient = 0; coefficient < K; ++coefficient) {
                                    double covariance_diagonal = 0.0;
                                    for (std::size_t left = 0; left < K; ++left) {
                                        for (std::size_t right = 0; right < K; ++right) {
                                            covariance_diagonal = std::fma(
                                                inverse[coefficient * K + left]
                                                    * metric_xx[left * K + right],
                                                inverse[coefficient * K + right],
                                                covariance_diagonal
                                            );
                                        }
                                    }
                                    const double standard_error = std::sqrt(
                                        std::max(0.0, residual_variance * covariance_diagonal)
                                    );
                                    standard_errors[active][coefficient] = standard_error;
                                    tstats[active][coefficient] = standard_error > 0.0
                                        ? beta[coefficient] / standard_error
                                        : kNaN;
                                }
                            }
                        }
                    }
                }
            }
            if constexpr (RecomputeEvery > 1) {
                result_cache.store(
                    group,
                    beta,
                    standard_errors[active],
                    tstats[active],
                    sse_values[active],
                    sst_values[active],
                    r2_values[active],
                    residual_variances[active],
                    effective_df_values[active],
                    effective_n_values[active]
                );
            }

        }
        Projections::for_each(
            [&]<class ProjectionOut, class ProjectionType>() noexcept {
                using Traits = ridge_detail::projection_traits<ProjectionType>;
                if constexpr (Traits::predicts) {
                    if constexpr (!Stateful) {
                        auto* projection_out =
                            ctx.template write_ptr<ProjectionOut>();
                        for (std::size_t lane = 0; lane < N; ++lane) {
                            if (!prediction_valid[lane]) {
                                projection_out[lane] = kNaN;
                                continue;
                            }
                            const double prediction = ridge_detail::dot(
                                features_by_lane[lane],
                                solved_betas[
                                    active_index_by_lane[lane]
                                ].data()
                            );
                            projection_out[lane] = std::is_same_v<
                                ProjectionType, RidgeResidualsProjection
                            > ? y_by_lane[lane] - prediction : prediction;
                        }
                    }
                } else if constexpr (Traits::full_coefficients) {
                    auto* projection_out =
                        ctx.template write_ptr<ProjectionOut>();
                    if constexpr (Groups == 1) {
                        for (std::size_t j = 0; j < K; ++j) {
                            if constexpr (std::is_same_v<
                                ProjectionType, RidgeBetaProjection
                            >) {
                                projection_out[j] = solved_betas[0][j];
                            } else if constexpr (std::is_same_v<
                                ProjectionType, RidgeStandardErrorsProjection
                            >) {
                                projection_out[j] = standard_errors[0][j];
                            } else {
                                projection_out[j] = tstats[0][j];
                            }
                        }
                    } else {
                        for (std::size_t lane = 0; lane < N; ++lane) {
                            const std::size_t active =
                                active_index_by_lane[lane];
                            for (std::size_t j = 0; j < K; ++j) {
                                if constexpr (std::is_same_v<
                                    ProjectionType, RidgeBetaProjection
                                >) {
                                    projection_out[lane * K + j] =
                                        solved_betas[active][j];
                                } else if constexpr (std::is_same_v<
                                    ProjectionType,
                                    RidgeStandardErrorsProjection
                                >) {
                                    projection_out[lane * K + j] =
                                        standard_errors[active][j];
                                } else {
                                    projection_out[lane * K + j] =
                                        tstats[active][j];
                                }
                            }
                        }
                    }
                } else if constexpr (
                    ridge_detail::projection_component<ProjectionType>::value
                    != std::numeric_limits<std::size_t>::max()
                ) {
                    constexpr std::size_t component =
                        ridge_detail::projection_component<
                            ProjectionType
                        >::value;
                    static_assert(component < K);
                    auto* projection_out =
                        ctx.template write_ptr<ProjectionOut>();
                    auto projected = [&](std::size_t active) {
                        if constexpr (
                            ridge_detail::is_coefficient_projection<
                                ProjectionType
                            >::value
                        ) {
                            return solved_betas[active][component];
                        } else if constexpr (
                            ridge_detail::is_standard_error_projection<
                                ProjectionType
                            >::value
                        ) {
                            return standard_errors[active][component];
                        } else {
                            return tstats[active][component];
                        }
                    };
                    if constexpr (Groups == 1) {
                        projection_out[0] = projected(0);
                    } else {
                        for (std::size_t lane = 0; lane < N; ++lane) {
                            projection_out[lane] = projected(
                                active_index_by_lane[lane]
                            );
                        }
                    }
                } else {
                    auto* projection_out =
                        ctx.template write_ptr<ProjectionOut>();
                    const auto& values = [&]() -> const auto& {
                        if constexpr (std::is_same_v<
                            ProjectionType, RidgeSseProjection
                        >) return sse_values;
                        else if constexpr (std::is_same_v<
                            ProjectionType, RidgeSstProjection
                        >) return sst_values;
                        else if constexpr (std::is_same_v<
                            ProjectionType, RidgeR2Projection
                        >) return r2_values;
                        else if constexpr (std::is_same_v<
                            ProjectionType, RidgeResidualVarianceProjection
                        >) return residual_variances;
                        else if constexpr (std::is_same_v<
                            ProjectionType, RidgeEffectiveDfProjection
                        >) return effective_df_values;
                        else return effective_n_values;
                    }();
                    if constexpr (Groups == 1) {
                        projection_out[0] = values[0];
                    } else {
                        for (std::size_t lane = 0; lane < N; ++lane) {
                            projection_out[lane] = values[
                                active_index_by_lane[lane]
                            ];
                        }
                    }
                }
            }
        );
        if constexpr (Stateful) ++state.t;
        recompute_schedule.next_row();
    }
};

}  // namespace stackdsl
