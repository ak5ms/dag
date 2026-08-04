#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "stackdsl/ops/eigen_solvers.hpp"

namespace stackdsl::nnqp {

inline constexpr double kTolerance = 1e-10;

template <std::size_t K>
inline bool solve_restricted(
    const eigen_detail::Matrix<K>& matrix,
    const eigen_detail::Vector<K>& rhs,
    const std::array<std::uint8_t, K>& free,
    eigen_detail::Vector<K>& beta
) noexcept {
    eigen_detail::Matrix<K> restricted = matrix;
    eigen_detail::Vector<K> restricted_rhs = rhs;
    bool any_free = false;
    for (std::size_t i = 0; i < K; ++i) {
        if (free[i]) {
            any_free = true;
            continue;
        }
        restricted.row(static_cast<int>(i)).setZero();
        restricted.col(static_cast<int>(i)).setZero();
        restricted(static_cast<int>(i), static_cast<int>(i)) = 1.0;
        restricted_rhs[static_cast<int>(i)] = 0.0;
    }
    if (!any_free) {
        beta.setZero();
        return true;
    }
    return eigen_detail::solve_matrix<K>(restricted, restricted_rhs, beta);
}

template <std::size_t K>
inline bool solve(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    const std::array<double, K>& fallback,
    std::array<double, K>& solution
) noexcept {
    const eigen_detail::ConstMatrixMap<K> matrix_map(system.data());
    const eigen_detail::ConstVectorMap<K> rhs_map(rhs.data());
    if (!matrix_map.allFinite() || !rhs_map.allFinite()) return false;

    const eigen_detail::Matrix<K> matrix = matrix_map;
    const eigen_detail::Vector<K> values = rhs_map;
    std::array<std::uint8_t, K> free{};
    eigen_detail::Vector<K> unconstrained;
    const bool unconstrained_ok =
        eigen_detail::solve_matrix<K>(matrix, values, unconstrained);
    for (std::size_t i = 0; i < K; ++i) {
        free[i] = static_cast<std::uint8_t>(
            unconstrained_ok && unconstrained[static_cast<int>(i)] > 0.0
        );
    }

    eigen_detail::Vector<K> beta =
        eigen_detail::ConstVectorMap<K>(fallback.data()).cwiseMax(0.0);
    eigen_detail::Vector<K> trial;
    eigen_detail::Vector<K> gradient;

    constexpr std::size_t max_iterations = 64;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        if (!solve_restricted<K>(matrix, values, free, trial)) return false;

        std::size_t drop = K;
        double most_negative = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            const double value = trial[static_cast<int>(i)];
            if (free[i] && value < -kTolerance && (drop == K || value < most_negative)) {
                drop = i;
                most_negative = value;
            }
        }
        if (drop != K) {
            free[drop] = 0;
            continue;
        }

        beta = trial.cwiseMax(0.0);
        gradient.noalias() = matrix * beta;
        gradient.noalias() -= values;

        std::size_t add = K;
        double minimum_gradient = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            const double value = gradient[static_cast<int>(i)];
            if (!free[i] && value < -kTolerance && (add == K || value < minimum_gradient)) {
                add = i;
                minimum_gradient = value;
            }
        }
        if (add == K) {
            eigen_detail::VectorMap<K>(solution.data()) = beta;
            return beta.allFinite();
        }
        free[add] = 1;
    }

    eigen_detail::VectorMap<K>(solution.data()) = beta.cwiseMax(0.0);
    return beta.allFinite();
}

}  // namespace stackdsl::nnqp
