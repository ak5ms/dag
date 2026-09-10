#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "stackdsl/ops/eigen_solvers.hpp"

namespace stackdsl::nnqp {

inline constexpr double kTolerance = 1e-10;

template <std::size_t K>
inline bool solve_rank_aware(
    const eigen_detail::Matrix<K>& matrix,
    const eigen_detail::Vector<K>& rhs,
    eigen_detail::Vector<K>& beta
) noexcept {
    eigen_detail::MallocAuditGuard allocation_audit;
    if (!matrix.allFinite() || !rhs.allFinite()) return false;

    Eigen::LLT<eigen_detail::Matrix<K>> llt(matrix);
    double normalized_determinant = 1.0;
    if (llt.info() == Eigen::Success) {
        const auto factor = llt.matrixL();
        for (std::size_t i = 0; i < K; ++i) {
            const double diagonal = factor.coeff(
                static_cast<int>(i),
                static_cast<int>(i)
            );
            const double pivot = diagonal * diagonal;
            const double input_diagonal = matrix(
                static_cast<int>(i),
                static_cast<int>(i)
            );
            normalized_determinant *= input_diagonal > 0.0
                ? pivot / input_diagonal
                : 0.0;
        }
        if (normalized_determinant > 1e-12) {
            beta.noalias() = llt.solve(rhs);
            if (llt.info() == Eigen::Success && beta.allFinite()) return true;
        }
    } else {
        normalized_determinant = 0.0;
    }

    Eigen::SelfAdjointEigenSolver<eigen_detail::Matrix<K>> eig(matrix);
    if (eig.info() != Eigen::Success) return false;
    const auto& eigenvalues = eig.eigenvalues();
    const double maximum_eigenvalue = eigenvalues.maxCoeff();
    if (!(maximum_eigenvalue > 0.0)) {
        beta.setZero();
        return true;
    }
    const double tolerance = maximum_eigenvalue * 1e-12;
    eigen_detail::Vector<K> projected;
    projected.noalias() = eig.eigenvectors().transpose() * rhs;
    for (std::size_t i = 0; i < K; ++i) {
        const double eigenvalue = eigenvalues[static_cast<int>(i)];
        projected[static_cast<int>(i)] = eigenvalue > tolerance
            ? projected[static_cast<int>(i)] / eigenvalue
            : 0.0;
    }
    beta.noalias() = eig.eigenvectors() * projected;
    return beta.allFinite();
}

template <std::size_t K>
inline bool solve_restricted(
    const eigen_detail::Matrix<K>& matrix,
    const eigen_detail::Vector<K>& rhs,
    const std::array<std::uint8_t, K>& free,
    eigen_detail::Vector<K>& beta
) noexcept {
    bool any_free = false;
    double active_scale = 0.0;
    for (std::size_t i = 0; i < K; ++i) {
        if (!free[i]) continue;
        any_free = true;
        active_scale = std::max(
            active_scale,
            std::abs(matrix(static_cast<int>(i), static_cast<int>(i)))
        );
    }
    if (!any_free) {
        beta.setZero();
        return true;
    }
    eigen_detail::Matrix<K> restricted = matrix;
    eigen_detail::Vector<K> restricted_rhs = rhs;
    const double inactive_diagonal = active_scale > 0.0 ? active_scale : 1.0;
    for (std::size_t i = 0; i < K; ++i) {
        if (free[i]) continue;
        restricted.row(static_cast<int>(i)).setZero();
        restricted.col(static_cast<int>(i)).setZero();
        restricted(static_cast<int>(i), static_cast<int>(i)) =
            inactive_diagonal;
        restricted_rhs[static_cast<int>(i)] = 0.0;
    }
    return solve_rank_aware<K>(restricted, restricted_rhs, beta);
}

template <std::size_t K>
inline bool solve_iterative(
    const eigen_detail::Matrix<K>& matrix,
    const eigen_detail::Vector<K>& values,
    const std::array<double, K>& fallback,
    std::array<double, K>& solution
) noexcept {
    std::array<std::uint8_t, K> free{};
    eigen_detail::Vector<K> unconstrained;
    const bool unconstrained_ok =
        solve_rank_aware<K>(matrix, values, unconstrained);
    bool unconstrained_feasible = unconstrained_ok;
    for (std::size_t i = 0; i < K; ++i) {
        unconstrained_feasible = unconstrained_feasible
            && unconstrained[static_cast<int>(i)] >= -kTolerance;
        free[i] = static_cast<std::uint8_t>(
            unconstrained_ok && unconstrained[static_cast<int>(i)] > 0.0
        );
    }
    if (unconstrained_feasible) {
        eigen_detail::Vector<K> residual = matrix * unconstrained;
        residual -= values;
        if (residual.cwiseAbs().maxCoeff() <= kTolerance) {
            eigen_detail::VectorMap<K>(solution.data()) =
                unconstrained.cwiseMax(0.0);
            return true;
        }
    }

    eigen_detail::Vector<K> beta =
        eigen_detail::ConstVectorMap<K>(fallback.data()).cwiseMax(0.0);
    eigen_detail::Vector<K> trial;
    eigen_detail::Vector<K> gradient;

    constexpr std::size_t max_iterations = 64;
    for (std::size_t iteration = 0; iteration < max_iterations; ++iteration) {
        if (!solve_restricted<K>(matrix, values, free, trial)) return false;

        double most_negative = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            const double value = trial[static_cast<int>(i)];
            if (free[i] && value < most_negative) most_negative = value;
        }
        if (most_negative < -kTolerance) {
            for (std::size_t i = 0; i < K; ++i) {
                if (
                    free[i]
                    && trial[static_cast<int>(i)]
                        <= most_negative + kTolerance
                ) {
                    free[i] = 0;
                }
            }
            continue;
        }

        beta = trial.cwiseMax(0.0);
        gradient.noalias() = matrix * beta;
        gradient.noalias() -= values;

        double minimum_gradient = 0.0;
        for (std::size_t i = 0; i < K; ++i) {
            const double value = gradient[static_cast<int>(i)];
            if (!free[i] && value < minimum_gradient) minimum_gradient = value;
        }
        if (minimum_gradient >= -kTolerance) {
            eigen_detail::VectorMap<K>(solution.data()) = beta;
            return beta.allFinite();
        }
        for (std::size_t i = 0; i < K; ++i) {
            if (
                !free[i]
                && gradient[static_cast<int>(i)]
                    <= minimum_gradient + kTolerance
            ) {
                free[i] = 1;
            }
        }
    }

    eigen_detail::VectorMap<K>(solution.data()) = beta.cwiseMax(0.0);
    return beta.allFinite();
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
    return solve_iterative<K>(matrix, values, fallback, solution);
}

}  // namespace stackdsl::nnqp
