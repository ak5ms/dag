#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include <Eigen/Cholesky>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>

#if defined(STACKDSL_EIGEN_RUNTIME_NO_MALLOC) && defined(EIGEN_NO_DEBUG)
#error "Eigen runtime allocation assertions require EIGEN_NO_DEBUG to be absent"
#endif

namespace stackdsl::eigen_detail {

template <std::size_t K>
using Matrix = Eigen::Matrix<
    double,
    static_cast<int>(K),
    static_cast<int>(K),
    Eigen::RowMajor
>;

template <std::size_t K>
using Vector = Eigen::Matrix<double, static_cast<int>(K), 1>;

template <std::size_t K>
using ConstMatrixMap = Eigen::Map<const Matrix<K>>;

template <std::size_t K>
using MatrixMap = Eigen::Map<Matrix<K>>;

template <std::size_t K>
using ConstVectorMap = Eigen::Map<const Vector<K>>;

template <std::size_t K>
using VectorMap = Eigen::Map<Vector<K>>;

struct MallocAuditGuard {
    MallocAuditGuard() noexcept {
#if defined(STACKDSL_EIGEN_RUNTIME_NO_MALLOC)
        Eigen::internal::set_is_malloc_allowed(false);
#endif
    }
    ~MallocAuditGuard() {
#if defined(STACKDSL_EIGEN_RUNTIME_NO_MALLOC)
        Eigen::internal::set_is_malloc_allowed(true);
#endif
    }
};

template <std::size_t K>
inline bool solve_matrix(
    const Matrix<K>& matrix,
    const Vector<K>& rhs,
    Vector<K>& solution
) noexcept {
    static_assert(Matrix<K>::SizeAtCompileTime != Eigen::Dynamic);
    static_assert(Vector<K>::SizeAtCompileTime != Eigen::Dynamic);
    MallocAuditGuard allocation_audit;
    if (!matrix.allFinite() || !rhs.allFinite()) return false;

    // All matrices are fixed-size. noalias() prevents avoidable expression
    // temporaries; decomposition workspaces remain stack-resident.
    Eigen::LLT<Matrix<K>> llt(matrix);
    if (llt.info() == Eigen::Success) {
        solution.noalias() = llt.solve(rhs);
        if (llt.info() == Eigen::Success && solution.allFinite()) return true;
    }

    Eigen::FullPivLU<Matrix<K>> lu(matrix);
    const double scale = matrix.cwiseAbs().maxCoeff();
    lu.setThreshold(std::max(1e-15, scale * 1e-12));
    if (lu.isInvertible()) {
        solution.noalias() = lu.solve(rhs);
        if (solution.allFinite()) return true;
    }

    Eigen::SelfAdjointEigenSolver<Matrix<K>> eig(matrix);
    if (eig.info() != Eigen::Success) return false;
    const auto& eigenvalues = eig.eigenvalues();
    const double max_eigenvalue = eigenvalues.cwiseAbs().maxCoeff();
    const double tolerance = std::max(1e-15, max_eigenvalue * 1e-12);
    Vector<K> projected;
    projected.noalias() = eig.eigenvectors().transpose() * rhs;
    for (std::size_t i = 0; i < K; ++i) {
        const double value = eigenvalues[static_cast<int>(i)];
        projected[static_cast<int>(i)] =
            std::abs(value) > tolerance
                ? projected[static_cast<int>(i)] / value
                : 0.0;
    }
    solution.noalias() = eig.eigenvectors() * projected;
    return solution.allFinite();
}

template <std::size_t K>
[[gnu::cold, gnu::noinline]] bool solve_unconstrained(
    const std::array<double, K * K>& system,
    const std::array<double, K>& rhs,
    std::array<double, K>& solution
) noexcept {
    const ConstMatrixMap<K> matrix(system.data());
    const ConstVectorMap<K> values(rhs.data());
    Vector<K> solved;
    if (!solve_matrix<K>(matrix, values, solved)) return false;
    VectorMap<K>(solution.data()) = solved;
    return true;
}

template <std::size_t K>
inline double dot(
    const std::array<double, K>& values,
    const double* beta
) noexcept {
    return ConstVectorMap<K>(values.data()).dot(ConstVectorMap<K>(beta));
}

}  // namespace stackdsl::eigen_detail
