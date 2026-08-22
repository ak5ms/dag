#pragma once

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>

#include "stackdsl/ops/einsum.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <class Src, class Out, class Shape, std::uint64_t FloorBits, class Execution>
class PsdFactorNode {
    static_assert(Shape::rank == 2, "psd_factor requires a matrix");
    static_assert(Shape::dims[0] == Shape::dims[1], "psd_factor requires square input");
    static constexpr std::size_t N = Shape::dims[0];
    static constexpr double floor_value = std::bit_cast<double>(FloorBits);

    using Matrix = Eigen::Matrix<double, N, N, Eigen::RowMajor>;
    Matrix covariance_{};
    Matrix repaired_{};
    Eigen::LLT<Matrix> llt_{};
    Eigen::SelfAdjointEigenSolver<Matrix> eigen_{};

public:
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    void on_data(Context& ctx) noexcept {
        Src::load_contiguous(ctx, 0, N * N, covariance_.data());
        for (std::size_t row = 0; row < N; ++row) {
            for (std::size_t col = 0; col < N; ++col) {
                double lhs = covariance_(row, col);
                double rhs = covariance_(col, row);
                if (!std::isfinite(lhs)) lhs = row == col ? floor_value : 0.0;
                if (!std::isfinite(rhs)) rhs = row == col ? floor_value : 0.0;
                repaired_(row, col) = 0.5 * (lhs + rhs);
            }
            repaired_(row, row) = std::max(repaired_(row, row), floor_value);
        }

        llt_.compute(repaired_);
        if (llt_.info() != Eigen::Success) {
            eigen_.compute(repaired_);
            auto values = eigen_.eigenvalues();
            for (Eigen::Index index = 0; index < values.size(); ++index) {
                values(index) = std::max(values(index), floor_value);
            }
            repaired_.noalias() = eigen_.eigenvectors()
                * values.asDiagonal()
                * eigen_.eigenvectors().transpose();
            llt_.compute(repaired_);
        }

        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        if (llt_.info() != Eigen::Success) {
            std::fill(out, out + N * N, kNaN);
            return;
        }
        const Matrix lower = llt_.matrixL();
        std::copy(lower.data(), lower.data() + N * N, out);
    }
};

}  // namespace stackdsl
