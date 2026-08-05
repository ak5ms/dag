#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

template <std::size_t Components>
struct LaneState {
    std::array<double, Components> moment{};
    double weight = 0.0;
    std::int64_t count = 0;
    bool initialized = false;
};

template <bool Kurtosis, bool Adjust, bool IgnoreNa>
void run_kernel(
    const double* __restrict x,
    const double* __restrict y,
    double* __restrict out,
    std::size_t rows,
    std::size_t lanes,
    double span,
    std::int64_t min_periods
) noexcept {
    constexpr std::size_t Components = Kurtosis ? 8 : 6;
    std::vector<LaneState<Components>> state(lanes);
    const double alpha = 2.0 / (span + 1.0);
    const double old_factor = 1.0 - alpha;
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t row = 0; row < rows; ++row) {
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const std::size_t offset = row * lanes + lane;
            const double xv = x[offset];
            const double yv = y[offset];
            const bool observation = std::isfinite(xv) && std::isfinite(yv);
            auto& current = state[lane];
            double old_weight = current.weight;
            if (current.initialized && (observation || !IgnoreNa)) {
                old_weight *= old_factor;
            }
            if (observation) {
                const double x2 = xv * xv;
                const double x3 = x2 * xv;
                const std::array<double, Components> next = [&] {
                    if constexpr (Kurtosis) {
                        return std::array<double, Components>{
                            xv, x2, x3, yv, yv * xv, yv * x2,
                            yv * x3, yv * yv
                        };
                    } else {
                        return std::array<double, Components>{
                            xv, x2, yv, yv * xv, yv * x2, yv * yv
                        };
                    }
                }();
                if (current.initialized) {
                    if constexpr (!Adjust && IgnoreNa) {
                        for (std::size_t item = 0; item < Components; ++item) {
                            current.moment[item] = std::fma(
                                alpha,
                                next[item] - current.moment[item],
                                current.moment[item]
                            );
                        }
                        old_weight = 1.0;
                    } else {
                        double new_weight = Adjust ? 1.0 : alpha;
                        if constexpr (!Adjust) {
                            if (std::abs(alpha - 0.5) <= 1e-12) {
                                new_weight = 1.0 - old_weight;
                            }
                        }
                        const double denominator = old_weight + new_weight;
                        for (std::size_t item = 0; item < Components; ++item) {
                            if (current.moment[item] != next[item]) {
                                current.moment[item] = (
                                    old_weight * current.moment[item]
                                    + new_weight * next[item]
                                ) / denominator;
                            }
                        }
                        old_weight = Adjust ? denominator : 1.0;
                    }
                } else {
                    current.moment = next;
                    current.initialized = true;
                    old_weight = 1.0;
                }
                ++current.count;
            }
            current.weight = old_weight;
            if (!current.initialized || current.count < min_periods) {
                out[offset] = nan;
                continue;
            }
            const double mx = current.moment[0];
            const double x2 = current.moment[1];
            const double my = current.moment[Kurtosis ? 3 : 2];
            const double y2 = current.moment[Kurtosis ? 7 : 5];
            const double variance_x = std::max(0.0, x2 - mx * mx);
            const double variance_y = std::max(0.0, y2 - my * my);
            double central;
            double denominator;
            if constexpr (Kurtosis) {
                const double x3 = current.moment[2];
                const double yx = current.moment[4];
                const double yx2 = current.moment[5];
                const double yx3 = current.moment[6];
                central = yx3 - 3.0 * mx * yx2 + 3.0 * mx * mx * yx
                    - mx * mx * mx * my
                    - my * (x3 - 3.0 * mx * x2 + 2.0 * mx * mx * mx);
                denominator = std::sqrt(variance_y) * variance_x
                    * std::sqrt(variance_x);
            } else {
                const double yx = current.moment[3];
                const double yx2 = current.moment[4];
                central = yx2 - 2.0 * mx * yx - my * x2
                    + 2.0 * my * mx * mx;
                denominator = std::sqrt(variance_y) * variance_x;
            }
            out[offset] = denominator > 0.0 && std::isfinite(denominator)
                ? central / denominator
                : nan;
        }
    }
}

}  // namespace

extern "C" void ewm_co_skew_recursive_ceiling(
    const double* x,
    const double* y,
    double* out,
    std::size_t rows,
    std::size_t lanes,
    double span
) noexcept {
    run_kernel<false, false, true>(x, y, out, rows, lanes, span, 0);
}

extern "C" void ewm_co_kurt_recursive_ceiling(
    const double* x,
    const double* y,
    double* out,
    std::size_t rows,
    std::size_t lanes,
    double span
) noexcept {
    run_kernel<true, false, true>(x, y, out, rows, lanes, span, 0);
}

extern "C" void ewm_co_skew_adjusted_ceiling(
    const double* x,
    const double* y,
    double* out,
    std::size_t rows,
    std::size_t lanes,
    double span,
    std::int64_t min_periods
) noexcept {
    run_kernel<false, true, false>(
        x, y, out, rows, lanes, span, min_periods
    );
}
