#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#ifndef CPP_STREAM_CEILING_N
#define CPP_STREAM_CEILING_N 9
#endif

#if defined(__GNUC__) || defined(__clang__)
#define CEILING_RESTRICT __restrict__
#else
#define CEILING_RESTRICT
#endif

namespace {
constexpr std::size_t kLanes = CPP_STREAM_CEILING_N;

struct RunningStats {
    std::uint64_t count = 0;
    double total = 0.0;
    double mean = 0.0;
    double m2 = 0.0;

    inline void add(double value) noexcept {
        if (!std::isfinite(value)) return;
        total += value;
        const std::uint64_t next = count + 1;
        const double delta = value - mean;
        mean += delta / static_cast<double>(next);
        const double delta2 = value - mean;
        m2 = std::fma(delta, delta2, m2);
        count = next;
    }

    inline double std0() const noexcept {
        if (count == 0) return std::numeric_limits<double>::quiet_NaN();
        return std::sqrt(std::max(0.0, m2 / static_cast<double>(count)));
    }
};

inline bool valid_width(std::size_t lanes, double* out) noexcept {
    if (lanes == kLanes) return true;
    if (out != nullptr) out[0] = std::numeric_limits<double>::quiet_NaN();
    return false;
}
}  // namespace

extern "C" void column_stats_ceiling(
    const double* CEILING_RESTRICT x,
    std::size_t rows,
    std::size_t lanes,
    double* CEILING_RESTRICT out
) noexcept {
    if (!valid_width(lanes, out)) return;
    std::array<RunningStats, kLanes> stats{};
    for (std::size_t row = 0; row < rows; ++row) {
        const double* CEILING_RESTRICT current = x + row * kLanes;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            stats[lane].add(current[lane]);
        }
    }
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
    for (std::size_t lane = 0; lane < kLanes; ++lane) {
        const RunningStats& item = stats[lane];
        out[lane * 3] = item.count == 0
            ? std::numeric_limits<double>::quiet_NaN()
            : item.total;
        out[lane * 3 + 1] = item.count == 0
            ? std::numeric_limits<double>::quiet_NaN()
            : item.total / static_cast<double>(item.count);
        out[lane * 3 + 2] = item.std0();
    }
}

extern "C" void stateless_sharpe_ceiling(
    const double* CEILING_RESTRICT x,
    const double* CEILING_RESTRICT y,
    std::size_t rows,
    std::size_t lanes,
    double* CEILING_RESTRICT out
) noexcept {
    if (!valid_width(lanes, out)) return;
    RunningStats stats{};
    for (std::size_t row = 0; row < rows; ++row) {
        const double* CEILING_RESTRICT left = x + row * kLanes;
        const double* CEILING_RESTRICT right = y + row * kLanes;
        double pnl = 0.0;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            pnl = std::fma(left[lane], right[lane], pnl);
        }
        stats.add(pnl);
    }
    const double denominator = stats.std0();
    out[0] = stats.count == 0
        ? std::numeric_limits<double>::quiet_NaN()
        : (stats.total / static_cast<double>(stats.count)) / denominator;
}

extern "C" void shifted_alpha_sharpe_ceiling(
    const double* CEILING_RESTRICT alpha,
    const double* CEILING_RESTRICT returns,
    std::size_t rows,
    std::size_t lanes,
    double* CEILING_RESTRICT out
) noexcept {
    if (!valid_width(lanes, out)) return;
    RunningStats stats{};
    if (rows > 0) stats.add(0.0);
    for (std::size_t row = 1; row < rows; ++row) {
        const double* CEILING_RESTRICT weight = alpha + (row - 1) * kLanes;
        const double* CEILING_RESTRICT ret = returns + row * kLanes;
        double pnl = 0.0;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t lane = 0; lane < kLanes; ++lane) {
            const double value = weight[lane];
            if (std::isfinite(value) && std::isfinite(ret[lane])) {
                pnl = std::fma(value, ret[lane], pnl);
            }
        }
        stats.add(pnl);
    }
    const double denominator = stats.std0();
    out[0] = stats.count == 0
        ? std::numeric_limits<double>::quiet_NaN()
        : (stats.total / static_cast<double>(stats.count)) / denominator;
}
