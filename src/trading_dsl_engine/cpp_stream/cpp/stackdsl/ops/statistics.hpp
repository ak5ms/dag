#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "stackdsl/engine.hpp"
#include "stackdsl/ops/cat.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

namespace stats_detail {

STACKDSL_HOT double variance(double first, double second) noexcept {
    const double value = second - first * first;
    return value > 0.0 ? value : 0.0;
}

STACKDSL_HOT double ratio(double numerator, double denominator) noexcept {
    return denominator > 0.0 && std::isfinite(denominator)
        ? numerator / denominator
        : kNaN;
}

template <std::size_t Arity>
STACKDSL_HOT double monomial(
    const std::array<double, Arity>& inputs,
    const std::array<std::uint8_t, 3>& powers
) noexcept {
    double result = 1.0;
    for (std::size_t input = 0; input < Arity; ++input) {
        for (std::uint8_t power = 0; power < powers[input]; ++power) {
            result *= inputs[input];
        }
    }
    return result;
}

template <std::size_t Order>
struct EwmMomentProjection {
    static_assert(Order >= 1 && Order <= 4);
    static constexpr std::size_t arity = 1;
    static constexpr std::size_t term_count = Order;
    static constexpr auto powers = [] {
        std::array<std::array<std::uint8_t, 3>, term_count> result{};
        for (std::size_t order = 0; order < term_count; ++order) {
            result[order][0] = static_cast<std::uint8_t>(order + 1);
        }
        return result;
    }();
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        const double mean = raw[0];
        if constexpr (Order == 1) return 0.0;
        if constexpr (Order == 2) return variance(mean, raw[1]);
        if constexpr (Order == 3) {
            return raw[2] - 3.0 * mean * raw[1] + 2.0 * mean * mean * mean;
        }
        return raw[3] - 4.0 * mean * raw[2]
            + 6.0 * mean * mean * raw[1]
            - 3.0 * mean * mean * mean * mean;
    }
};

struct EwmVarianceProjection : EwmMomentProjection<2> {};

struct EwmStdProjection : EwmMomentProjection<2> {
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        return std::sqrt(EwmVarianceProjection::project(raw));
    }
};

struct EwmSkewnessProjection : EwmMomentProjection<3> {
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        const double var = variance(raw[0], raw[1]);
        const double central = EwmMomentProjection<3>::project(raw);
        return ratio(central, var * std::sqrt(var));
    }
};

struct EwmKurtosisProjection : EwmMomentProjection<4> {
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        const double var = variance(raw[0], raw[1]);
        return ratio(EwmMomentProjection<4>::project(raw), var * var);
    }
};

struct EwmCovarianceProjection {
    static constexpr std::size_t arity = 2;
    static constexpr std::size_t term_count = 3;
    static constexpr std::array<std::array<std::uint8_t, 3>, term_count> powers{{
        {{1, 0, 0}}, {{0, 1, 0}}, {{1, 1, 0}},
    }};
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        return raw[2] - raw[0] * raw[1];
    }
};

struct EwmCorrelationProjection {
    static constexpr std::size_t arity = 2;
    static constexpr std::size_t term_count = 5;
    static constexpr std::array<std::array<std::uint8_t, 3>, term_count> powers{{
        {{1, 0, 0}}, {{0, 1, 0}}, {{2, 0, 0}}, {{0, 2, 0}}, {{1, 1, 0}},
    }};
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        const double vx = variance(raw[0], raw[2]);
        const double vy = variance(raw[1], raw[3]);
        return ratio(raw[4] - raw[0] * raw[1], std::sqrt(vx * vy));
    }
};

// Inputs follow the source definition: (y, x).
struct EwmCoSkewnessProjection {
    static constexpr std::size_t arity = 2;
    static constexpr std::size_t term_count = 6;
    static constexpr std::array<std::array<std::uint8_t, 3>, term_count> powers{{
        {{1, 0, 0}}, {{0, 1, 0}}, {{2, 0, 0}}, {{0, 2, 0}},
        {{1, 1, 0}}, {{1, 2, 0}},
    }};
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        const double my = raw[0], mx = raw[1];
        const double vy = variance(my, raw[2]), vx = variance(mx, raw[3]);
        const double central = raw[5] - 2.0 * mx * raw[4]
            - my * raw[3] + 2.0 * my * mx * mx;
        return ratio(central, std::sqrt(vy) * vx);
    }
};

// Inputs follow the source definition: (y, x).
struct EwmCoKurtosisProjection {
    static constexpr std::size_t arity = 2;
    static constexpr std::size_t term_count = 9;
    static constexpr std::array<std::array<std::uint8_t, 3>, term_count> powers{{
        {{1, 0, 0}}, {{0, 1, 0}}, {{2, 0, 0}}, {{0, 2, 0}},
        {{1, 1, 0}}, {{0, 3, 0}}, {{1, 2, 0}}, {{1, 3, 0}}, {{0, 4, 0}},
    }};
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        const double my = raw[0], mx = raw[1];
        const double vy = variance(my, raw[2]), vx = variance(mx, raw[3]);
        const double central = raw[7] - 3.0 * mx * raw[6]
            + 3.0 * mx * mx * raw[4] - my * raw[5]
            + 3.0 * my * mx * raw[3] - 3.0 * my * mx * mx * mx;
        return ratio(central, std::sqrt(vy) * vx * std::sqrt(vx));
    }
};

struct EwmTripleCorrelationProjection {
    static constexpr std::size_t arity = 3;
    static constexpr std::size_t term_count = 10;
    static constexpr std::array<std::array<std::uint8_t, 3>, term_count> powers{{
        {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}},
        {{2, 0, 0}}, {{0, 2, 0}}, {{0, 0, 2}},
        {{1, 1, 0}}, {{1, 0, 1}}, {{0, 1, 1}}, {{1, 1, 1}},
    }};
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        const double mx = raw[0], my = raw[1], mz = raw[2];
        const double vx = variance(mx, raw[3]);
        const double vy = variance(my, raw[4]);
        const double vz = variance(mz, raw[5]);
        const double central = raw[9] - mx * raw[8] - my * raw[7]
            - mz * raw[6] + 2.0 * mx * my * mz;
        return ratio(central, std::sqrt(vx * vy * vz));
    }
};

struct EwmPartialCorrelationProjection {
    static constexpr std::size_t arity = 3;
    static constexpr std::size_t term_count = 9;
    static constexpr std::array<std::array<std::uint8_t, 3>, term_count> powers{{
        {{1, 0, 0}}, {{0, 1, 0}}, {{0, 0, 1}},
        {{2, 0, 0}}, {{0, 2, 0}}, {{0, 0, 2}},
        {{1, 1, 0}}, {{1, 0, 1}}, {{0, 1, 1}},
    }};
    STACKDSL_HOT static double project(
        const std::array<double, term_count>& raw
    ) noexcept {
        const double vx = variance(raw[0], raw[3]);
        const double vy = variance(raw[1], raw[4]);
        const double vz = variance(raw[2], raw[5]);
        const double rxy = ratio(raw[6] - raw[0] * raw[1], std::sqrt(vx * vy));
        const double rxz = ratio(raw[7] - raw[0] * raw[2], std::sqrt(vx * vz));
        const double ryz = ratio(raw[8] - raw[1] * raw[2], std::sqrt(vy * vz));
        return ratio(
            rxy - rxz * ryz,
            std::sqrt(std::max(0.0, 1.0 - rxz * rxz))
                * std::sqrt(std::max(0.0, 1.0 - ryz * ryz))
        );
    }
};

}  // namespace stats_detail

using stats_detail::EwmCoKurtosisProjection;
using stats_detail::EwmCoSkewnessProjection;
using stats_detail::EwmCorrelationProjection;
using stats_detail::EwmCovarianceProjection;
using stats_detail::EwmKurtosisProjection;
using stats_detail::EwmMomentProjection;
using stats_detail::EwmPartialCorrelationProjection;
using stats_detail::EwmSkewnessProjection;
using stats_detail::EwmStdProjection;
using stats_detail::EwmTripleCorrelationProjection;
using stats_detail::EwmVarianceProjection;

template <
    std::size_t N,
    class Features,
    class Out,
    std::uint64_t AlphaBits,
    int MinPeriods,
    class Projection,
    class Execution = DirectExecution<N>
>
struct EwmStatsNode;

template <
    std::size_t N,
    class Out,
    std::uint64_t AlphaBits,
    int MinPeriods,
    class Projection,
    class Execution,
    class... Sources
>
struct EwmStatsNode<
    N, FeatureList<Sources...>, Out, AlphaBits, MinPeriods, Projection, Execution
> {
    static constexpr std::size_t Arity = sizeof...(Sources);
    static constexpr std::size_t Terms = Projection::term_count;
    static_assert(Arity == Projection::arity);
    static_assert(MinPeriods >= 0);
    static constexpr double alpha = std::bit_cast<double>(AlphaBits);
    static_assert(alpha > 0.0 && alpha <= 1.0);

    alignas(64) std::array<double, Execution::state_size * Terms> raw{};
    alignas(64) std::array<std::uint64_t, Execution::state_size> count{};
    alignas(64) std::array<std::uint8_t, Execution::state_size> initialized{};

    void setup() noexcept {
        raw.fill(0.0);
        count.fill(0);
        initialized.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            std::array<double, Arity> inputs{};
            load_features(ctx, lane, inputs, FeatureList<Sources...>{});
            bool valid = true;
            for (double input : inputs) valid = valid && finite(input);
            if (valid) {
                for (std::size_t term = 0; term < Terms; ++term) {
                    const double observation = stats_detail::monomial(
                        inputs, Projection::powers[term]
                    );
                    double& value = raw[index * Terms + term];
                    value = initialized[index]
                        ? std::fma(alpha, observation - value, value)
                        : observation;
                }
                initialized[index] = 1;
                ++count[index];
            }
            if (
                !initialized[index] ||
                (MinPeriods > 0 && count[index] < static_cast<std::uint64_t>(MinPeriods))
            ) {
                out[lane] = kNaN;
                continue;
            }
            std::array<double, Terms> values{};
            for (std::size_t term = 0; term < Terms; ++term) {
                values[term] = raw[index * Terms + term];
            }
            out[lane] = Projection::project(values);
        }
    }
};

struct RollingSumProjection {};
struct RollingMeanProjection {};
struct RollingStdProjection {};

template <
    std::size_t N,
    class In,
    class Out,
    std::size_t Periods,
    std::size_t MinPeriods,
    std::size_t Ddof,
    class Projection,
    class Execution = DirectExecution<N>
>
struct RollingMomentsNode {
    static_assert(Periods > 0 && MinPeriods <= Periods);
    static constexpr std::size_t StateSize = Execution::state_size;
    alignas(64) std::array<double, StateSize * Periods> ring{};
    alignas(64) std::array<double, StateSize> mean{};
    alignas(64) std::array<double, StateSize> m2{};
    alignas(64) std::array<std::uint32_t, StateSize> valid_count{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        ring.fill(kNaN);
        mean.fill(0.0);
        m2.fill(0.0);
        valid_count.fill(0);
        step.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const std::size_t position = index * Periods + step[index] % Periods;
            const double outgoing = ring[position];
            if (finite(outgoing)) remove(index, outgoing);
            const double incoming = ctx.template read<In>(lane);
            ring[position] = incoming;
            if (finite(incoming)) add(index, incoming);
            ++step[index];
            const std::size_t count = valid_count[index];
            if (count < MinPeriods || count == 0) {
                if constexpr (std::is_same_v<Projection, RollingSumProjection>) {
                    out[lane] = MinPeriods == 0 ? 0.0 : kNaN;
                } else out[lane] = kNaN;
            } else if constexpr (std::is_same_v<Projection, RollingSumProjection>) {
                out[lane] = mean[index] * static_cast<double>(count);
            } else if constexpr (std::is_same_v<Projection, RollingMeanProjection>) {
                out[lane] = mean[index];
            } else {
                out[lane] = count > Ddof
                    ? std::sqrt(std::max(0.0, m2[index]) / static_cast<double>(count - Ddof))
                    : kNaN;
            }
        }
    }

private:
    STACKDSL_HOT void add(std::size_t index, double value) noexcept {
        const double count = static_cast<double>(valid_count[index] + 1);
        const double delta = value - mean[index];
        mean[index] += delta / count;
        m2[index] = std::fma(delta, value - mean[index], m2[index]);
        ++valid_count[index];
    }

    STACKDSL_HOT void remove(std::size_t index, double value) noexcept {
        const std::uint32_t count = valid_count[index];
        if (count <= 1) {
            valid_count[index] = 0;
            mean[index] = 0.0;
            m2[index] = 0.0;
            return;
        }
        const double next_mean = (
            static_cast<double>(count) * mean[index] - value
        ) / static_cast<double>(count - 1);
        m2[index] -= (value - mean[index]) * (value - next_mean);
        if (m2[index] < 0.0 && m2[index] > -1e-12) m2[index] = 0.0;
        mean[index] = next_mean;
        --valid_count[index];
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    std::size_t Periods,
    std::size_t MinPeriods,
    bool IsMax,
    bool ReturnArg,
    class Execution = DirectExecution<N>
>
struct RollingExtremaNode {
    static_assert(Periods > 0 && MinPeriods <= Periods);
    static constexpr std::size_t StateSize = Execution::state_size;
    alignas(64) std::array<double, StateSize * Periods> deque_values{};
    alignas(64) std::array<std::uint64_t, StateSize * Periods> deque_steps{};
    alignas(64) std::array<std::uint8_t, StateSize * Periods> ring_valid{};
    alignas(64) std::array<std::uint32_t, StateSize> head{};
    alignas(64) std::array<std::uint32_t, StateSize> size{};
    alignas(64) std::array<std::uint32_t, StateSize> valid_count{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        deque_values.fill(0.0);
        deque_steps.fill(0);
        ring_valid.fill(0);
        head.fill(0);
        size.fill(0);
        valid_count.fill(0);
        step.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const std::uint64_t now = step[index];
            const std::size_t ring_position = index * Periods + now % Periods;
            if (ring_valid[ring_position]) --valid_count[index];
            const double value = ctx.template read<In>(lane);
            ring_valid[ring_position] = static_cast<std::uint8_t>(finite(value));
            expire(index, now);
            if (finite(value)) {
                ++valid_count[index];
                while (size[index] > 0) {
                    const std::size_t back = physical(index, size[index] - 1);
                    const bool dominated = IsMax
                        ? deque_values[back] <= value
                        : deque_values[back] >= value;
                    if (!dominated) break;
                    --size[index];
                }
                const std::size_t tail = physical(index, size[index]);
                deque_values[tail] = value;
                deque_steps[tail] = now;
                ++size[index];
            }
            if (valid_count[index] < MinPeriods || size[index] == 0) {
                out[lane] = kNaN;
            } else {
                const std::size_t front = physical(index, 0);
                out[lane] = ReturnArg
                    ? static_cast<double>(now - deque_steps[front])
                    : deque_values[front];
            }
            ++step[index];
        }
    }

private:
    STACKDSL_HOT std::size_t physical(
        std::size_t index,
        std::size_t logical
    ) const noexcept {
        return index * Periods + (head[index] + logical) % Periods;
    }

    STACKDSL_HOT void expire(std::size_t index, std::uint64_t now) noexcept {
        while (size[index] > 0) {
            const std::size_t front = physical(index, 0);
            if (deque_steps[front] + Periods > now) break;
            head[index] = (head[index] + 1) % Periods;
            --size[index];
        }
    }
};

struct RollingQuantileProjection {};
struct RollingPctRankProjection {};

template <
    std::size_t N,
    class In,
    class Out,
    std::size_t Periods,
    std::size_t MinPeriods,
    std::uint64_t QuantileBits,
    class Projection,
    class Execution = DirectExecution<N>
>
struct RollingOrderNode {
    static_assert(Periods > 0 && MinPeriods <= Periods);
    static constexpr double quantile = std::bit_cast<double>(QuantileBits);
    static_assert(quantile >= 0.0 && quantile <= 1.0);
    static constexpr std::size_t StateSize = Execution::state_size;
    alignas(64) std::array<double, StateSize * Periods> ring{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        ring.fill(kNaN);
        step.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const double current = ctx.template read<In>(lane);
            ring[index * Periods + step[index] % Periods] = current;
            ++step[index];
            std::array<double, Periods> values{};
            std::size_t count = 0;
            for (std::size_t position = 0; position < Periods; ++position) {
                const double value = ring[index * Periods + position];
                if (finite(value)) values[count++] = value;
            }
            if (count < MinPeriods || count == 0) {
                out[lane] = kNaN;
            } else if constexpr (std::is_same_v<Projection, RollingPctRankProjection>) {
                if (!finite(current)) out[lane] = kNaN;
                else {
                    std::size_t upper = 0;
                    for (std::size_t position = 0; position < count; ++position) {
                        upper += static_cast<std::size_t>(values[position] <= current);
                    }
                    out[lane] = static_cast<double>(upper) /
                        static_cast<double>(count + 1);
                }
            } else {
                const double position = quantile * static_cast<double>(count - 1);
                const std::size_t lower = static_cast<std::size_t>(position);
                const std::size_t upper = std::min(count - 1, lower + 1);
                std::nth_element(values.begin(), values.begin() + lower, values.begin() + count);
                const double lower_value = values[lower];
                if (upper == lower) out[lane] = lower_value;
                else {
                    const double upper_value = *std::min_element(
                        values.begin() + lower + 1, values.begin() + count
                    );
                    out[lane] = lower_value + (position - static_cast<double>(lower))
                        * (upper_value - lower_value);
                }
            }
        }
    }
};

namespace stats_detail {

template <std::size_t Periods>
struct TheilPoint {
    double x;
    double y;
};

template <std::size_t Periods>
STACKDSL_HOT double exact_theilsen(
    const std::array<TheilPoint<Periods>, Periods>& points,
    std::size_t count
) noexcept {
    constexpr std::size_t MaxPairs = Periods * (Periods - 1) / 2;
    std::array<double, MaxPairs> slopes{};
    std::size_t slope_count = 0;
    for (std::size_t left = 0; left < count; ++left) {
        for (std::size_t right = left + 1; right < count; ++right) {
            const double dx = points[right].x - points[left].x;
            if (dx == 0.0) continue;
            const double slope = (points[right].y - points[left].y) / dx;
            if (finite(slope)) slopes[slope_count++] = slope;
        }
    }
    if (slope_count == 0) return kNaN;
    const std::size_t upper = slope_count / 2;
    std::nth_element(slopes.begin(), slopes.begin() + upper, slopes.begin() + slope_count);
    const double upper_value = slopes[upper];
    if (slope_count % 2 != 0) return upper_value;
    const double lower_value = *std::max_element(slopes.begin(), slopes.begin() + upper);
    return 0.5 * (lower_value + upper_value);
}

template <std::size_t Periods>
STACKDSL_HOT std::uint64_t count_slopes_le(
    const std::array<TheilPoint<Periods>, Periods>& points,
    std::size_t count,
    double candidate
) noexcept {
    std::array<double, Periods> z{};
    std::array<double, Periods> ordered{};
    for (std::size_t index = 0; index < count; ++index) {
        z[index] = std::fma(-candidate, points[index].x, points[index].y);
        ordered[index] = z[index];
    }
    std::sort(ordered.begin(), ordered.begin() + count);
    std::array<std::uint32_t, Periods + 1> fenwick{};
    auto add = [&](std::size_t position) {
        for (++position; position <= count; position += position & (~position + 1)) {
            ++fenwick[position];
        }
    };
    auto prefix = [&](std::size_t length) {
        std::uint64_t result = 0;
        for (; length > 0; length -= length & (~length + 1)) result += fenwick[length];
        return result;
    };
    std::uint64_t result = 0;
    std::uint64_t processed = 0;
    std::size_t group = 0;
    while (group < count) {
        std::size_t end = group + 1;
        while (end < count && points[end].x == points[group].x) ++end;
        for (std::size_t index = group; index < end; ++index) {
            const std::size_t rank = static_cast<std::size_t>(
                std::lower_bound(ordered.begin(), ordered.begin() + count, z[index])
                    - ordered.begin()
            );
            result += processed - prefix(rank);
        }
        for (std::size_t index = group; index < end; ++index) {
            const std::size_t rank = static_cast<std::size_t>(
                std::lower_bound(ordered.begin(), ordered.begin() + count, z[index])
                    - ordered.begin()
            );
            add(rank);
        }
        processed += end - group;
        group = end;
    }
    return result;
}

template <std::size_t Periods>
STACKDSL_HOT double selected_slope(
    const std::array<TheilPoint<Periods>, Periods>& points,
    std::size_t count,
    std::uint64_t rank,
    double bound
) noexcept {
    double lower = -bound;
    double upper = bound;
    for (std::size_t iteration = 0; iteration < 72; ++iteration) {
        const double middle = lower + 0.5 * (upper - lower);
        if (!(middle > lower && middle < upper)) break;
        if (count_slopes_le(points, count, middle) > rank) upper = middle;
        else lower = middle;
    }
    return upper;
}

template <std::size_t Periods>
STACKDSL_HOT double subquadratic_theilsen(
    std::array<TheilPoint<Periods>, Periods> points,
    std::size_t count
) noexcept {
    std::sort(
        points.begin(), points.begin() + count,
        [](const auto& left, const auto& right) {
            return left.x < right.x || (left.x == right.x && left.y < right.y);
        }
    );
    double min_dx = std::numeric_limits<double>::infinity();
    double min_y = points[0].y, max_y = points[0].y;
    std::uint64_t pairs = 0, prior = 0;
    std::size_t group = 0;
    while (group < count) {
        std::size_t end = group + 1;
        while (end < count && points[end].x == points[group].x) ++end;
        pairs += prior * static_cast<std::uint64_t>(end - group);
        prior += end - group;
        if (end < count) min_dx = std::min(min_dx, points[end].x - points[group].x);
        group = end;
    }
    for (std::size_t index = 1; index < count; ++index) {
        min_y = std::min(min_y, points[index].y);
        max_y = std::max(max_y, points[index].y);
    }
    if (pairs == 0 || !std::isfinite(min_dx) || !(min_dx > 0.0)) return kNaN;
    if (max_y == min_y) return 0.0;
    double bound = (max_y - min_y) / min_dx;
    if (!std::isfinite(bound)) bound = std::numeric_limits<double>::max() / 16.0;
    bound = std::nextafter(bound, std::numeric_limits<double>::infinity());
    const std::uint64_t upper_rank = pairs / 2;
    const double upper = selected_slope(points, count, upper_rank, bound);
    if (pairs % 2 != 0) return upper;
    const double lower = selected_slope(points, count, upper_rank - 1, bound);
    return 0.5 * (lower + upper);
}

}  // namespace stats_detail

template <
    std::size_t N,
    class Y,
    class X,
    class Out,
    std::size_t Periods,
    std::size_t MinPeriods,
    class Execution = DirectExecution<N>
>
struct RollingTheilSenNode {
    static_assert(Periods >= 2 && MinPeriods >= 2 && MinPeriods <= Periods);
    static constexpr std::size_t StateSize = Execution::state_size;
    alignas(64) std::array<double, StateSize * Periods> x_ring{};
    alignas(64) std::array<double, StateSize * Periods> y_ring{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        x_ring.fill(kNaN);
        y_ring.fill(kNaN);
        step.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t index = Execution::state_index(ctx, lane);
            const std::size_t position = index * Periods + step[index] % Periods;
            x_ring[position] = ctx.template read<X>(lane);
            y_ring[position] = ctx.template read<Y>(lane);
            ++step[index];
            std::array<stats_detail::TheilPoint<Periods>, Periods> points{};
            std::size_t count = 0;
            for (std::size_t item = 0; item < Periods; ++item) {
                const double x = x_ring[index * Periods + item];
                const double y = y_ring[index * Periods + item];
                if (finite(x) && finite(y)) points[count++] = {x, y};
            }
            if (count < MinPeriods) out[lane] = kNaN;
            else if constexpr (Periods <= 256) {
                out[lane] = stats_detail::exact_theilsen(points, count);
            } else {
                out[lane] = stats_detail::subquadratic_theilsen(points, count);
            }
        }
    }
};

}  // namespace stackdsl
