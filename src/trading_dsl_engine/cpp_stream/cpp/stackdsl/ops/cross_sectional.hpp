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
#include "stackdsl/utils.hpp"

namespace stackdsl {

struct XsCountProjection {};
struct XsSumProjection {};
struct XsMeanProjection {};
struct XsStdProjection {};
struct XsMinProjection {};
struct XsMaxProjection {};
struct XsQuantileProjection {};

struct XsValueItem {
    std::uint32_t group;
    double value;
    std::uint32_t lane;
};

template <
    std::size_t N,
    class In,
    class Out,
    class Projection,
    std::uint64_t QuantileBits = 0,
    class Execution = DirectExecution<N>
>
struct XsAggregateNode {
    static constexpr std::size_t Groups = Execution::cross_state_size;
    static constexpr double quantile = std::bit_cast<double>(QuantileBits);
    static_assert(
        !std::is_same_v<Projection, XsQuantileProjection>
        || (quantile >= 0.0 && quantile <= 1.0)
    );

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        std::array<std::uint32_t, Groups> count{};
        std::array<double, Groups> total{};
        std::array<double, Groups> mean{};
        std::array<double, Groups> m2{};
        std::array<double, Groups> result{};
        std::array<XsValueItem, N> items{};
        std::size_t item_count = 0;
        result.fill(kNaN);

        for (std::size_t lane = 0; lane < N; ++lane) {
            const double value = ctx.template read<In>(lane);
            if (!finite(value)) continue;
            const std::size_t group = Execution::cross_group(ctx, lane);
            const std::uint32_t next_count = count[group] + 1;
            if constexpr (std::is_same_v<Projection, XsStdProjection>) {
                const double delta = value - mean[group];
                mean[group] += delta / static_cast<double>(next_count);
                m2[group] = std::fma(delta, value - mean[group], m2[group]);
            } else if constexpr (std::is_same_v<Projection, XsMinProjection>) {
                total[group] = count[group] == 0
                    ? value
                    : std::min(total[group], value);
            } else if constexpr (std::is_same_v<Projection, XsMaxProjection>) {
                total[group] = count[group] == 0
                    ? value
                    : std::max(total[group], value);
            } else if constexpr (std::is_same_v<Projection, XsQuantileProjection>) {
                items[item_count++] = {
                    static_cast<std::uint32_t>(group),
                    value,
                    static_cast<std::uint32_t>(lane),
                };
            } else {
                total[group] += value;
            }
            count[group] = next_count;
        }

        if constexpr (std::is_same_v<Projection, XsQuantileProjection>) {
            std::sort(
                items.begin(),
                items.begin() + static_cast<std::ptrdiff_t>(item_count),
                [](const XsValueItem& left, const XsValueItem& right) {
                    return left.group < right.group
                        || (left.group == right.group && left.value < right.value);
                }
            );
            std::size_t begin = 0;
            while (begin < item_count) {
                std::size_t end = begin + 1;
                while (end < item_count && items[end].group == items[begin].group) {
                    ++end;
                }
                const std::size_t size = end - begin;
                const double position = quantile * static_cast<double>(size - 1);
                const std::size_t lower = static_cast<std::size_t>(position);
                const std::size_t upper = std::min(size - 1, lower + 1);
                const double lower_value = items[begin + lower].value;
                const double upper_value = items[begin + upper].value;
                result[items[begin].group] = lower_value
                    + (position - static_cast<double>(lower))
                        * (upper_value - lower_value);
                begin = end;
            }
        } else {
            for (std::size_t group = 0; group < Groups; ++group) {
                if constexpr (std::is_same_v<Projection, XsCountProjection>) {
                    result[group] = static_cast<double>(count[group]);
                } else if (count[group] != 0) {
                    if constexpr (std::is_same_v<Projection, XsSumProjection>) {
                        result[group] = total[group];
                    } else if constexpr (std::is_same_v<Projection, XsMeanProjection>) {
                        result[group] = total[group] / static_cast<double>(count[group]);
                    } else if constexpr (std::is_same_v<Projection, XsStdProjection>) {
                        result[group] = std::sqrt(
                            std::max(0.0, m2[group])
                            / static_cast<double>(count[group])
                        );
                    } else {
                        result[group] = total[group];
                    }
                }
            }
        }

        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            out[lane] = result[Execution::cross_group(ctx, lane)];
        }
    }
};

template <
    std::size_t N,
    class In,
    class Weight,
    class Out,
    class Execution = DirectExecution<N>
>
struct XsWeightedMeanNode {
    static constexpr std::size_t Groups = Execution::cross_state_size;
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        std::array<double, Groups> weighted{};
        std::array<double, Groups> weight_sum{};
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double value = ctx.template read<In>(lane);
            const double weight = ctx.template read<Weight>(lane);
            if (!finite(value) || !finite(weight)) continue;
            const std::size_t group = Execution::cross_group(ctx, lane);
            weighted[group] = std::fma(value, weight, weighted[group]);
            weight_sum[group] += weight;
        }
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const std::size_t group = Execution::cross_group(ctx, lane);
            out[lane] = weight_sum[group] != 0.0
                ? weighted[group] / weight_sum[group]
                : kNaN;
        }
    }
};

template <
    std::size_t N,
    class Target,
    class Regressor,
    class Out,
    bool Intercept,
    class Execution = DirectExecution<N>
>
struct XsProjectionNode {
    static constexpr std::size_t Groups = Execution::cross_state_size;
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        std::array<double, Groups> sx{};
        std::array<double, Groups> sy{};
        std::array<double, Groups> sxx{};
        std::array<double, Groups> sxy{};
        std::array<std::uint32_t, Groups> count{};
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double y = ctx.template read<Target>(lane);
            const double x = ctx.template read<Regressor>(lane);
            if (!finite(x) || !finite(y)) continue;
            const std::size_t group = Execution::cross_group(ctx, lane);
            sx[group] += x;
            sy[group] += y;
            sxx[group] = std::fma(x, x, sxx[group]);
            sxy[group] = std::fma(x, y, sxy[group]);
            ++count[group];
        }
        std::array<double, Groups> alpha{};
        std::array<double, Groups> beta{};
        std::array<std::uint8_t, Groups> valid{};
        for (std::size_t group = 0; group < Groups; ++group) {
            if constexpr (Intercept) {
                if (count[group] < 2) continue;
                const double n = static_cast<double>(count[group]);
                const double denominator = sxx[group] - sx[group] * sx[group] / n;
                if (!(denominator > 0.0)) continue;
                beta[group] = (sxy[group] - sx[group] * sy[group] / n)
                    / denominator;
                alpha[group] = sy[group] / n - beta[group] * sx[group] / n;
            } else {
                if (count[group] == 0 || !(sxx[group] > 0.0)) continue;
                beta[group] = sxy[group] / sxx[group];
            }
            valid[group] = 1;
        }
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            const double x = ctx.template read<Regressor>(lane);
            const std::size_t group = Execution::cross_group(ctx, lane);
            out[lane] = valid[group] && finite(x)
                ? std::fma(beta[group], x, alpha[group])
                : kNaN;
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    std::uint64_t PowerBits,
    class Execution = DirectExecution<N>
>
struct XsGeneralizedRankNode {
    static constexpr double power = std::bit_cast<double>(PowerBits);
    static_assert(power >= 0.0);
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        std::array<double, N> values{};
        std::array<std::uint32_t, N> groups{};
        for (std::size_t lane = 0; lane < N; ++lane) {
            values[lane] = ctx.template read<In>(lane);
            groups[lane] = Execution::cross_group(ctx, lane);
        }
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            if (!finite(values[lane])) {
                out[lane] = kNaN;
                continue;
            }
            double score = 0.0;
            std::size_t count = 0;
            for (std::size_t other = 0; other < N; ++other) {
                if (!finite(values[other]) || groups[other] != groups[lane]) continue;
                const double difference = values[lane] - values[other];
                if (difference > 0.0) score += std::pow(difference, power);
                else if (difference < 0.0) score -= std::pow(-difference, power);
                ++count;
            }
            out[lane] = count
                ? score / static_cast<double>(count)
                : kNaN;
        }
    }
};

template <
    std::size_t N,
    class In,
    class Out,
    class Execution = DirectExecution<N>
>
struct XsDensifyNode {
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        std::array<XsValueItem, N> items{};
        std::size_t count = 0;
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double value = ctx.template read<In>(lane);
            if (finite(value)) {
                items[count++] = {
                    Execution::cross_group(ctx, lane),
                    value,
                    static_cast<std::uint32_t>(lane),
                };
            } else {
                out[lane] = kNaN;
            }
        }
        std::sort(
            items.begin(),
            items.begin() + static_cast<std::ptrdiff_t>(count),
            [](const XsValueItem& left, const XsValueItem& right) {
                return left.group < right.group
                    || (left.group == right.group && left.value < right.value);
            }
        );
        std::size_t begin = 0;
        while (begin < count) {
            std::size_t end = begin + 1;
            while (end < count && items[end].group == items[begin].group) ++end;
            std::size_t unique = 0;
            for (std::size_t position = begin; position < end; ++position) {
                if (
                    position > begin
                    && items[position].value != items[position - 1].value
                ) {
                    ++unique;
                }
                out[items[position].lane] = static_cast<double>(unique);
            }
            begin = end;
        }
    }
};

}  // namespace stackdsl
