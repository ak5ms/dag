#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "stackdsl/ops/cross_sectional.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

// Python lowers xs_gauss through XsGeneralizedRankOp with power=-0.0.  Signed
// zero is mathematically identical to ordinary power=0 but has a distinct bit
// pattern, giving codegen a zero-overhead compile-time tag without adding a
// second cross-sectional lowering path.
inline constexpr std::uint64_t kXsGaussPowerTag = 0x8000000000000000ULL;

template <
    std::size_t N,
    class In,
    class Out,
    class Execution
>
struct XsGeneralizedRankNode<N, In, Out, kXsGaussPowerTag, Execution> {
    static constexpr std::size_t Groups = Execution::cross_state_size;

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        std::array<XsValueItem, N> items{};
        std::array<double, N> levels{};
        std::array<double, N> raw{};
        std::array<double, N> values{};
        raw.fill(kNaN);

        std::size_t item_count = 0;
        for (std::size_t lane = 0; lane < N; ++lane) {
            const double value = ctx.template read<In>(lane);
            values[lane] = value;
            if (!finite(value)) continue;
            items[item_count++] = {
                static_cast<std::uint32_t>(Execution::cross_group(ctx, lane)),
                value,
                static_cast<std::uint32_t>(lane),
            };
        }

        // The probability grid depends only on sorted absolute magnitudes.
        std::sort(
            items.begin(),
            items.begin() + static_cast<std::ptrdiff_t>(item_count),
            [](const XsValueItem& left, const XsValueItem& right) {
                if (left.group != right.group) return left.group < right.group;
                const double left_abs = std::abs(left.value);
                const double right_abs = std::abs(right.value);
                if (left_abs != right_abs) return left_abs < right_abs;
                return left.lane < right.lane;
            }
        );

        std::size_t group_begin = 0;
        while (group_begin < item_count) {
            std::size_t group_end = group_begin + 1;
            while (
                group_end < item_count
                && items[group_end].group == items[group_begin].group
            ) {
                ++group_end;
            }

            long double total = 0.0L;
            for (std::size_t position = group_begin; position < group_end; ++position) {
                total += static_cast<long double>(std::abs(items[position].value));
            }
            const long double min_abs = static_cast<long double>(
                std::abs(items[group_begin].value)
            );
            const long double max_abs = static_cast<long double>(
                std::abs(items[group_end - 1].value)
            );
            const long double denominator = total + 0.5L * (min_abs + max_abs);

            if (!(denominator > 0.0L)) {
                // All-zero cross sections have no ordering information and zero
                // variance; their final normalized score is defined as zero.
                for (std::size_t position = group_begin; position < group_end; ++position) {
                    levels[position] = 0.5;
                }
            } else {
                long double cumulative = 0.0L;
                for (std::size_t position = group_begin; position < group_end; ++position) {
                    cumulative += static_cast<long double>(
                        std::abs(items[position].value)
                    );
                    levels[position] = static_cast<double>(cumulative / denominator);
                }

                // Zero magnitudes otherwise create q=0 and norm_inv(q)=-inf.
                // Interpolate the zero-mass levels evenly inside (0, q_first+),
                // leaving every ordinary nonzero probability unchanged.
                std::size_t first_positive = group_begin;
                while (
                    first_positive < group_end
                    && std::abs(items[first_positive].value) == 0.0
                ) {
                    ++first_positive;
                }
                if (first_positive > group_begin && first_positive < group_end) {
                    const long double first_level =
                        static_cast<long double>(std::abs(items[first_positive].value))
                        / denominator;
                    const std::size_t zero_count = first_positive - group_begin;
                    for (std::size_t index = 0; index < zero_count; ++index) {
                        levels[group_begin + index] = static_cast<double>(
                            first_level * static_cast<long double>(index + 1)
                            / static_cast<long double>(zero_count + 1)
                        );
                    }
                }
            }
            group_begin = group_end;
        }

        // Assign the magnitude-derived grid by ascending x rank.  Equal values
        // use their upper occupied rank, matching xs_rank/xs_pct_rank semantics.
        std::sort(
            items.begin(),
            items.begin() + static_cast<std::ptrdiff_t>(item_count),
            [](const XsValueItem& left, const XsValueItem& right) {
                if (left.group != right.group) return left.group < right.group;
                if (left.value != right.value) return left.value < right.value;
                return left.lane < right.lane;
            }
        );

        group_begin = 0;
        while (group_begin < item_count) {
            std::size_t group_end = group_begin + 1;
            while (
                group_end < item_count
                && items[group_end].group == items[group_begin].group
            ) {
                ++group_end;
            }
            std::size_t tie_begin = group_begin;
            while (tie_begin < group_end) {
                std::size_t tie_end = tie_begin + 1;
                while (
                    tie_end < group_end
                    && items[tie_end].value == items[tie_begin].value
                ) {
                    ++tie_end;
                }
                double q = levels[tie_end - 1];
                if (!(q > 0.0)) q = std::numeric_limits<double>::min();
                if (!(q < 1.0)) q = std::nextafter(1.0, 0.0);
                const double score = norm_inv(q);
                for (std::size_t position = tie_begin; position < tie_end; ++position) {
                    raw[items[position].lane] = score;
                }
                tie_begin = tie_end;
            }
            group_begin = group_end;
        }

        // Population standard deviation of the raw Gaussian scores.  Do not
        // subtract the mean from the output: xs_gauss is z / std(z), exactly.
        std::array<double, Groups> mean{};
        std::array<double, Groups> m2{};
        std::array<std::uint32_t, Groups> count{};
        for (std::size_t lane = 0; lane < N; ++lane) {
            if (!finite(raw[lane])) continue;
            const std::size_t group = Execution::cross_group(ctx, lane);
            const std::uint32_t next_count = count[group] + 1;
            const double delta = raw[lane] - mean[group];
            mean[group] += delta / static_cast<double>(next_count);
            m2[group] = std::fma(delta, raw[lane] - mean[group], m2[group]);
            count[group] = next_count;
        }

        std::array<double, Groups> scale{};
        for (std::size_t group = 0; group < Groups; ++group) {
            if (count[group] != 0) {
                scale[group] = std::sqrt(
                    std::max(0.0, m2[group]) / static_cast<double>(count[group])
                );
            }
        }

        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            if (!finite(values[lane])) {
                out[lane] = kNaN;
                continue;
            }
            const double stddev = scale[Execution::cross_group(ctx, lane)];
            out[lane] = stddev > 0.0 && finite(stddev)
                ? raw[lane] / stddev
                : 0.0;
        }
    }
};

}  // namespace stackdsl
