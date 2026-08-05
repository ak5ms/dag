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
#include "stackdsl/ops/order_tree.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {


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

template <std::size_t StateSize, std::size_t Periods>
struct RollingOrderScanState {
    alignas(64) std::array<double, StateSize * Periods> ring{};

    void setup() noexcept { ring.fill(kNaN); }
};

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
    static constexpr bool PctRank =
        std::is_same_v<Projection, RollingPctRankProjection>;
    static constexpr bool UseTree =
        (!PctRank && Periods >= 32) || (PctRank && Periods > 2048);
    using OrderState = std::conditional_t<
        UseTree,
        FixedOrderStatisticTree<StateSize, Periods>,
        RollingOrderScanState<StateSize, Periods>
    >;
    OrderState order{};
    alignas(64) std::array<std::uint64_t, StateSize> step{};

    void setup() noexcept {
        order.setup();
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
            const std::uint64_t sequence = step[index];
            if constexpr (UseTree) {
                order.replace(
                    index,
                    static_cast<std::size_t>(sequence % Periods),
                    current,
                    sequence
                );
            } else {
                order.ring[
                    index * Periods + static_cast<std::size_t>(sequence % Periods)
                ] = current;
            }
            ++step[index];
            if constexpr (!UseTree) {
                std::array<double, Periods> values;
                std::size_t count = 0;
                for (std::size_t position = 0; position < Periods; ++position) {
                    const double value = order.ring[index * Periods + position];
                    if (finite(value)) values[count++] = value;
                }
                if (count < MinPeriods || count == 0) {
                    out[lane] = kNaN;
                } else if constexpr (PctRank) {
                    if (!finite(current)) {
                        out[lane] = kNaN;
                    } else {
                        std::size_t upper = 0;
                        for (std::size_t position = 0; position < count; ++position) {
                            upper += static_cast<std::size_t>(
                                values[position] <= current
                            );
                        }
                        out[lane] = static_cast<double>(upper)
                            / static_cast<double>(count + 1);
                    }
                } else {
                    const double position =
                        quantile * static_cast<double>(count - 1);
                    const std::size_t lower = static_cast<std::size_t>(position);
                    const std::size_t upper = std::min(count - 1, lower + 1);
                    std::nth_element(
                        values.begin(), values.begin() + lower,
                        values.begin() + count
                    );
                    const double lower_value = values[lower];
                    if (upper == lower) {
                        out[lane] = lower_value;
                    } else {
                        const double upper_value = *std::min_element(
                            values.begin() + lower + 1,
                            values.begin() + count
                        );
                        out[lane] = lower_value
                            + (position - static_cast<double>(lower))
                                * (upper_value - lower_value);
                    }
                }
                continue;
            }
            if constexpr (UseTree) {
                const std::size_t count = order.size(index);
                if (count < MinPeriods || count == 0) {
                    out[lane] = kNaN;
                } else if constexpr (PctRank) {
                    if (!finite(current)) out[lane] = kNaN;
                    else {
                        const std::size_t upper =
                            order.count_less_equal(index, current);
                        out[lane] = static_cast<double>(upper) /
                            static_cast<double>(count + 1);
                    }
                } else {
                    const double position =
                        quantile * static_cast<double>(count - 1);
                    const std::size_t lower = static_cast<std::size_t>(position);
                    const std::size_t upper = std::min(count - 1, lower + 1);
                    const double lower_value = order.kth(index, lower);
                    if (upper == lower) out[lane] = lower_value;
                    else {
                        const double upper_value = order.kth(index, upper);
                        out[lane] = lower_value
                            + (position - static_cast<double>(lower))
                                * (upper_value - lower_value);
                    }
                }
            }
        }
    }
};

template <
    std::size_t N,
    class Tensor,
    class Out,
    std::uint64_t QuantileBits,
    class Execution = DirectExecution<N>
>
struct VectorQuantileNode {
    using Shape = typename Tensor::shape;
    static_assert(Shape::rank > 0);
    static constexpr std::size_t Width = Shape::dims[Shape::rank - 1];
    static constexpr std::size_t OutputSize = Shape::size / Width;
    static constexpr double quantile = std::bit_cast<double>(QuantileBits);
    static_assert(quantile >= 0.0 && quantile <= 1.0);

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [begin, end] =
            execution_output_range<OutputSize, Execution>(ctx);
        for (std::size_t output = begin; output < end; ++output) {
            std::array<double, Width> values{};
            std::size_t count = 0;
            const std::size_t input_begin = output * Width;
            for (std::size_t offset = 0; offset < Width; ++offset) {
                const double value = Tensor::read_flat(ctx, input_begin + offset);
                if (finite(value)) values[count++] = value;
            }
            if (count == 0) {
                out[output] = kNaN;
                continue;
            }
            const double position = quantile * static_cast<double>(count - 1);
            const std::size_t lower = static_cast<std::size_t>(position);
            const std::size_t upper = std::min(count - 1, lower + 1);
            std::nth_element(
                values.begin(), values.begin() + lower, values.begin() + count
            );
            const double lower_value = values[lower];
            if (lower == upper) {
                out[output] = lower_value;
            } else {
                const double upper_value = *std::min_element(
                    values.begin() + lower + 1, values.begin() + count
                );
                out[output] = lower_value
                    + (position - static_cast<double>(lower))
                        * (upper_value - lower_value);
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
    std::array<double, MaxPairs> slopes;
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
    struct RankedValue {
        double value;
        std::uint32_t index;
    };
    std::array<RankedValue, Periods> ordered;
    std::array<std::uint32_t, Periods> ranks;
    for (std::size_t index = 0; index < count; ++index) {
        ordered[index] = {
            std::fma(-candidate, points[index].x, points[index].y),
            static_cast<std::uint32_t>(index),
        };
    }
    std::sort(
        ordered.begin(), ordered.begin() + count,
        [](const RankedValue& left, const RankedValue& right) {
            return left.value < right.value
                || (left.value == right.value && left.index < right.index);
        }
    );
    std::size_t rank_begin = 0;
    while (rank_begin < count) {
        std::size_t rank_end = rank_begin + 1;
        while (
            rank_end < count
            && ordered[rank_end].value == ordered[rank_begin].value
        ) {
            ++rank_end;
        }
        for (std::size_t item = rank_begin; item < rank_end; ++item) {
            ranks[ordered[item].index] = static_cast<std::uint32_t>(rank_begin);
        }
        rank_begin = rank_end;
    }
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
            const std::size_t rank = ranks[index];
            result += processed - prefix(rank);
        }
        for (std::size_t index = group; index < end; ++index) {
            add(ranks[index]);
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
            else if constexpr (Periods <= 512) {
                out[lane] = stats_detail::exact_theilsen(points, count);
            } else {
                out[lane] = stats_detail::subquadratic_theilsen(points, count);
            }
        }
    }
};

}  // namespace stackdsl
