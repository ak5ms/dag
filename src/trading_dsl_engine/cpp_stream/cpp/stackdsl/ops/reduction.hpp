#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "stackdsl/ops/einsum.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <std::size_t... Axes>
struct AxisList {
    static constexpr std::size_t size = sizeof...(Axes);
    static constexpr std::array<std::size_t, size> values{Axes...};

    static constexpr bool contains(std::size_t axis) noexcept {
        return ((axis == Axes) || ... || false);
    }
};

struct SumReductionPolicy {};
struct MeanReductionPolicy {};
struct StdReductionPolicy {};

template <class Shape, class Axes>
consteval std::size_t reduced_output_size() {
    std::size_t result = 1;
    for (std::size_t axis = 0; axis < Shape::rank; ++axis) {
        if (!Axes::contains(axis)) result *= Shape::dims[axis];
    }
    return result;
}

template <class Shape, class Axes>
STACKDSL_HOT std::size_t reduced_output_index(std::size_t flat) noexcept {
    if constexpr (Axes::size == 0) {
        return flat;
    } else {
        std::array<std::size_t, Shape::rank> indexes{};
        for (std::size_t axis = Shape::rank; axis-- > 0;) {
            indexes[axis] = flat % Shape::dims[axis];
            flat /= Shape::dims[axis];
        }
        std::size_t output = 0;
        for (std::size_t axis = 0; axis < Shape::rank; ++axis) {
            if (!Axes::contains(axis)) {
                output = output * Shape::dims[axis] + indexes[axis];
            }
        }
        return output;
    }
}

template <
    class Policy,
    std::size_t Size,
    std::size_t Ddof,
    bool IgnoreNa
>
struct ReductionState {
    alignas(64) std::array<double, Size> total{};
    alignas(64) std::array<double, Size> mean{};
    alignas(64) std::array<double, Size> m2{};
    alignas(64) std::array<std::uint64_t, Size> count{};
    alignas(64) std::array<bool, Size> invalid{};

    STACKDSL_HOT void reset() noexcept {
        total.fill(0.0);
        mean.fill(0.0);
        m2.fill(0.0);
        count.fill(0);
        if constexpr (!IgnoreNa) invalid.fill(false);
    }

    STACKDSL_HOT void add(std::size_t index, double value) noexcept {
        if (!finite(value)) {
            if constexpr (!IgnoreNa) invalid[index] = true;
            return;
        }
        if constexpr (std::is_same_v<Policy, StdReductionPolicy>) {
            const std::uint64_t next_count = count[index] + 1;
            const double delta = value - mean[index];
            mean[index] += delta / static_cast<double>(next_count);
            const double delta2 = value - mean[index];
            m2[index] = std::fma(delta, delta2, m2[index]);
            count[index] = next_count;
        } else {
            total[index] += value;
            ++count[index];
        }
    }

    STACKDSL_HOT double result(std::size_t index) const noexcept {
        if constexpr (!IgnoreNa) {
            if (invalid[index]) return kNaN;
        }
        if (count[index] == 0) return kNaN;
        if constexpr (std::is_same_v<Policy, SumReductionPolicy>) {
            return total[index];
        } else if constexpr (std::is_same_v<Policy, MeanReductionPolicy>) {
            return total[index] / static_cast<double>(count[index]);
        } else {
            if (count[index] <= Ddof) return kNaN;
            const double denominator = static_cast<double>(count[index] - Ddof);
            return std::sqrt(std::max(0.0, m2[index] / denominator));
        }
    }
};

template <
    class Tensor,
    class Out,
    class Axes,
    class Policy,
    std::size_t Ddof,
    bool IgnoreNa,
    bool Temporal
>
struct ReductionNode {
    using Shape = typename Tensor::shape;
    static constexpr std::size_t input_size = Shape::size;
    static constexpr std::size_t output_size = reduced_output_size<Shape, Axes>();
    static constexpr bool retains_leading_axis =
        Shape::rank > 0 && !Axes::contains(0);
    static constexpr std::size_t leading_extent =
        Shape::rank > 0 ? Shape::dims[0] : 1;
    static constexpr std::size_t input_lane_width =
        retains_leading_axis ? input_size / leading_extent : input_size;
    static constexpr std::size_t output_lane_width =
        retains_leading_axis ? output_size / leading_extent : output_size;
    using State = ReductionState<Policy, output_size, Ddof, IgnoreNa>;

    State state{};

    STACKDSL_HOT void setup() noexcept { state.reset(); }

    template <class Context>
    STACKDSL_HOT static void accumulate(State& target, const Context& ctx) noexcept {
        if constexpr (retains_leading_axis) {
            const std::size_t lane_begin = std::min(ctx.lane_begin, leading_extent);
            const std::size_t lane_end = std::min(ctx.lane_end, leading_extent);
            for (std::size_t lane = lane_begin; lane < lane_end; ++lane) {
                const std::size_t begin = lane * input_lane_width;
                const std::size_t end = begin + input_lane_width;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
                for (std::size_t offset = begin; offset < end; ++offset) {
                    target.add(
                        reduced_output_index<Shape, Axes>(offset),
                        Tensor::read_flat(ctx, offset)
                    );
                }
            }
        } else {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
            for (std::size_t offset = 0; offset < input_size; ++offset) {
                target.add(
                    reduced_output_index<Shape, Axes>(offset),
                    Tensor::read_flat(ctx, offset)
                );
            }
        }
    }

    template <class Context>
    STACKDSL_HOT static void write_result(
        const State& source, Context& ctx
    ) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        if constexpr (retains_leading_axis) {
            const std::size_t lane_begin = std::min(ctx.lane_begin, leading_extent);
            const std::size_t lane_end = std::min(ctx.lane_end, leading_extent);
            for (std::size_t lane = lane_begin; lane < lane_end; ++lane) {
                const std::size_t begin = lane * output_lane_width;
                const std::size_t end = begin + output_lane_width;
                for (std::size_t index = begin; index < end; ++index) {
                    out[index] = source.result(index);
                }
            }
        } else {
            for (std::size_t index = 0; index < output_size; ++index) {
                out[index] = source.result(index);
            }
        }
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        if constexpr (Temporal) {
            accumulate(state, ctx);
        } else {
            State row{};
            row.reset();
            accumulate(row, ctx);
            write_result(row, ctx);
        }
    }

    template <class Context>
    STACKDSL_HOT void finalize(Context& ctx) noexcept {
        write_result(state, ctx);
    }
};

template <class Tensor, class Out>
struct EmitLastNode {
    static constexpr std::size_t size = Tensor::shape::size;
    alignas(64) std::array<double, size> value{};
    bool seen = false;

    STACKDSL_HOT void setup() noexcept {
        value.fill(kNaN);
        seen = false;
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        Tensor::load_contiguous(ctx, 0, size, value.data());
        seen = true;
    }

    template <class Context>
    STACKDSL_HOT void finalize(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        for (std::size_t index = 0; index < size; ++index) {
            out[index] = seen ? value[index] : kNaN;
        }
    }
};

}  // namespace stackdsl
