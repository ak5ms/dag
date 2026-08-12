#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>

#include "stackdsl/engine.hpp"
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
struct MinReductionPolicy {};
struct MaxReductionPolicy {};

template <class Shape, class Axes>
consteval std::size_t reduced_output_size() {
    std::size_t result = 1;
    for (std::size_t axis = 0; axis < Shape::rank; ++axis) {
        if (!Axes::contains(axis)) result *= Shape::dims[axis];
    }
    return result;
}

template <class Shape, class Axes>
consteval bool reduces_contiguous_suffix() {
    bool reduction_started = false;
    for (std::size_t axis = 0; axis < Shape::rank; ++axis) {
        if (Axes::contains(axis)) {
            reduction_started = true;
        } else if (reduction_started) {
            return false;
        }
    }
    return true;
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
        if constexpr (std::is_same_v<Policy, StdReductionPolicy>) {
            mean.fill(0.0);
            m2.fill(0.0);
        } else {
            total.fill(0.0);
        }
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
        } else if constexpr (std::is_same_v<Policy, MinReductionPolicy>) {
            total[index] = count[index] == 0
                ? value
                : std::min(total[index], value);
            ++count[index];
        } else if constexpr (std::is_same_v<Policy, MaxReductionPolicy>) {
            total[index] = count[index] == 0
                ? value
                : std::max(total[index], value);
            ++count[index];
        } else {
            total[index] += value;
            ++count[index];
        }
    }

    STACKDSL_HOT void merge_block(
        std::size_t index,
        double block_total,
        double block_mean,
        double block_m2,
        std::uint64_t block_count,
        bool block_invalid
    ) noexcept {
        if constexpr (!IgnoreNa) {
            if (block_invalid) invalid[index] = true;
        }
        if (block_count == 0) return;
        if constexpr (std::is_same_v<Policy, StdReductionPolicy>) {
            if (count[index] == 0) {
                mean[index] = block_mean;
                m2[index] = block_m2;
                count[index] = block_count;
                return;
            }
            const std::uint64_t previous_count = count[index];
            const std::uint64_t combined_count = previous_count + block_count;
            const double delta = block_mean - mean[index];
            mean[index] += delta * static_cast<double>(block_count)
                / static_cast<double>(combined_count);
            m2[index] += block_m2
                + delta * delta
                    * static_cast<double>(previous_count)
                    * static_cast<double>(block_count)
                    / static_cast<double>(combined_count);
            count[index] = combined_count;
        } else if constexpr (std::is_same_v<Policy, MinReductionPolicy>) {
            total[index] = count[index] == 0
                ? block_total
                : std::min(total[index], block_total);
            count[index] += block_count;
        } else if constexpr (std::is_same_v<Policy, MaxReductionPolicy>) {
            total[index] = count[index] == 0
                ? block_total
                : std::max(total[index], block_total);
            count[index] += block_count;
        } else {
            total[index] += block_total;
            count[index] += block_count;
        }
    }

    STACKDSL_HOT double result(std::size_t index) const noexcept {
        if constexpr (!IgnoreNa) {
            if (invalid[index]) return kNaN;
        }
        if (count[index] == 0) return kNaN;
        if constexpr (
            std::is_same_v<Policy, SumReductionPolicy>
            || std::is_same_v<Policy, MinReductionPolicy>
            || std::is_same_v<Policy, MaxReductionPolicy>
        ) {
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
    bool Temporal,
    class Execution
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
    static constexpr bool contiguous_suffix =
        reduces_contiguous_suffix<Shape, Axes>();
    static constexpr std::size_t contiguous_reduction_width =
        contiguous_suffix ? input_size / output_size : 1;
    using State = ReductionState<Policy, output_size, Ddof, IgnoreNa>;

    State state{};

    STACKDSL_HOT void setup() noexcept { state.reset(); }

    template <class Context>
    STACKDSL_HOT static std::pair<std::size_t, std::size_t> active_output_range(
        const Context& ctx
    ) noexcept {
        if constexpr (retains_leading_axis) {
            return execution_output_range<output_size, Execution>(ctx);
        }
        return {0, output_size};
    }

    template <class Context>
    STACKDSL_HOT static void accumulate(State& target, const Context& ctx) noexcept {
        if constexpr (contiguous_suffix) {
            const auto [output_begin, output_end] = active_output_range(ctx);
            for (
                std::size_t output = output_begin;
                output < output_end;
                ++output
            ) {
                const std::size_t input_begin =
                    output * contiguous_reduction_width;
                if constexpr (
                    contiguous_reduction_width > 1
                    && !std::is_same_v<Policy, StdReductionPolicy>
                ) {
                    std::array<double, 4> partial{};
                    std::uint64_t block_count = 0;
                    bool block_invalid = false;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
                    for (
                        std::size_t reduction = 0;
                        reduction < contiguous_reduction_width;
                        ++reduction
                    ) {
                        const double value =
                            Tensor::read_flat(ctx, input_begin + reduction);
                        if (finite(value)) {
                            if constexpr (std::is_same_v<Policy, MinReductionPolicy>) {
                                partial[0] = block_count == 0
                                    ? value
                                    : std::min(partial[0], value);
                            } else if constexpr (std::is_same_v<Policy, MaxReductionPolicy>) {
                                partial[0] = block_count == 0
                                    ? value
                                    : std::max(partial[0], value);
                            } else {
                                partial[reduction & 3U] += value;
                            }
                            ++block_count;
                        } else if constexpr (!IgnoreNa) {
                            block_invalid = true;
                        }
                    }
                    const double block_total =
                        std::is_same_v<Policy, MinReductionPolicy>
                        || std::is_same_v<Policy, MaxReductionPolicy>
                        ? partial[0]
                        : (partial[0] + partial[1])
                            + (partial[2] + partial[3]);
                    target.merge_block(
                        output,
                        block_total,
                        0.0,
                        0.0,
                        block_count,
                        block_invalid
                    );
                } else if constexpr (contiguous_reduction_width > 1) {
                    double block_mean = 0.0;
                    double block_m2 = 0.0;
                    std::uint64_t block_count = 0;
                    bool block_invalid = false;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
                    for (
                        std::size_t reduction = 0;
                        reduction < contiguous_reduction_width;
                        ++reduction
                    ) {
                        const double value =
                            Tensor::read_flat(ctx, input_begin + reduction);
                        if (!finite(value)) {
                            if constexpr (!IgnoreNa) block_invalid = true;
                            continue;
                        }
                        const std::uint64_t next_count = block_count + 1;
                        const double delta = value - block_mean;
                        block_mean += delta / static_cast<double>(next_count);
                        const double delta2 = value - block_mean;
                        block_m2 = std::fma(delta, delta2, block_m2);
                        block_count = next_count;
                    }
                    target.merge_block(
                        output,
                        0.0,
                        block_mean,
                        block_m2,
                        block_count,
                        block_invalid
                    );
                } else {
                    target.add(output, Tensor::read_flat(ctx, input_begin));
                }
            }
        } else if constexpr (retains_leading_axis) {
            const auto [output_begin, output_end] = active_output_range(ctx);
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
            for (std::size_t offset = 0; offset < input_size; ++offset) {
                const std::size_t output =
                    reduced_output_index<Shape, Axes>(offset);
                if (output < output_begin || output >= output_end) continue;
                target.add(output, Tensor::read_flat(ctx, offset));
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
        const auto [begin, end] = active_output_range(ctx);
        for (std::size_t index = begin; index < end; ++index) {
            out[index] = source.result(index);
        }
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        if constexpr (Temporal) {
            accumulate(state, ctx);
            // Temporal state is projected only by finalize(). If this reduction
            // feeds downstream algebra, codegen schedules that complete suffix in
            // the final phase after this accumulator has written its result once.
        } else {
            state.reset();
            accumulate(state, ctx);
            write_result(state, ctx);
        }
    }

    template <class Context>
    STACKDSL_HOT void finalize(Context& ctx) noexcept {
        if constexpr (Temporal) write_result(state, ctx);
    }
};

template <class Tensor, class Out>
struct ReductionBinding {
    using tensor_type = Tensor;
    using output_type = Out;
};

template <
    class Axes,
    class Policy,
    std::size_t Ddof,
    bool IgnoreNa,
    bool Temporal,
    class Execution,
    class... Bindings
>
struct ReductionBundleNode {
    static_assert(sizeof...(Bindings) > 1);
    static constexpr std::size_t component_count = sizeof...(Bindings);
    using BindingTuple = std::tuple<Bindings...>;
    using FirstBinding = std::tuple_element_t<0, BindingTuple>;
    using Shape = typename FirstBinding::tensor_type::shape;
    static_assert((std::is_same_v<Shape, typename Bindings::tensor_type::shape> && ...));

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

    std::array<State, component_count> state{};

    STACKDSL_HOT void setup() noexcept {
        for (auto& item : state) item.reset();
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        if constexpr (!Temporal) {
            for (auto& item : state) item.reset();
        }
        accumulate(ctx);
        if constexpr (!Temporal) write_result(ctx);
    }

    template <class Context>
    STACKDSL_HOT void finalize(Context& ctx) noexcept {
        if constexpr (Temporal) write_result(ctx);
    }

private:
    template <class Context>
    STACKDSL_HOT static std::pair<std::size_t, std::size_t> active_output_range(
        const Context& ctx
    ) noexcept {
        if constexpr (retains_leading_axis) {
            return execution_output_range<output_size, Execution>(ctx);
        }
        return {0, output_size};
    }

    template <std::size_t Index, class Context>
    STACKDSL_HOT static double read_binding(
        const Context& ctx, std::size_t offset
    ) noexcept {
        using Tensor = typename std::tuple_element_t<
            Index, BindingTuple
        >::tensor_type;
        return Tensor::read_flat(ctx, offset);
    }

    template <class Context, std::size_t... Indexes>
    STACKDSL_HOT void add_offset(
        const Context& ctx,
        std::size_t output,
        std::size_t offset,
        std::index_sequence<Indexes...>
    ) noexcept {
        // One physical traversal exposes all sibling tensor expressions in the
        // same optimizer region and computes the output index only once.
        (state[Indexes].add(
            output,
            read_binding<Indexes>(ctx, offset)
        ), ...);
    }

    template <class Context, std::size_t... Indexes>
    STACKDSL_HOT static std::array<double, component_count> read_values(
        const Context& ctx,
        std::size_t offset,
        std::index_sequence<Indexes...>
    ) noexcept {
        return {read_binding<Indexes>(ctx, offset)...};
    }

    template <class Context>
    STACKDSL_HOT void accumulate(const Context& ctx) noexcept {
        using Indexes = std::make_index_sequence<component_count>;
        if constexpr (reduces_contiguous_suffix<Shape, Axes>()) {
            constexpr std::size_t reduction_width = input_size / output_size;
            const auto [output_begin, output_end] = active_output_range(ctx);
            for (std::size_t output = output_begin; output < output_end; ++output) {
                const std::size_t base = output * reduction_width;
                if constexpr (
                    reduction_width > 1
                    && !std::is_same_v<Policy, StdReductionPolicy>
                ) {
                    std::array<std::array<double, 4>, component_count> partial{};
                    std::array<std::uint64_t, component_count> count{};
                    std::array<bool, component_count> invalid{};
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 8
#endif
                    for (std::size_t reduction = 0; reduction < reduction_width; ++reduction) {
                        const auto values = read_values(
                            ctx, base + reduction, Indexes{}
                        );
                        for (std::size_t component = 0; component < component_count; ++component) {
                            const double value = values[component];
                            if (finite(value)) {
                                if constexpr (std::is_same_v<Policy, MinReductionPolicy>) {
                                    partial[component][0] = count[component] == 0
                                        ? value
                                        : std::min(partial[component][0], value);
                                } else if constexpr (std::is_same_v<Policy, MaxReductionPolicy>) {
                                    partial[component][0] = count[component] == 0
                                        ? value
                                        : std::max(partial[component][0], value);
                                } else {
                                    partial[component][reduction & 3U] += value;
                                }
                                ++count[component];
                            } else if constexpr (!IgnoreNa) {
                                invalid[component] = true;
                            }
                        }
                    }
                    for (std::size_t component = 0; component < component_count; ++component) {
                        const double total =
                            std::is_same_v<Policy, MinReductionPolicy>
                            || std::is_same_v<Policy, MaxReductionPolicy>
                            ? partial[component][0]
                            : (partial[component][0] + partial[component][1])
                                + (partial[component][2] + partial[component][3]);
                        state[component].merge_block(
                            output,
                            total,
                            0.0,
                            0.0,
                            count[component],
                            invalid[component]
                        );
                    }
                } else if constexpr (reduction_width > 1) {
                    std::array<double, component_count> block_mean{};
                    std::array<double, component_count> block_m2{};
                    std::array<std::uint64_t, component_count> block_count{};
                    std::array<bool, component_count> block_invalid{};
                    for (std::size_t reduction = 0; reduction < reduction_width; ++reduction) {
                        const auto values = read_values(
                            ctx, base + reduction, Indexes{}
                        );
                        for (std::size_t component = 0; component < component_count; ++component) {
                            const double value = values[component];
                            if (!finite(value)) {
                                if constexpr (!IgnoreNa) block_invalid[component] = true;
                                continue;
                            }
                            const std::uint64_t next = block_count[component] + 1;
                            const double delta = value - block_mean[component];
                            block_mean[component] += delta / static_cast<double>(next);
                            const double delta2 = value - block_mean[component];
                            block_m2[component] = std::fma(
                                delta, delta2, block_m2[component]
                            );
                            block_count[component] = next;
                        }
                    }
                    for (std::size_t component = 0; component < component_count; ++component) {
                        state[component].merge_block(
                            output,
                            0.0,
                            block_mean[component],
                            block_m2[component],
                            block_count[component],
                            block_invalid[component]
                        );
                    }
                } else {
                    add_offset(ctx, output, base, Indexes{});
                }
            }
        } else if constexpr (retains_leading_axis) {
            const auto [output_begin, output_end] = active_output_range(ctx);
            for (std::size_t offset = 0; offset < input_size; ++offset) {
                const std::size_t output =
                    reduced_output_index<Shape, Axes>(offset);
                if (output < output_begin || output >= output_end) continue;
                add_offset(ctx, output, offset, Indexes{});
            }
        } else {
            for (std::size_t offset = 0; offset < input_size; ++offset) {
                add_offset(
                    ctx,
                    reduced_output_index<Shape, Axes>(offset),
                    offset,
                    Indexes{}
                );
            }
        }
    }

    template <class Context, std::size_t... Indexes>
    STACKDSL_HOT static auto output_pointers(
        Context& ctx, std::index_sequence<Indexes...>
    ) noexcept {
        return std::array<double*, component_count>{
            ctx.template write_ptr<
                typename std::tuple_element_t<Indexes, BindingTuple>::output_type
            >()...
        };
    }

    template <class Context>
    STACKDSL_HOT void write_result(Context& ctx) noexcept {
        auto outputs = output_pointers(
            ctx, std::make_index_sequence<component_count>{}
        );
        const auto [output_begin, output_end] = active_output_range(ctx);
        for (std::size_t output = output_begin; output < output_end; ++output) {
            for (std::size_t component = 0; component < component_count; ++component) {
                outputs[component][output] = state[component].result(output);
            }
        }
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
