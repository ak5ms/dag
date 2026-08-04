#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "stackdsl/engine.hpp"
#include "stackdsl/ops/einsum.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <class InputShape, class OutputShape>
consteval bool tensor_broadcast_compatible() {
    if constexpr (InputShape::rank > OutputShape::rank) {
        return false;
    } else {
        constexpr std::size_t offset = OutputShape::rank - InputShape::rank;
        for (std::size_t axis = 0; axis < InputShape::rank; ++axis) {
            const std::size_t input_extent = InputShape::dims[axis];
            const std::size_t output_extent = OutputShape::dims[offset + axis];
            if (input_extent != 1 && input_extent != output_extent) return false;
        }
        return true;
    }
}

template <class InputShape, class OutputShape>
STACKDSL_HOT std::size_t tensor_broadcast_index(std::size_t flat) noexcept {
    static_assert(tensor_broadcast_compatible<InputShape, OutputShape>());
    if constexpr (InputShape::rank == 0) {
        (void)flat;
        return 0;
    } else {
        std::array<std::size_t, OutputShape::rank> indexes{};
        for (std::size_t axis = OutputShape::rank; axis-- > 0;) {
            indexes[axis] = flat % OutputShape::dims[axis];
            flat /= OutputShape::dims[axis];
        }
        constexpr std::size_t offset = OutputShape::rank - InputShape::rank;
        std::size_t input = 0;
        for (std::size_t axis = 0; axis < InputShape::rank; ++axis) {
            const std::size_t extent = InputShape::dims[axis];
            input = input * extent + (extent == 1 ? 0 : indexes[offset + axis]);
        }
        return input;
    }
}

template <class Tensor, class OutputShape, class Context>
STACKDSL_HOT double tensor_broadcast_read(
    const Context& ctx,
    std::size_t output_index
) noexcept {
    return Tensor::read_flat(
        ctx,
        tensor_broadcast_index<typename Tensor::shape, OutputShape>(output_index)
    );
}

template <
    class Input,
    class Out,
    class OutputShape,
    class Result,
    class Op,
    class Execution
>
struct TensorUnaryNode {
    static_assert(Op::arity == 1);
    static_assert(tensor_broadcast_compatible<typename Input::shape, OutputShape>());

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [begin, end] =
            execution_output_range<OutputShape::size, Execution>(ctx);
        for (std::size_t index = begin; index < end; ++index) {
            out[index] = Op::template apply<Result>(
                tensor_broadcast_read<Input, OutputShape>(ctx, index)
            );
        }
    }
};

template <
    class Left,
    class Right,
    class Out,
    class OutputShape,
    class Result,
    class Op,
    class Execution
>
struct TensorBinaryNode {
    static_assert(Op::arity == 2);
    static_assert(tensor_broadcast_compatible<typename Left::shape, OutputShape>());
    static_assert(tensor_broadcast_compatible<typename Right::shape, OutputShape>());

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [begin, end] =
            execution_output_range<OutputShape::size, Execution>(ctx);
        for (std::size_t index = begin; index < end; ++index) {
            out[index] = Op::template apply<Result>(
                tensor_broadcast_read<Left, OutputShape>(ctx, index),
                tensor_broadcast_read<Right, OutputShape>(ctx, index)
            );
        }
    }
};

template <
    class A,
    class B,
    class C,
    class Out,
    class OutputShape,
    class Result,
    class Op,
    class Execution
>
struct TensorTernaryNode {
    static_assert(Op::arity == 3);
    static_assert(tensor_broadcast_compatible<typename A::shape, OutputShape>());
    static_assert(tensor_broadcast_compatible<typename B::shape, OutputShape>());
    static_assert(tensor_broadcast_compatible<typename C::shape, OutputShape>());

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [begin, end] =
            execution_output_range<OutputShape::size, Execution>(ctx);
        for (std::size_t index = begin; index < end; ++index) {
            out[index] = Op::template apply<Result>(
                tensor_broadcast_read<A, OutputShape>(ctx, index),
                tensor_broadcast_read<B, OutputShape>(ctx, index),
                tensor_broadcast_read<C, OutputShape>(ctx, index)
            );
        }
    }
};

template <class Input, class Out, class Execution>
struct TensorCopyNode {
    using Shape = typename Input::shape;

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [begin, end] = execution_output_range<Shape::size, Execution>(ctx);
        for (std::size_t index = begin; index < end; ++index) {
            out[index] = Input::read_flat(ctx, index);
        }
    }
};

template <class Input, class Out, std::size_t Index, class Execution>
struct TensorColumnNode {
    using Shape = typename Input::shape;
    static_assert(Shape::rank > 0);
    static constexpr std::size_t Width = Shape::dims[Shape::rank - 1];
    static constexpr std::size_t OutputSize = Shape::size / Width;
    static_assert(Index < Width);

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [begin, end] =
            execution_output_range<OutputSize, Execution>(ctx);
        for (std::size_t output = begin; output < end; ++output) {
            out[output] = Input::read_flat(ctx, output * Width + Index);
        }
    }
};

template <class Input, class Out, class Execution>
struct TensorCumsumNode {
    using Shape = typename Input::shape;
    alignas(64) std::array<double, Shape::size> value{};

    STACKDSL_HOT void setup() noexcept { value.fill(0.0); }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [begin, end] = execution_output_range<Shape::size, Execution>(ctx);
        for (std::size_t index = begin; index < end; ++index) {
            const double x = Input::read_flat(ctx, index);
            if (finite(x)) {
                value[index] += x;
                out[index] = value[index];
            } else {
                out[index] = kNaN;
            }
        }
    }
};

template <class Input, class Out, std::int64_t Limit, class Execution>
struct TensorFFillNode {
    using Shape = typename Input::shape;
    alignas(64) std::array<double, Shape::size> last{};
    alignas(64) std::array<std::int64_t, Shape::size> streak{};
    alignas(64) std::array<std::uint8_t, Shape::size> seen{};

    STACKDSL_HOT void setup() noexcept {
        last.fill(kNaN);
        streak.fill(0);
        seen.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [begin, end] = execution_output_range<Shape::size, Execution>(ctx);
        for (std::size_t index = begin; index < end; ++index) {
            const double x = Input::read_flat(ctx, index);
            if (finite(x)) {
                last[index] = x;
                streak[index] = 0;
                seen[index] = 1;
                out[index] = x;
            } else if (seen[index] && (Limit < 0 || streak[index] < Limit)) {
                ++streak[index];
                out[index] = last[index];
            } else {
                out[index] = kNaN;
            }
        }
    }
};

template <
    class Input,
    class Out,
    std::size_t Lag,
    std::size_t MaxLag,
    class Execution
>
struct TensorShiftNode {
    static_assert(Lag <= MaxLag);
    using Shape = typename Input::shape;
    static constexpr std::size_t Capacity = MaxLag + 1;
    alignas(64) std::array<std::array<double, Shape::size>, Capacity> buffer{};
    std::size_t position = 0;
    std::size_t count = 0;

    STACKDSL_HOT void setup() noexcept {
        for (auto& row : buffer) row.fill(kNaN);
        position = 0;
        count = 0;
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t read_position = (position + Capacity - Lag) % Capacity;
        const auto [begin, end] = execution_output_range<Shape::size, Execution>(ctx);
        for (std::size_t index = begin; index < end; ++index) {
            const double x = Input::read_flat(ctx, index);
            out[index] = Lag == 0
                ? x
                : (count >= Lag ? buffer[read_position][index] : kNaN);
            buffer[position][index] = x;
        }
        position = (position + 1) % Capacity;
        if (count < Capacity) ++count;
    }
};

template <
    class Input,
    class Out,
    std::uint64_t SpanBits,
    int MinPeriods,
    bool IgnoreNa,
    bool Adjust,
    class Execution
>
struct TensorEwmNode {
    using Shape = typename Input::shape;
    static constexpr double span = std::bit_cast<double>(SpanBits);
    static_assert(span > 0.0);
    static constexpr double alpha = 2.0 / (span + 1.0);
    static constexpr double old_weight_factor = 1.0 - alpha;

    alignas(64) std::array<double, Shape::size> value{};
    alignas(64) std::array<double, Shape::size> weight{};
    alignas(64) std::array<std::int64_t, Shape::size> count{};
    alignas(64) std::array<std::uint8_t, Shape::size> initialized{};

    STACKDSL_HOT void setup() noexcept {
        value.fill(0.0);
        weight.fill(0.0);
        count.fill(0);
        initialized.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        auto* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [begin, end] = execution_output_range<Shape::size, Execution>(ctx);
        for (std::size_t index = begin; index < end; ++index) {
            const double x = Input::read_flat(ctx, index);
            const bool observation = finite(x);

            if constexpr (MinPeriods <= 0 && IgnoreNa && !Adjust) {
                if (observation) {
                    if (initialized[index]) {
                        value[index] = std::fma(
                            alpha,
                            x - value[index],
                            value[index]
                        );
                    } else {
                        value[index] = x;
                        initialized[index] = 1;
                    }
                }
                out[index] = initialized[index] ? value[index] : kNaN;
                continue;
            }

            double old_weight = weight[index];
            if (initialized[index] && (observation || !IgnoreNa)) {
                old_weight *= old_weight_factor;
            }
            if (observation) {
                if (initialized[index]) {
                    double new_weight = Adjust ? 1.0 : alpha;
                    if constexpr (!Adjust) {
                        if (std::abs(alpha - 0.5) <= 1e-12) {
                            new_weight = 1.0 - old_weight;
                        }
                    }
                    if (value[index] != x) {
                        value[index] = (
                            old_weight * value[index] + new_weight * x
                        ) / (old_weight + new_weight);
                    }
                    old_weight = Adjust ? old_weight + new_weight : 1.0;
                } else {
                    value[index] = x;
                    initialized[index] = 1;
                    old_weight = 1.0;
                }
                ++count[index];
            }
            weight[index] = old_weight;
            const bool enough = MinPeriods <= 0 || count[index] >= MinPeriods;
            out[index] = initialized[index] && enough ? value[index] : kNaN;
        }
    }
};

}  // namespace stackdsl
