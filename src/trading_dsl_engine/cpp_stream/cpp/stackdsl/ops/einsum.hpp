#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "stackdsl/engine.hpp"
#include "stackdsl/ops/cat.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

// The Python frontend canonicalizes NumPy-style labels to integer axis maps.
// Native execution therefore has no runtime parser, strings, dynamic shapes, or
// contraction-path decisions.
template <std::size_t... Dims>
struct TensorShape {
    static constexpr std::size_t rank = sizeof...(Dims);
    static constexpr std::array<std::size_t, rank> dims{Dims...};
    static constexpr std::size_t size = (Dims * ... * std::size_t{1});
};

template <std::size_t... Axes>
struct IndexMap {
    static constexpr std::size_t rank = sizeof...(Axes);
    static constexpr std::array<std::size_t, rank> axes{Axes...};
};

template <class Src>
struct ScalarTensorSource {
    using shape = TensorShape<>;

    template <class Context>
    STACKDSL_HOT static double read_flat(
        const Context& ctx, std::size_t
    ) noexcept {
        return ctx.template read<Src>(0);
    }

    template <class Context>
    STACKDSL_HOT static void load_contiguous(
        const Context& ctx,
        std::size_t,
        std::size_t count,
        double* STACKDSL_RESTRICT out
    ) noexcept {
        const double value = ctx.template read<Src>(0);
        for (std::size_t index = 0; index < count; ++index) out[index] = value;
    }
};

template <std::size_t N, class Src>
struct VectorTensorSource {
    using shape = TensorShape<N>;

    template <class Context>
    STACKDSL_HOT static double read_flat(
        const Context& ctx, std::size_t offset
    ) noexcept {
        return ctx.template read<Src>(offset);
    }

    template <class Context>
    STACKDSL_HOT static void load_contiguous(
        const Context& ctx,
        std::size_t base,
        std::size_t count,
        double* STACKDSL_RESTRICT out
    ) noexcept {
        for (std::size_t index = 0; index < count; ++index) {
            out[index] = ctx.template read<Src>(base + index);
        }
    }
};

template <std::size_t N, class Features>
struct FeatureTensorSource;

template <std::size_t N, class... Sources>
struct FeatureTensorSource<N, FeatureList<Sources...>> {
    static constexpr std::size_t K = FeatureList<Sources...>::width;
    using shape = TensorShape<N, K>;

    template <class Context>
    STACKDSL_HOT static double read_flat(
        const Context& ctx, std::size_t offset
    ) noexcept {
        return read_feature_at(
            ctx,
            offset / K,
            offset % K,
            FeatureList<Sources...>{}
        );
    }

    template <class Context>
    STACKDSL_HOT static void load_contiguous(
        const Context& ctx,
        std::size_t base,
        std::size_t count,
        double* STACKDSL_RESTRICT out
    ) noexcept {
        std::size_t written = 0;
        while (written < count) {
            const std::size_t offset = base + written;
            const std::size_t lane = offset / K;
            const std::size_t feature = offset % K;
            const std::size_t remaining = count - written;
            if (feature == 0 && remaining >= K) {
                std::array<double, K> values{};
                load_features(ctx, lane, values, FeatureList<Sources...>{});
                for (std::size_t index = 0; index < K; ++index) {
                    out[written + index] = values[index];
                }
                written += K;
            } else {
                out[written] = read_feature_at(
                    ctx, lane, feature, FeatureList<Sources...>{}
                );
                ++written;
            }
        }
    }
};

template <class Src, class Shape>
struct FlatTensorSource {
    using shape = Shape;
    static_assert(Shape::size == Src::tensor_size);

    template <class Context>
    STACKDSL_HOT static double read_flat(
        const Context& ctx, std::size_t offset
    ) noexcept {
        if constexpr (requires { ctx.scratch_matrix_f64; }) {
            return ctx.scratch_matrix_f64[Src::tensor_slot_index][offset];
        } else {
            return ctx.scratch_matrix[Src::tensor_slot_index][offset];
        }
    }

    template <class Context>
    STACKDSL_HOT static void load_contiguous(
        const Context& ctx,
        std::size_t base,
        std::size_t count,
        double* STACKDSL_RESTRICT out
    ) noexcept {
        for (std::size_t index = 0; index < count; ++index) {
            out[index] = read_flat(ctx, base + index);
        }
    }
};

template <class Shape>
consteval std::size_t prefix_size(std::size_t count) {
    std::size_t result = 1;
    for (std::size_t axis = 0; axis < count; ++axis) {
        result *= Shape::dims[axis];
    }
    return result;
}

template <class Shape>
consteval std::size_t suffix_size(std::size_t begin) {
    std::size_t result = 1;
    for (std::size_t axis = begin; axis < Shape::rank; ++axis) {
        result *= Shape::dims[axis];
    }
    return result;
}

template <class Shape>
STACKDSL_HOT void unravel_range(
    std::size_t flat,
    std::size_t begin,
    std::size_t end,
    std::array<std::size_t, Shape::rank>& indexes
) noexcept {
    for (std::size_t axis = end; axis-- > begin;) {
        const std::size_t extent = Shape::dims[axis];
        indexes[axis] = flat % extent;
        flat /= extent;
    }
}

template <class Tensor, class Map, class LoopShape, class Context>
STACKDSL_HOT double read_mapped(
    const Context& ctx,
    const std::array<std::size_t, LoopShape::rank>& indexes
) noexcept {
    static_assert(Tensor::shape::rank == Map::rank);
    std::size_t offset = 0;
    for (std::size_t axis = 0; axis < Tensor::shape::rank; ++axis) {
        const std::size_t extent = Tensor::shape::dims[axis];
        const std::size_t index =
            extent == 1 ? 0 : indexes[Map::axes[axis]];
        offset = offset * extent + index;
    }
    return Tensor::read_flat(ctx, offset);
}

template <class Tensor, class Map, class LoopShape>
consteval bool is_full_identity_mapping() {
    if constexpr (
        Tensor::shape::rank != LoopShape::rank ||
        Map::rank != LoopShape::rank
    ) {
        return false;
    } else {
        for (std::size_t axis = 0; axis < LoopShape::rank; ++axis) {
            if (Map::axes[axis] != axis) return false;
            if (Tensor::shape::dims[axis] != LoopShape::dims[axis]) {
                return false;
            }
        }
        return true;
    }
}

template <
    class Tensor,
    class Out,
    class LoopShape,
    class Map,
    std::size_t OutputRank,
    class Execution
>
struct UnaryEinsumNode {
    static_assert(OutputRank <= LoopShape::rank);
    static constexpr std::size_t output_size =
        prefix_size<LoopShape>(OutputRank);
    static constexpr std::size_t reduction_size =
        suffix_size<LoopShape>(OutputRank);
    static constexpr std::size_t reduction_rank =
        LoopShape::rank - OutputRank;

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [output_begin, output_end] =
            execution_output_range<output_size, Execution>(ctx);

        if constexpr (is_full_identity_mapping<Tensor, Map, LoopShape>()) {
            std::array<double, reduction_size> values{};
            for (std::size_t output = output_begin; output < output_end; ++output) {
                const std::size_t base = output * reduction_size;
                Tensor::load_contiguous(
                    ctx, base, reduction_size, values.data()
                );
                double value = 0.0;
                for (
                    std::size_t reduction = 0;
                    reduction < reduction_size;
                    ++reduction
                ) {
                    value += values[reduction];
                }
                out[output] = value;
            }
            return;
        }

        std::array<std::size_t, LoopShape::rank> indexes{};
        for (std::size_t output = output_begin; output < output_end; ++output) {
            unravel_range<LoopShape>(output, 0, OutputRank, indexes);
            double value = 0.0;
            if constexpr (reduction_rank == 0) {
                value = read_mapped<Tensor, Map, LoopShape>(ctx, indexes);
            } else if constexpr (reduction_rank == 1) {
                for (
                    std::size_t reduction = 0;
                    reduction < LoopShape::dims[OutputRank];
                    ++reduction
                ) {
                    indexes[OutputRank] = reduction;
                    value += read_mapped<Tensor, Map, LoopShape>(ctx, indexes);
                }
            } else {
                for (
                    std::size_t reduction = 0;
                    reduction < reduction_size;
                    ++reduction
                ) {
                    unravel_range<LoopShape>(
                        reduction,
                        OutputRank,
                        LoopShape::rank,
                        indexes
                    );
                    value += read_mapped<Tensor, Map, LoopShape>(ctx, indexes);
                }
            }
            out[output] = value;
        }
    }
};

template <
    class Left,
    class Right,
    class Out,
    class LoopShape,
    class LeftMap,
    class RightMap,
    std::size_t OutputRank,
    class Execution
>
struct BinaryEinsumNode {
    static_assert(OutputRank <= LoopShape::rank);
    static constexpr std::size_t output_size =
        prefix_size<LoopShape>(OutputRank);
    static constexpr std::size_t reduction_size =
        suffix_size<LoopShape>(OutputRank);
    static constexpr std::size_t reduction_rank =
        LoopShape::rank - OutputRank;

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const auto [output_begin, output_end] =
            execution_output_range<output_size, Execution>(ctx);

        if constexpr (
            is_full_identity_mapping<Left, LeftMap, LoopShape>() &&
            is_full_identity_mapping<Right, RightMap, LoopShape>()
        ) {
            std::array<double, reduction_size> left_values{};
            std::array<double, reduction_size> right_values{};
            for (std::size_t output = output_begin; output < output_end; ++output) {
                const std::size_t base = output * reduction_size;
                Left::load_contiguous(
                    ctx, base, reduction_size, left_values.data()
                );
                Right::load_contiguous(
                    ctx, base, reduction_size, right_values.data()
                );
                double value = 0.0;
                for (
                    std::size_t reduction = 0;
                    reduction < reduction_size;
                    ++reduction
                ) {
                    value = std::fma(
                        left_values[reduction],
                        right_values[reduction],
                        value
                    );
                }
                out[output] = value;
            }
            return;
        }

        std::array<std::size_t, LoopShape::rank> indexes{};
        for (std::size_t output = output_begin; output < output_end; ++output) {
            unravel_range<LoopShape>(output, 0, OutputRank, indexes);
            double value = 0.0;
            if constexpr (reduction_rank == 0) {
                value =
                    read_mapped<Left, LeftMap, LoopShape>(ctx, indexes) *
                    read_mapped<Right, RightMap, LoopShape>(ctx, indexes);
            } else if constexpr (reduction_rank == 1) {
                for (
                    std::size_t reduction = 0;
                    reduction < LoopShape::dims[OutputRank];
                    ++reduction
                ) {
                    indexes[OutputRank] = reduction;
                    value = std::fma(
                        read_mapped<Left, LeftMap, LoopShape>(ctx, indexes),
                        read_mapped<Right, RightMap, LoopShape>(ctx, indexes),
                        value
                    );
                }
            } else {
                for (
                    std::size_t reduction = 0;
                    reduction < reduction_size;
                    ++reduction
                ) {
                    unravel_range<LoopShape>(
                        reduction,
                        OutputRank,
                        LoopShape::rank,
                        indexes
                    );
                    value = std::fma(
                        read_mapped<Left, LeftMap, LoopShape>(ctx, indexes),
                        read_mapped<Right, RightMap, LoopShape>(ctx, indexes),
                        value
                    );
                }
            }
            out[output] = value;
        }
    }
};

}  // namespace stackdsl
