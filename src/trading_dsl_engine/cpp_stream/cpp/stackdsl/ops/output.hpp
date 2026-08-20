#pragma once

#include <array>
#include <cstddef>

#include "stackdsl/utils.hpp"

namespace stackdsl {

// Exact compile-time geometry for one public formula output.
template <
    std::size_t Offset,
    std::size_t Size,
    bool Final,
    bool LanePartitionable
>
struct OutputSpec {
    static constexpr std::size_t offset = Offset;
    static constexpr std::size_t size = Size;
    static constexpr std::size_t end = Offset + Size;
    static constexpr bool final = Final;
    static constexpr bool lane_partitionable = LanePartitionable;
};

// Keep every output size/offset visible to C++ rather than collapsing the layout
// to one aggregate width in Python. The compiler folds all geometry below.
template <class... Outputs>
struct OutputLayout {
    static constexpr std::size_t count = sizeof...(Outputs);
    static constexpr std::array<std::size_t, count> sizes{Outputs::size...};
    static constexpr std::array<std::size_t, count> offsets{Outputs::offset...};
    static constexpr std::array<bool, count> finals{Outputs::final...};
    static constexpr std::array<bool, count> lane_partitionable{
        Outputs::lane_partitionable...
    };

    static constexpr std::size_t row_width = []() consteval {
        std::size_t result = 0;
        ((result = !Outputs::final && Outputs::end > result
            ? Outputs::end
            : result), ...);
        return result;
    }();

    static constexpr std::size_t final_width = []() consteval {
        std::size_t result = 0;
        ((result = Outputs::final && Outputs::end > result
            ? Outputs::end
            : result), ...);
        return result;
    }();

    static constexpr bool has_final_output = final_width > 0;
    static constexpr bool row_lane_partitionable =
        ((Outputs::final || Outputs::lane_partitionable) && ... && true);
};

// Ordinary physical operators own public storage directly. The offset is a
// template constant consumed by RowContext::write_ptr, so there is no runtime
// output dispatch or pointer rebasing in the hot loop.
template <std::size_t Offset>
struct OutputSliceDst {
    using value_type = double;
    static constexpr std::size_t output_offset = Offset;
};

// Read a previously produced public row output directly from the current row's
// packed region. RowContext keeps row_output distinct from output so this remains
// valid while output is temporarily pointed at the final-output region.
template <std::ptrdiff_t Offset, std::size_t Width = 1, bool RowScalar = false>
struct PackedOutputSrc {
    using value_type = double;
    static constexpr std::size_t feature_width = Width;
    static constexpr bool row_scalar = RowScalar;

    template <class Context>
    STACKDSL_HOT static double read(
        const Context& ctx, std::size_t lane
    ) noexcept {
        const std::ptrdiff_t index = Offset + static_cast<std::ptrdiff_t>(
            RowScalar ? 0 : lane * Width
        );
        return ctx.row_output[index];
    }

    template <class Context>
    STACKDSL_HOT static double read_feature(
        const Context& ctx, std::size_t lane, std::size_t feature
    ) noexcept {
        const std::ptrdiff_t index = Offset + static_cast<std::ptrdiff_t>(
            lane * Width + feature
        );
        return ctx.row_output[index];
    }

    template <class Context>
    STACKDSL_HOT static const double* read_ptr(const Context& ctx) noexcept {
        static_assert(Offset >= 0);
        return ctx.row_output + static_cast<std::size_t>(Offset);
    }
};

// Arbitrary-rank view over the same current-row public storage.
template <std::ptrdiff_t Offset, class Shape>
struct PackedOutputTensorSource {
    using shape = Shape;

    template <class Context>
    STACKDSL_HOT static double read_flat(
        const Context& ctx, std::size_t index
    ) noexcept {
        return ctx.row_output[
            Offset + static_cast<std::ptrdiff_t>(index)
        ];
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

}  // namespace stackdsl
