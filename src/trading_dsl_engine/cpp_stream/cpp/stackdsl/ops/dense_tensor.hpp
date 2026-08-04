#pragma once

#include <cstddef>

#include "stackdsl/utils.hpp"

namespace stackdsl {

// Adapter for an arbitrary-rank C-order tensor stored contiguously in one input
// row. Shape is a compile-time TensorShape supplied by einsum code generation.
template <class Src, class Shape>
struct DenseTensorSource {
    using shape = Shape;

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

}  // namespace stackdsl
