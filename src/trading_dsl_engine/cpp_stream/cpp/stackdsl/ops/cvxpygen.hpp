#pragma once

#include <cstddef>
#include <stdexcept>

#include "stackdsl/ops/einsum.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <std::size_t Index, class TensorSource>
struct CvxpygenParameterBinding {
    static constexpr std::size_t index = Index;
    using source_type = TensorSource;
};

template <class... Bindings>
struct CvxpygenParameterList {};

template <
    std::size_t PrimalIndex,
    std::size_t Offset,
    std::size_t Count,
    std::size_t Stride,
    class Out
>
struct CvxpygenPrimalProjection {
    static constexpr std::size_t primal_index = PrimalIndex;
    static constexpr std::size_t offset = Offset;
    static constexpr std::size_t count = Count;
    static constexpr std::size_t stride = Stride;
    using output_type = Out;
};

template <class... Projections>
struct CvxpygenProjectionList {};

template <class Program, class Parameters, class Projections>
class CvxpygenNode;

template <class Program, class... Bindings, class... Projections>
class CvxpygenNode<
    Program,
    CvxpygenParameterList<Bindings...>,
    CvxpygenProjectionList<Projections...>
> {
    Program program_{};

    template <class Binding, class Context>
    STACKDSL_HOT void load_parameter(const Context& ctx) {
        constexpr std::size_t index = Binding::index;
        using Source = typename Binding::source_type;
        static_assert(
            Source::shape::size == Program::template parameter_size<index>(),
            "CVXPYgen parameter and cpp_stream source sizes differ"
        );
        auto target = program_.template parameter_buffer<index>();
        Source::load_contiguous(ctx, 0, target.size(), target.data());
        program_.template mark_parameter_dirty<index>();
    }

    template <class Projection, class Context>
    STACKDSL_HOT void project(Context& ctx) const noexcept {
        const auto source = program_.template primal<Projection::primal_index>();
        auto* STACKDSL_RESTRICT out =
            ctx.template write_ptr<typename Projection::output_type>();
        for (std::size_t index = 0; index < Projection::count; ++index) {
            out[index] = source[
                Projection::offset + index * Projection::stride
            ];
        }
    }

public:
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) {
        (load_parameter<Bindings>(ctx), ...);
        program_.solve();
        (project<Projections>(ctx), ...);
    }
};

}  // namespace stackdsl
