#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

// Compile-time feature concatenation. Physical lowering recursively flattens
// nested cat(...) expressions into this list, so consumers read the original
// mapped/scratch values directly without materializing an intermediate N x K
// matrix. This is a general value-layout primitive, not a Ridge-specific path.
template <class... Sources>
struct FeatureList {
    static constexpr std::size_t width = sizeof...(Sources);
};

template <class Context, class... Sources, std::size_t... I>
STACKDSL_HOT void load_features_impl(
    const Context& ctx,
    std::size_t lane,
    std::array<double, sizeof...(Sources)>& values,
    FeatureList<Sources...>,
    std::index_sequence<I...>
) noexcept {
    ((values[I] = ctx.template read<Sources>(lane)), ...);
}

template <class Context, class... Sources>
STACKDSL_HOT void load_features(
    const Context& ctx,
    std::size_t lane,
    std::array<double, sizeof...(Sources)>& values,
    FeatureList<Sources...> sources = {}
) noexcept {
    load_features_impl(
        ctx,
        lane,
        values,
        sources,
        std::index_sequence_for<Sources...>{});
}

template <std::size_t N, class Features, class Out, class Execution = DirectExecution<N>>
struct CatNode;

template <std::size_t N, class Out, class Execution, class... Sources>
struct CatNode<N, FeatureList<Sources...>, Out, Execution> {
    static constexpr std::size_t K = sizeof...(Sources);
    static_assert(K > 0, "cat requires at least one feature");

    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        (void)sizeof(Execution);
        static_assert(
            std::is_same_v<Out, OutputDst>,
            "matrix cat output is currently materialized only at a program/groupby root");
        double* STACKDSL_RESTRICT out = ctx.output;
        for (std::size_t lane = 0; lane < N; ++lane) {
            std::array<double, K> values{};
            load_features(ctx, lane, values, FeatureList<Sources...>{});
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
            for (std::size_t feature = 0; feature < K; ++feature) {
                out[lane * K + feature] = values[feature];
            }
        }
    }
};

}  // namespace stackdsl
