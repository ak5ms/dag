#pragma once

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template <class... Sources>
struct FeatureList {
    static constexpr std::size_t width = (source_width_v<Sources> + ... + 0);
};

template <class Source, class Context>
STACKDSL_HOT double read_source_feature(
    const Context& ctx,
    std::size_t lane,
    std::size_t feature
) noexcept {
    if constexpr (requires { Source::read_feature(ctx, lane, feature); }) {
        return Source::read_feature(ctx, lane, feature);
    } else if constexpr (source_width_v<Source> == 1) {
        (void)feature;
        return ctx.template read<Source>(lane);
    } else {
        return ctx.template read_feature<Source>(lane, feature);
    }
}

template <class Source, class Context>
STACKDSL_HOT void load_source_features(
    const Context& ctx,
    std::size_t lane,
    double* STACKDSL_RESTRICT out
) noexcept {
    if constexpr (requires { Source::load_features(ctx, lane, out); }) {
        Source::load_features(ctx, lane, out);
    } else {
        for (std::size_t feature = 0; feature < source_width_v<Source>; ++feature) {
            out[feature] = read_source_feature<Source>(ctx, lane, feature);
        }
    }
}

template <class Context>
STACKDSL_HOT double read_feature_at(
    const Context&, std::size_t, std::size_t, FeatureList<>
) noexcept {
    return kNaN;
}

template <class Context, class First, class... Rest>
STACKDSL_HOT double read_feature_at(
    const Context& ctx,
    std::size_t lane,
    std::size_t feature,
    FeatureList<First, Rest...>
) noexcept {
    if (feature < source_width_v<First>) {
        return read_source_feature<First>(ctx, lane, feature);
    }
    return read_feature_at(
        ctx,
        lane,
        feature - source_width_v<First>,
        FeatureList<Rest...>{}
    );
}

template <class Context, class... Sources>
STACKDSL_HOT void load_features(
    const Context& ctx,
    std::size_t lane,
    std::array<double, FeatureList<Sources...>::width>& values,
    FeatureList<Sources...> = {}
) noexcept {
    std::size_t offset = 0;
    ((load_source_features<Sources>(ctx, lane, values.data() + offset), offset += source_width_v<Sources>), ...);
}

template <std::size_t N, class Features, class Out, class Execution = DirectExecution<N>>
struct CatNode;

template <std::size_t N, class Out, class Execution, class... Sources>
struct CatNode<N, FeatureList<Sources...>, Out, Execution> {
    static constexpr std::size_t K = FeatureList<Sources...>::width;
    static_assert(K > 0);
    STACKDSL_HOT void setup() noexcept {}

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out = ctx.template write_ptr<Out>();
        const std::size_t begin = execution_lane_begin<N, Execution>(ctx);
        const std::size_t end = execution_lane_end<N, Execution>(ctx);
        for (std::size_t lane = begin; lane < end; ++lane) {
            std::array<double, K> values{};
            load_features(ctx, lane, values, FeatureList<Sources...>{});
            for (std::size_t feature = 0; feature < K; ++feature) {
                out[lane * K + feature] = values[feature];
            }
        }
    }
};

}  // namespace stackdsl
