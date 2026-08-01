#pragma once

#include <array>
#include <cstddef>

#include "stackdsl/engine.hpp"
#include "stackdsl/ops/cat.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template<std::size_t N,class LeftFeatures,class RightFeatures,class Out,class Execution=DirectExecution<N>>
struct EinsumNfNfToNNode;

template<std::size_t N,class Out,class Execution,class... Left,class... Right>
struct EinsumNfNfToNNode<N,FeatureList<Left...>,FeatureList<Right...>,Out,Execution> {
    static constexpr std::size_t K=FeatureList<Left...>::width;
    static_assert(K==FeatureList<Right...>::width);
    STACKDSL_HOT void setup() noexcept {}
    template<class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        (void)sizeof(Execution);
        auto* out=ctx.template write_ptr<Out>();
        for(std::size_t lane=0;lane<N;++lane){
            std::array<double,K> left{},right{};
            load_features(ctx,lane,left,FeatureList<Left...>{});
            load_features(ctx,lane,right,FeatureList<Right...>{});
            double value=0.0;
            for(std::size_t j=0;j<K;++j)value=std::fma(left[j],right[j],value);
            out[lane]=value;
        }
    }
};

}  // namespace stackdsl
