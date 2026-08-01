#pragma once

#include <array>
#include <cstddef>

#include "stackdsl/engine.hpp"
#include "stackdsl/ops/cat.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

template<std::size_t N,class Left,class Right,class Out,class Execution=DirectExecution<N>>
struct EinsumNfNfToNNode {
    static constexpr std::size_t K=source_width_v<Left>;
    static_assert(K==source_width_v<Right>);
    STACKDSL_HOT void setup() noexcept {}
    template<class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        (void)sizeof(Execution);
        auto* out=ctx.template write_ptr<Out>();
        for(std::size_t lane=0;lane<N;++lane){
            std::array<double,K> left{},right{};
            load_source_features<Left>(ctx,lane,left.data());
            load_source_features<Right>(ctx,lane,right.data());
            double value=0.0;
            for(std::size_t j=0;j<K;++j)value=std::fma(left[j],right[j],value);
            out[lane]=value;
        }
    }
};

}  // namespace stackdsl
