#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "stackdsl/engine.hpp"
#include "stackdsl/ops/cat.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

struct InstrumentBasisBetaProjection {};
struct InstrumentBasisPredsProjection {};

template<std::size_t N,class Features,class Y,class Weights,class Out,std::uint64_t AlphaBits,class Projection,class Execution=DirectExecution<N>>
struct InstrumentBasisMeanNode;

template<std::size_t N,class Y,class Weights,class Out,std::uint64_t AlphaBits,class Projection,class Execution,class... FeatureSources>
struct InstrumentBasisMeanNode<N,FeatureList<FeatureSources...>,Y,Weights,Out,AlphaBits,Projection,Execution> {
    static constexpr std::size_t K=FeatureList<FeatureSources...>::width;
    static constexpr std::size_t StateSize=Execution::state_size;
    std::array<double,StateSize*K> num{};
    std::array<double,StateSize*K> den{};
    std::array<double,StateSize*K> beta{};
    std::array<std::uint8_t,StateSize*K> has_value{};

    void setup() noexcept { num.fill(0.0);den.fill(0.0);beta.fill(0.0);has_value.fill(0); }

    template<class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        constexpr double alpha=std::bit_cast<double>(AlphaBits);
        auto* out=ctx.template write_ptr<Out>();
        constexpr bool beta_projection=std::is_same_v<Projection,InstrumentBasisBetaProjection>;
        const std::size_t begin=execution_lane_begin<N,Execution>(ctx);
        const std::size_t end=execution_lane_end<N,Execution>(ctx);
        for(std::size_t lane=begin;lane<end;++lane){
            std::array<double,K> features{};
            load_features(ctx,lane,features,FeatureList<FeatureSources...>{});
            const double y=ctx.template read<Y>(lane);
            const double weight=ctx.template read<Weights>(lane);
            const bool valid_row=finite(y)&&finite(weight);
            const std::size_t state_index=Execution::state_index(ctx,lane);
            const std::size_t base=state_index*K;
            if constexpr(!beta_projection){
                bool all_features=true;
                double prediction=0.0;
                for(std::size_t j=0;j<K;++j){
                    all_features&=finite(features[j]);
                    prediction=std::fma(features[j],beta[base+j],prediction);
                }
                out[lane]=valid_row&&all_features?prediction:kNaN;
            }
            for(std::size_t j=0;j<K;++j){
                const double x=features[j];
                if(!valid_row||!finite(x)) continue;
                const std::size_t index=base+j;
                const double num_new=x*y*weight;
                const double den_new=x*weight;
                if(has_value[index]){
                    num[index]=std::fma(alpha,num_new-num[index],num[index]);
                    den[index]=std::fma(alpha,den_new-den[index],den[index]);
                }else{
                    num[index]=num_new;
                    den[index]=den_new;
                    has_value[index]=1;
                }
                const double candidate=den[index]!=0.0?num[index]/den[index]:kNaN;
                if(finite(candidate))beta[index]=candidate;
            }
            if constexpr(beta_projection){
                for(std::size_t j=0;j<K;++j)out[lane*K+j]=beta[base+j];
            }
        }
    }
};

}  // namespace stackdsl
