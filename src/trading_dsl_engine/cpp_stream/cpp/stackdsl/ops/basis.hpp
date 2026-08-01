#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>

#include "stackdsl/utils.hpp"

namespace stackdsl {

template<std::size_t K,class EventTs,class SessionStart,class SessionEnd>
struct RbfBasisSrc {
    static constexpr std::size_t feature_width=K;
    using value_type=double;
    template<class Context>
    STACKDSL_HOT static void load_features(const Context& ctx,std::size_t lane,double* out) noexcept {
        const double ts=ctx.template read<EventTs>(lane);
        const double start=ctx.template read<SessionStart>(lane);
        const double end=ctx.template read<SessionEnd>(lane);
        const double length=end-start;
        if(!finite(ts)||!finite(start)||!finite(end)||!(length>0.0)||ts<start||ts>=end){
            for(std::size_t j=0;j<K;++j)out[j]=kNaN;
            return;
        }
        const double phase=std::clamp((ts-start)/length,0.0,1.0);
        constexpr double sigma=1.0/static_cast<double>(K>1?K-1:1);
        double total=0.0;
        for(std::size_t j=0;j<K;++j){
            const double center=K==1?0.0:static_cast<double>(j)/static_cast<double>(K-1);
            const double z=(phase-center)/sigma;
            out[j]=std::exp(-0.5*z*z);
            total+=out[j];
        }
        if(total<=1e-18){ for(std::size_t j=0;j<K;++j)out[j]=1.0/static_cast<double>(K); }
        else for(std::size_t j=0;j<K;++j)out[j]/=total;
    }
};

template<std::size_t K,std::size_t Steps,class EventTs,class SessionStart,class SessionEnd>
struct FutureRbfBasisSumSrc {
    static constexpr std::size_t feature_width=K;
    using value_type=double;

    static std::array<double,(Steps+1)*K> make_table() {
        std::array<double,(Steps+1)*K> table{};
        std::array<double,K> running{};
        constexpr double sigma=1.0/static_cast<double>(K>1?K-1:1);
        for(std::size_t reverse=0;reverse<Steps;++reverse){
            const std::size_t row=Steps-1-reverse;
            const double phase=static_cast<double>(row)/static_cast<double>(Steps);
            std::array<double,K> values{};
            double total=0.0;
            for(std::size_t j=0;j<K;++j){
                const double center=K==1?0.0:static_cast<double>(j)/static_cast<double>(K-1);
                const double z=(phase-center)/sigma;
                values[j]=std::exp(-0.5*z*z);
                total+=values[j];
            }
            for(std::size_t j=0;j<K;++j){
                running[j]+=total<=1e-18?1.0/static_cast<double>(K):values[j]/total;
                table[row*K+j]=running[j];
            }
        }
        return table;
    }
    inline static const std::array<double,(Steps+1)*K> table=make_table();

    template<class Context>
    STACKDSL_HOT static void load_features(const Context& ctx,std::size_t lane,double* out) noexcept {
        const double ts=ctx.template read<EventTs>(lane);
        const double start=ctx.template read<SessionStart>(lane);
        const double end=ctx.template read<SessionEnd>(lane);
        const double length=end-start;
        if(!finite(ts)||!finite(start)||!finite(end)||!(length>0.0)){
            for(std::size_t j=0;j<K;++j)out[j]=kNaN;
            return;
        }
        std::size_t index;
        if(ts<start) index=0;
        else if(ts>=end) index=Steps;
        else {
            const double phase=std::clamp((ts-start)/length,0.0,1.0);
            index=std::min<std::size_t>(Steps,static_cast<std::size_t>(std::floor(phase*Steps))+1);
        }
        for(std::size_t j=0;j<K;++j)out[j]=table[index*K+j];
    }
};

}  // namespace stackdsl
