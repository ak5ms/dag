#pragma once

#include <cmath>
#include <cstddef>

#include "stackdsl/engine.hpp"
#include "stackdsl/utils.hpp"

namespace stackdsl {

struct VolumeForFitSessionPolicy {
    static constexpr std::size_t arity=4;
    STACKDSL_HOT static double apply(double volume,double ts,double start,double end) noexcept {
        const bool inside=finite(ts)&&finite(start)&&finite(end)&&end>start&&ts>=start&&ts<end;
        if(!inside)return kNaN;
        return std::max(finite(volume)?volume:0.0,0.0);
    }
};

struct VolumeForSeenSessionPolicy {
    static constexpr std::size_t arity=5;
    STACKDSL_HOT static double apply(double volume,double ts,double start,double end,double tradable) noexcept {
        const bool inside=finite(ts)&&finite(start)&&finite(end)&&end>start&&ts>=start&&ts<end;
        if(!(inside&&finite(tradable)&&tradable==1.0))return 0.0;
        return std::max(finite(volume)?volume:0.0,0.0);
    }
};

struct NonnegativePolicy {
    static constexpr std::size_t arity=1;
    STACKDSL_HOT static double apply(double value) noexcept { return std::max(finite(value)?value:0.0,0.0); }
};

struct PctSeenSessionVolumePolicy {
    static constexpr std::size_t arity=4;
    STACKDSL_HOT static double apply(double seen,double forecast,double ts,double start) noexcept {
        const double total=seen+forecast;
        return finite(ts)&&finite(start)&&ts>=start&&total>0.0?seen/total:kNaN;
    }
};

template<std::size_t N,class Out,class Policy,class Execution,class... Inputs>
struct StatelessNode {
    static_assert(sizeof...(Inputs)==Policy::arity);
    STACKDSL_HOT void setup() noexcept {}
    template<class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        (void)sizeof(Execution);
        auto* out=ctx.template write_ptr<Out>();
        for(std::size_t lane=0;lane<N;++lane)out[lane]=Policy::apply(ctx.template read<Inputs>(lane)...);
    }
};

}  // namespace stackdsl
