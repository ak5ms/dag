#pragma once

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "stackdsl/utils.hpp"

namespace stackdsl {

template <std::size_t N, class In, class Out, std::uint64_t SpanBits, int MinPeriods, bool IgnoreNa, bool Adjust>
struct EwmNode {
    static constexpr double span = std::bit_cast<double>(SpanBits);
    static_assert(span > 0.0);
    static constexpr double alpha = 2.0 / (span + 1.0);
    static constexpr double old_weight_factor = 1.0 - alpha;

    alignas(64) std::array<double, N> value{};
    alignas(64) std::array<double, N> weight{};
    alignas(64) std::array<std::int64_t, N> count{};
    alignas(64) std::array<std::uint8_t, N> initialized{};
    bool all_initialized=false;

    void setup() noexcept {
        value.fill(0.0);
        weight.fill(0.0);
        count.fill(0);
        initialized.fill(0);
        all_initialized=false;
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out=ctx.template write_ptr<Out>();

        // This is the overwhelmingly common trading path. Once every lane has
        // observed a finite value, an all-finite row is the ordinary recursive
        // EWM recurrence and requires no weights, counters, or per-lane branches.
        if constexpr (MinPeriods<=0 && IgnoreNa && !Adjust) {
            std::array<double,N> input{};
            bool all_finite=true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
            for (std::size_t i=0;i<N;++i) {
                input[i]=ctx.template read<In>(i);
                all_finite=all_finite && finite(input[i]);
            }
            if (all_initialized && all_finite) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
                for (std::size_t i=0;i<N;++i) {
                    const double next=std::fma(alpha,input[i]-value[i],value[i]);
                    value[i]=next;
                    out[i]=next;
                }
                return;
            }

            bool now_all_initialized=true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
            for (std::size_t i=0;i<N;++i) {
                if (finite(input[i])) {
                    if (initialized[i]) value[i]=std::fma(alpha,input[i]-value[i],value[i]);
                    else { value[i]=input[i]; initialized[i]=1; }
                }
                out[i]=initialized[i]?value[i]:kNaN;
                now_all_initialized=now_all_initialized && initialized[i];
            }
            all_initialized=now_all_initialized;
            return;
        }

        for (std::size_t i=0;i<N;++i) {
            const double x=ctx.template read<In>(i);
            const bool observation=finite(x);
            double old_weight=weight[i];
            if (initialized[i] && (observation || !IgnoreNa)) old_weight*=old_weight_factor;
            if (observation) {
                if (initialized[i]) {
                    double new_weight=Adjust?1.0:alpha;
                    if constexpr (!Adjust) {
                        if (std::abs(alpha-0.5)<=1e-12) new_weight=1.0-old_weight;
                    }
                    if (value[i]!=x) value[i]=(old_weight*value[i]+new_weight*x)/(old_weight+new_weight);
                    old_weight=Adjust?old_weight+new_weight:1.0;
                } else {
                    value[i]=x;
                    initialized[i]=1;
                    old_weight=1.0;
                }
                ++count[i];
            }
            weight[i]=old_weight;
            const bool enough=MinPeriods<=0 || count[i]>=MinPeriods;
            out[i]=initialized[i] && enough?value[i]:kNaN;
        }
    }
};

// Generated grouped plans use this node rather than the generic legacy node in
// groupby.hpp. The common recursive policy omits weight/count traffic while the
// fallback preserves adjust/min_periods/ignore_na semantics.
template <std::size_t N,std::size_t Capacity,class In,class Out,std::uint64_t SpanBits,int MinPeriods,bool IgnoreNa,bool Adjust>
struct FastGroupedEwmNode {
    static constexpr double span=std::bit_cast<double>(SpanBits);
    static_assert(span>0.0);
    static constexpr double alpha=2.0/(span+1.0);
    static constexpr double old_weight_factor=1.0-alpha;
    static constexpr std::size_t state_size=N*Capacity;

    alignas(64) std::array<double,state_size> value{};
    alignas(64) std::array<double,state_size> weight{};
    alignas(64) std::array<std::int64_t,state_size> count{};
    alignas(64) std::array<std::uint8_t,state_size> initialized{};

    void setup() noexcept {
        value.fill(0.0);
        weight.fill(0.0);
        count.fill(0);
        initialized.fill(0);
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out=ctx.template write_ptr<Out>();

        if constexpr (MinPeriods<=0 && IgnoreNa && !Adjust) {
            for (std::size_t lane=0;lane<N;++lane) {
                const std::size_t index=static_cast<std::size_t>((*ctx.group_slots)[lane])*N+lane;
                const double x=ctx.template read<In>(lane);
                if (finite(x)) {
                    if (initialized[index]) value[index]=std::fma(alpha,x-value[index],value[index]);
                    else { value[index]=x; initialized[index]=1; }
                }
                out[lane]=initialized[index]?value[index]:kNaN;
            }
            return;
        }

        for (std::size_t lane=0;lane<N;++lane) {
            const std::size_t index=static_cast<std::size_t>((*ctx.group_slots)[lane])*N+lane;
            const double x=ctx.template read<In>(lane);
            const bool observation=finite(x);
            double old_weight=weight[index];
            if (initialized[index] && (observation || !IgnoreNa)) old_weight*=old_weight_factor;
            if (observation) {
                if (initialized[index]) {
                    double new_weight=Adjust?1.0:alpha;
                    if constexpr (!Adjust) {
                        if (std::abs(alpha-0.5)<=1e-12) new_weight=1.0-old_weight;
                    }
                    if (value[index]!=x) value[index]=(old_weight*value[index]+new_weight*x)/(old_weight+new_weight);
                    old_weight=Adjust?old_weight+new_weight:1.0;
                } else {
                    value[index]=x;
                    initialized[index]=1;
                    old_weight=1.0;
                }
                ++count[index];
            }
            weight[index]=old_weight;
            const bool enough=MinPeriods<=0 || count[index]>=MinPeriods;
            out[lane]=initialized[index] && enough?value[index]:kNaN;
        }
    }
};

}  // namespace stackdsl
