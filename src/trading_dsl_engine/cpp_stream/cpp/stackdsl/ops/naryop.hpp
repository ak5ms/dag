#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

#include "stackdsl/utils.hpp"

namespace stackdsl {

struct AddOp { static constexpr int arity=2; STACKDSL_HOT static double apply(double a,double b) noexcept { return a+b; } };
struct SubOp { static constexpr int arity=2; STACKDSL_HOT static double apply(double a,double b) noexcept { return a-b; } };
struct MulOp { static constexpr int arity=2; STACKDSL_HOT static double apply(double a,double b) noexcept { return a*b; } };
struct DivOp { static constexpr int arity=2; STACKDSL_HOT static double apply(double a,double b) noexcept { return b==0.0?kNaN:a/b; } };

template <std::size_t N,class Lhs,class Rhs,class Out,class Op>
struct BinaryNode {
    static_assert(Op::arity==2);
    STACKDSL_HOT void setup() noexcept {}
    template <class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out=ctx.template write_ptr<Out>();
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC ivdep
#endif
        for (std::size_t i=0;i<N;++i) out[i]=Op::apply(ctx.template read<Lhs>(i),ctx.template read<Rhs>(i));
    }
};

template <std::size_t N,class In,class Out>
struct CopyNode {
    STACKDSL_HOT void setup() noexcept {}
    template <class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out=ctx.template write_ptr<Out>();
        for (std::size_t i=0;i<N;++i) out[i]=ctx.template read<In>(i);
    }
};

struct RankItem { double value; std::uint32_t lane; };

template <std::size_t N,class In,class Out>
struct XsRankNode {
    static constexpr int arity=1;
    RankScoreTable<N> scores{};
    void setup() noexcept { scores.setup(); }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out=ctx.template write_ptr<Out>();
        if constexpr (N <= 16) rank_count(ctx, out);
        else rank_sort(ctx, out);
    }

private:
    template <class Context>
    STACKDSL_HOT void rank_count(Context& ctx, double* STACKDSL_RESTRICT out) noexcept {
        std::array<double,N> values{};
        bool all_finite=true;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
        for (std::size_t i=0;i<N;++i) {
            values[i]=ctx.template read<In>(i);
            all_finite=all_finite && finite(values[i]);
        }

        if (all_finite) {
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
            for (std::size_t i=0;i<N;++i) {
                std::size_t upper=0;
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC unroll 16
#endif
                for (std::size_t j=0;j<N;++j) upper+=static_cast<std::size_t>(values[j]<=values[i]);
                out[i]=scores.get(N,upper-1);
            }
            return;
        }

        std::array<std::uint8_t,N> valid{};
        std::size_t count=0;
        for (std::size_t i=0;i<N;++i) {
            valid[i]=static_cast<std::uint8_t>(finite(values[i]));
            count+=valid[i];
            if (!valid[i]) out[i]=kNaN;
        }
        for (std::size_t i=0;i<N;++i) {
            if (!valid[i]) continue;
            std::size_t upper=0;
            const double value=values[i];
            for (std::size_t j=0;j<N;++j) upper+=static_cast<std::size_t>(valid[j] && values[j]<=value);
            out[i]=scores.get(count,upper-1);
        }
    }

    template <class Context>
    STACKDSL_HOT void rank_sort(Context& ctx, double* STACKDSL_RESTRICT out) noexcept {
        std::array<RankItem,N> items{};
        std::size_t count=0;
        for (std::size_t i=0;i<N;++i) {
            const double value=ctx.template read<In>(i);
            if (finite(value)) items[count++]={value,static_cast<std::uint32_t>(i)};
            else out[i]=kNaN;
        }
        std::sort(items.begin(),items.begin()+static_cast<std::ptrdiff_t>(count),[](const RankItem& a,const RankItem& b){ return a.value<b.value; });
        std::size_t tie_start=0;
        while (tie_start<count) {
            std::size_t upper=tie_start+1;
            while (upper<count && items[upper].value==items[tie_start].value) ++upper;
            const double score=scores.get(count,upper-1);
            for (std::size_t pos=tie_start;pos<upper;++pos) out[items[pos].lane]=score;
            tie_start=upper;
        }
    }
};

struct FastGroupRankItem { std::uint32_t group; double value; std::uint16_t lane; };

// Optimized grouped rank used by generated grouped RHS plans. It is kept here
// with the other stateless cross-sectional kernels so groupby.hpp only owns key
// resolution and group execution plumbing.
template <std::size_t N,std::size_t Capacity,class In,class Out>
struct FastGroupedXsRankNode {
    RankScoreTable<N> scores{};
    void setup() noexcept { scores.setup(); }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        double* STACKDSL_RESTRICT out=ctx.template write_ptr<Out>();
        if constexpr (N <= 16) rank_count(ctx,out);
        else rank_sort(ctx,out);
    }

private:
    template <class Context>
    STACKDSL_HOT void rank_count(Context& ctx,double* STACKDSL_RESTRICT out) noexcept {
        std::array<double,N> values{};
        std::array<std::uint32_t,N> groups{};
        bool all_finite=true;
        for (std::size_t lane=0;lane<N;++lane) {
            values[lane]=ctx.template read<In>(lane);
            all_finite=all_finite && finite(values[lane]);
            groups[lane]=static_cast<std::uint32_t>((*ctx.partitions)[lane])*static_cast<std::uint32_t>(Capacity)
                +static_cast<std::uint32_t>((*ctx.group_slots)[lane]);
        }

        if (all_finite) {
            for (std::size_t lane=0;lane<N;++lane) {
                std::size_t count=0,upper=0;
                for (std::size_t other=0;other<N;++other) {
                    const bool same=groups[other]==groups[lane];
                    count+=static_cast<std::size_t>(same);
                    upper+=static_cast<std::size_t>(same && values[other]<=values[lane]);
                }
                out[lane]=scores.get(count,upper-1);
            }
            return;
        }

        std::array<std::uint8_t,N> valid{};
        for (std::size_t lane=0;lane<N;++lane) {
            valid[lane]=static_cast<std::uint8_t>(finite(values[lane]));
            if (!valid[lane]) out[lane]=kNaN;
        }
        for (std::size_t lane=0;lane<N;++lane) {
            if (!valid[lane]) continue;
            std::size_t count=0,upper=0;
            for (std::size_t other=0;other<N;++other) {
                if (!valid[other] || groups[other]!=groups[lane]) continue;
                ++count;
                upper+=static_cast<std::size_t>(values[other]<=values[lane]);
            }
            out[lane]=scores.get(count,upper-1);
        }
    }

    template <class Context>
    STACKDSL_HOT void rank_sort(Context& ctx,double* STACKDSL_RESTRICT out) noexcept {
        std::array<FastGroupRankItem,N> items{};
        std::size_t count=0;
        for (std::size_t lane=0;lane<N;++lane) {
            const double value=ctx.template read<In>(lane);
            if (!finite(value)) { out[lane]=kNaN; continue; }
            const std::uint32_t group=static_cast<std::uint32_t>((*ctx.partitions)[lane])*static_cast<std::uint32_t>(Capacity)
                +static_cast<std::uint32_t>((*ctx.group_slots)[lane]);
            items[count++]={group,value,static_cast<std::uint16_t>(lane)};
        }
        std::sort(items.begin(),items.begin()+static_cast<std::ptrdiff_t>(count),[](const FastGroupRankItem& a,const FastGroupRankItem& b){
            return a.group<b.group || (a.group==b.group && a.value<b.value);
        });
        std::size_t group_start=0;
        while (group_start<count) {
            std::size_t group_end=group_start+1;
            while (group_end<count && items[group_end].group==items[group_start].group) ++group_end;
            const std::size_t group_count=group_end-group_start;
            std::size_t tie_start=group_start;
            while (tie_start<group_end) {
                std::size_t upper=tie_start+1;
                while (upper<group_end && items[upper].value==items[tie_start].value) ++upper;
                const double score=scores.get(group_count,upper-group_start-1);
                for (std::size_t pos=tie_start;pos<upper;++pos) out[items[pos].lane]=score;
                tie_start=upper;
            }
            group_start=group_end;
        }
    }
};

}  // namespace stackdsl
