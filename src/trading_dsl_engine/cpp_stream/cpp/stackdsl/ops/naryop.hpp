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
        std::array<std::uint8_t,N> valid{};
        std::size_t count=0;
        for (std::size_t i=0;i<N;++i) {
            values[i]=ctx.template read<In>(i);
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

}  // namespace stackdsl
