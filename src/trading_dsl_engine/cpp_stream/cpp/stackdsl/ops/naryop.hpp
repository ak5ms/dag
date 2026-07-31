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
    template <class Context> STACKDSL_HOT void on_data(Context& ctx) noexcept {
        std::array<RankItem,N> items{};
        double* STACKDSL_RESTRICT out=ctx.template write_ptr<Out>();
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
