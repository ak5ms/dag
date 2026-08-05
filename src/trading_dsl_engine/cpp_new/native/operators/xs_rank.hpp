#pragma once
#include "../spans.hpp"
#include <algorithm>
#include <cmath>
namespace tde::cpp_new {
struct RankItem { double value; std::size_t instrument; };
struct XsRankScratch { Span<RankItem> items; Span<double> scores; };
template<class NormalScore> inline void xs_rank_tick(Span<const double> x, XsRankScratch s, Span<double> out, NormalScore score) noexcept {
 std::size_t n=0; for(std::size_t i=0;i<x.size;++i) if(std::isfinite(x[i])) s.items[n++]={x[i],i}; else out[i]=NAN;
 std::sort(s.items.data,s.items.data+n,[](auto a,auto b){ return a.value<b.value || (a.value==b.value&&a.instrument<b.instrument); });
 for(std::size_t begin=0;begin<n;){ std::size_t end=begin+1; while(end<n&&s.items[end].value==s.items[begin].value)++end; const double v=score(end,n+1); for(auto j=begin;j<end;++j) out[s.items[j].instrument]=v; begin=end; }
}
}
