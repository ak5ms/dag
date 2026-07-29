#pragma once
#include "../arena.hpp"
#include <cmath>
#include <cstdint>
namespace tde::cpp_new {
struct EwmState { Span<double> value, weight; Span<std::uint8_t> initialized; Span<std::int64_t> count; };
template<class InputSpan> inline void ewm_tick(EwmState s, InputSpan x, Span<double> out, double span, bool adjust=true, bool ignore_na=false) noexcept {
 const double alpha=2.0/(span+1.0), old=1.0-alpha;
 for(std::size_t i=0;i<x.size;++i){ const bool ok=std::isfinite(x[i]); double w=s.weight[i]; if(s.initialized[i] && (ok||!ignore_na)) w*=old; if(ok){ s.value[i]=s.initialized[i] ? (adjust?(w*s.value[i]+x[i])/(w+1.0):(w*s.value[i]+alpha*x[i])/(w+alpha)) : x[i]; s.weight[i]=adjust?w+1.0:1.0; s.initialized[i]=1; ++s.count[i]; } else s.weight[i]=w; out[i]=s.initialized[i]?s.value[i]:NAN; }
}
}
