#pragma once
#include "../spans.hpp"
#include <cstdint>
namespace tde::cpp_new {
struct RidgeState { Span<double> xx,xy,xx_clock,xy_clock,beta,predictions,solve_scratch; Span<std::uint8_t> has_xx,has_xy; std::int64_t tick{}; };
struct RidgeScratch { Span<double> worker_moments, features; };
struct RidgeValue { Span<const double> beta,predictions; };
// Formula emission supplies fixed feature width and a non-allocating Eigen solve.
// Pairwise moments deliberately have independent validity/clock lanes.
template<class... Inputs> inline RidgeValue ridge_tick(RidgeState&, Inputs&&...) noexcept { return {}; }
}
