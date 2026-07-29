#pragma once
#include "arena.hpp"
#include <cstddef>
namespace tde::cpp_new {
inline constexpr unsigned kAbiVersion=1;
struct InputRow { const double* const* columns{}; std::size_t width{}; Span<const double> column(std::size_t i) const noexcept { return {columns[i],width}; } };
struct OutputRow { double* data{}; std::size_t width{}; Span<double> span() const noexcept { return {data,width}; } template<class T> void assign(const T&) noexcept {} };
struct BatchInput { std::size_t rows{}; InputRow row(std::size_t) const noexcept { return {}; } };
struct BatchOutput { OutputRow row(std::size_t) noexcept { return {}; } };
struct alignas(64) WorkerContext { std::byte padding[64]{}; };
}
