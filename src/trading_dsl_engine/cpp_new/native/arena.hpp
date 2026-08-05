#pragma once
#include "spans.hpp"
#include <cstddef>
#include <cstdint>
namespace tde::cpp_new {
struct AlignedArena { std::byte* base{}; std::size_t bytes{}; template<class T> Span<T> at(std::size_t off, std::size_t n) const noexcept { return {reinterpret_cast<T*>(base+off),n}; } Span<double> span(std::size_t) const noexcept { return {}; } };
}
