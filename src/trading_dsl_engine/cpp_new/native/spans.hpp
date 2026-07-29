#pragma once
#include <cassert>
#include <cstddef>
namespace tde::cpp_new {
template<class T> struct Span { T* data{}; std::size_t size{}; T& operator[](std::size_t i) const noexcept { assert(i<size); return data[i]; } };
}
