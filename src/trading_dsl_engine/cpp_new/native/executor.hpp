#pragma once
#include "abi.hpp"
#include <cstddef>
namespace tde::cpp_new { struct StaticSchedule { std::size_t workers{1}; std::size_t instrument_threshold{2048}; std::size_t lane_threshold{4}; }; }
