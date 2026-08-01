#pragma once

#include <cstddef>
#include <limits>

namespace stackdsl {

struct NaNLiteralSrc {
    using value_type = double;
    static constexpr double value = std::numeric_limits<double>::quiet_NaN();
    static constexpr std::size_t feature_width = 1;
};

struct PositiveInfinityLiteralSrc {
    using value_type = double;
    static constexpr double value = std::numeric_limits<double>::infinity();
    static constexpr std::size_t feature_width = 1;
};

struct NegativeInfinityLiteralSrc {
    using value_type = double;
    static constexpr double value = -std::numeric_limits<double>::infinity();
    static constexpr std::size_t feature_width = 1;
};

}  // namespace stackdsl
