#pragma once

#include <cstddef>
#include <cstdint>

#include "stackdsl/utils.hpp"

namespace stackdsl {

// Execution is a plan-level concern, not an operator variant. Every generated
// node receives one of these scopes through the same final template parameter.
template <std::size_t N>
struct DirectExecution {
    static constexpr std::size_t state_size = N;
    static constexpr std::size_t cross_state_size = 1;
    static constexpr bool contiguous_lanes = true;

    template <class Context>
    STACKDSL_HOT static std::size_t state_index(const Context&, std::size_t lane) noexcept {
        return lane;
    }

    template <class Context>
    STACKDSL_HOT static std::uint32_t rank_group(const Context&, std::size_t) noexcept {
        return 0;
    }

    template <class Context>
    STACKDSL_HOT static std::uint32_t cross_group(const Context&, std::size_t) noexcept {
        return 0;
    }
};

template <std::size_t N, std::size_t Capacity, std::size_t PartitionCount>
struct GroupedExecution {
    static_assert(PartitionCount > 0);
    static constexpr std::size_t state_size = N * Capacity;
    static constexpr std::size_t cross_state_size = PartitionCount * Capacity;
    static constexpr bool contiguous_lanes = false;

    template <class Context>
    STACKDSL_HOT static std::size_t state_index(const Context& ctx, std::size_t lane) noexcept {
        return static_cast<std::size_t>((*ctx.group_slots)[lane]) * N + lane;
    }

    template <class Context>
    STACKDSL_HOT static std::uint32_t rank_group(const Context& ctx, std::size_t lane) noexcept {
        return static_cast<std::uint32_t>((*ctx.partitions)[lane]) * static_cast<std::uint32_t>(Capacity)
            + static_cast<std::uint32_t>((*ctx.group_slots)[lane]);
    }

    template <class Context>
    STACKDSL_HOT static std::uint32_t cross_group(const Context& ctx, std::size_t lane) noexcept {
        return rank_group(ctx, lane);
    }
};

template <class... Stages>
STACKDSL_HOT void setup_stages(Stages&... stages) noexcept {
    (stages.setup(), ...);
}

template <class Context, class... Stages>
STACKDSL_HOT void run_stages(Context& ctx, Stages&... stages) noexcept {
    (stages.on_data(ctx), ...);
}

}  // namespace stackdsl
