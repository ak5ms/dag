#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>

#include "stackdsl/utils.hpp"

namespace stackdsl {

// Execution is a plan-level concern, not an operator variant. Every generated
// node receives one of these scopes through the same final template parameter.
template <std::size_t N>
struct DirectExecution {
    static constexpr std::size_t lane_count = N;
    static constexpr std::size_t state_size = N;
    static constexpr std::size_t cross_state_size = 1;
    static constexpr bool contiguous_lanes = true;

    template <class Context>
    STACKDSL_HOT static std::size_t lane_begin(const Context& ctx) noexcept {
        return ctx.lane_begin;
    }

    template <class Context>
    STACKDSL_HOT static std::size_t lane_end(const Context& ctx) noexcept {
        return ctx.lane_end;
    }

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
    static constexpr std::size_t lane_count = N;
    static constexpr std::size_t state_size = N * Capacity;
    static constexpr std::size_t cross_state_size = PartitionCount * Capacity;
    static constexpr bool contiguous_lanes = false;

    template <class Context>
    STACKDSL_HOT static std::size_t lane_begin(const Context& ctx) noexcept {
        return ctx.lane_begin;
    }

    template <class Context>
    STACKDSL_HOT static std::size_t lane_end(const Context& ctx) noexcept {
        return ctx.lane_end;
    }

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

// Row-scalar physical nodes are instantiated with NodeN=1 even when the plan's
// execution scope has N instrument lanes. Every lane worker computes such a value
// locally, while ordinary vector nodes receive that worker's assigned lane range.
template <std::size_t NodeN, class Execution, class Context>
STACKDSL_HOT std::size_t execution_lane_begin(const Context& ctx) noexcept {
    if constexpr (NodeN == Execution::lane_count) return Execution::lane_begin(ctx);
    else return 0;
}

template <std::size_t NodeN, class Execution, class Context>
STACKDSL_HOT std::size_t execution_lane_end(const Context& ctx) noexcept {
    if constexpr (NodeN == Execution::lane_count) return Execution::lane_end(ctx);
    else return NodeN;
}

template <std::size_t OutputSize, class Execution, class Context>
STACKDSL_HOT std::pair<std::size_t, std::size_t> execution_output_range(
    const Context& ctx
) noexcept {
    const std::size_t begin = Execution::lane_begin(ctx);
    const std::size_t end = Execution::lane_end(ctx);
    if (begin == 0 && end == Execution::lane_count) return {0, OutputSize};
    if constexpr (OutputSize % Execution::lane_count == 0) {
        constexpr std::size_t per_lane = OutputSize / Execution::lane_count;
        return {begin * per_lane, end * per_lane};
    } else {
        // The planner never lane-shards such a stage. Keeping a full range here
        // preserves serial/row-sharded behavior for scalar and irregular outputs.
        return {0, OutputSize};
    }
}

template <class... Stages>
STACKDSL_HOT void setup_stages(Stages&... stages) noexcept {
    (stages.setup(), ...);
}

template <class Context, class... Stages>
STACKDSL_HOT void run_stages(Context& ctx, Stages&... stages) noexcept {
    (stages.on_data(ctx), ...);
}

}  // namespace stackdsl
