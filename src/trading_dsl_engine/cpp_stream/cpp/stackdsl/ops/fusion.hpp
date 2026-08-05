#pragma once

#include "stackdsl/utils.hpp"

namespace stackdsl {

// A generic physical-stage wrapper used after dependency analysis proves that
// two adjacent nodes can execute in the same row scope.  Both operators keep
// their canonical implementation; the wrapper merely exposes one inlinable
// call boundary so the C++ optimizer can scalar-replace scratch and combine the
// producer's update loop with a stateless epilogue.
template <class First, class Second>
struct FusedStageNode {
    First first{};
    Second second{};

    STACKDSL_HOT void setup() noexcept {
        first.setup();
        second.setup();
    }

    template <class Context>
    STACKDSL_HOT void on_data(Context& ctx) noexcept {
        first.on_data(ctx);
        second.on_data(ctx);
    }
};

}  // namespace stackdsl
