#pragma once

#include "stackdsl/utils.hpp"

namespace stackdsl {

template <class... Stages>
STACKDSL_HOT void setup_stages(Stages&... stages) noexcept {
    (stages.setup(), ...);
}

template <class Context, class... Stages>
STACKDSL_HOT void run_stages(Context& ctx, Stages&... stages) noexcept {
    (stages.on_data(ctx), ...);
}

}  // namespace stackdsl
