#pragma once

// Aggregation header used by generated formula translation units.
#include "stackdsl/utils.hpp"
#include "stackdsl/engine.hpp"
#include "stackdsl/io.hpp"
#include "stackdsl/parallel.hpp"
#include "stackdsl/ops/literal.hpp"
#include "stackdsl/ops/naryop.hpp"
#include "stackdsl/ops/cross_sectional.hpp"
#include "stackdsl/ops/cat.hpp"
#include "stackdsl/ops/cumsum.hpp"
#include "stackdsl/ops/reduction.hpp"
#include "stackdsl/ops/history.hpp"
#include "stackdsl/ops/advanced_history.hpp"
#include "stackdsl/ops/ewm.hpp"
#include "stackdsl/ops/statistics.hpp"
#include "stackdsl/ops/basis.hpp"
#include "stackdsl/ops/instrument_basis_mean.hpp"
#include "stackdsl/ops/einsum.hpp"
#include "stackdsl/ops/dense_tensor.hpp"
#include "stackdsl/ops/tensor_ops.hpp"
#include "stackdsl/ops/custom.hpp"
#include "stackdsl/ops/ridge.hpp"
#include "stackdsl/ops/groupby.hpp"

// Compile-time worker-state merging is kept out of generated Jinja. This include
// appears after the node definitions so existing node adapters can be selected,
// while future nodes may provide merge_state_from(source, partition) themselves.
#include "stackdsl/state_merge.hpp"
