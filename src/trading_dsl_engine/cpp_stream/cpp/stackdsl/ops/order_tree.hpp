#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "stackdsl/utils.hpp"

namespace stackdsl {

// An allocation-free order-statistics treap.  Ring positions are stable node
// identifiers, so replacing the outgoing observation performs one erase and one
// insert without allocating or moving any node storage.  (value, id) is the key,
// which preserves duplicates exactly.
template <std::size_t Capacity>
struct FixedOrderTree {
    static_assert(Capacity > 0);
    using Index = std::uint32_t;
    static constexpr Index nil = std::numeric_limits<Index>::max();

    std::array<double, Capacity> value{};
    std::array<Index, Capacity> left{};
    std::array<Index, Capacity> right{};
    std::array<Index, Capacity> parent{};
    std::array<Index, Capacity> subtree_size{};
    std::array<std::uint32_t, Capacity> priority{};
    std::array<std::uint8_t, Capacity> active{};
    Index root = nil;
    Index count = 0;

    STACKDSL_HOT void setup() noexcept {
        value.fill(0.0);
        left.fill(nil);
        right.fill(nil);
        parent.fill(nil);
        subtree_size.fill(0);
        active.fill(0);
        for (Index node = 0; node < Capacity; ++node) {
            priority[node] = mixed_priority(node + 1U);
        }
        root = nil;
        count = 0;
    }

    STACKDSL_HOT bool contains(Index node) const noexcept {
        return active[node] != 0;
    }

    STACKDSL_HOT std::size_t size() const noexcept { return count; }

    STACKDSL_HOT void replace(Index node, double incoming) noexcept {
        if (active[node]) erase(node);
        if (finite(incoming)) insert(node, incoming);
    }

    STACKDSL_HOT double kth(std::size_t rank) const noexcept {
        Index node = root;
        while (node != nil) {
            const std::size_t left_count = node_size(left[node]);
            if (rank < left_count) {
                node = left[node];
            } else if (rank == left_count) {
                return value[node];
            } else {
                rank -= left_count + 1;
                node = right[node];
            }
        }
        return kNaN;
    }

    STACKDSL_HOT std::size_t upper_count(double candidate) const noexcept {
        std::size_t result = 0;
        Index node = root;
        while (node != nil) {
            if (value[node] <= candidate) {
                result += node_size(left[node]) + 1;
                node = right[node];
            } else {
                node = left[node];
            }
        }
        return result;
    }

    STACKDSL_HOT double minimum() const noexcept {
        Index node = root;
        if (node == nil) return kNaN;
        while (left[node] != nil) node = left[node];
        return value[node];
    }

    STACKDSL_HOT double maximum() const noexcept {
        Index node = root;
        if (node == nil) return kNaN;
        while (right[node] != nil) node = right[node];
        return value[node];
    }

    template <class Function>
    STACKDSL_HOT void for_each_active(Function&& function) const noexcept {
        for (Index node = 0; node < Capacity; ++node) {
            if (active[node]) function(value[node]);
        }
    }

private:
    STACKDSL_HOT static std::uint32_t mixed_priority(
        std::uint32_t value
    ) noexcept {
        value += 0x9e3779b9U;
        value = (value ^ (value >> 16U)) * 0x85ebca6bU;
        value = (value ^ (value >> 13U)) * 0xc2b2ae35U;
        return value ^ (value >> 16U);
    }

    STACKDSL_HOT std::size_t node_size(Index node) const noexcept {
        return node == nil ? 0U : subtree_size[node];
    }

    STACKDSL_HOT bool priority_less(Index lhs, Index rhs) const noexcept {
        return priority[lhs] < priority[rhs]
            || (priority[lhs] == priority[rhs] && lhs < rhs);
    }

    STACKDSL_HOT bool key_less(
        double lhs_value,
        Index lhs,
        double rhs_value,
        Index rhs
    ) const noexcept {
        return lhs_value < rhs_value
            || (lhs_value == rhs_value && lhs < rhs);
    }

    STACKDSL_HOT void refresh(Index node) noexcept {
        if (node != nil) {
            subtree_size[node] = static_cast<Index>(
                1U + node_size(left[node]) + node_size(right[node])
            );
        }
    }

    STACKDSL_HOT void refresh_to_root(Index node) noexcept {
        while (node != nil) {
            refresh(node);
            node = parent[node];
        }
    }

    STACKDSL_HOT void rotate_left(Index node) noexcept {
        const Index pivot = right[node];
        const Index ancestor = parent[node];
        right[node] = left[pivot];
        if (left[pivot] != nil) parent[left[pivot]] = node;
        left[pivot] = node;
        parent[node] = pivot;
        parent[pivot] = ancestor;
        if (ancestor == nil) root = pivot;
        else if (left[ancestor] == node) left[ancestor] = pivot;
        else right[ancestor] = pivot;
        refresh(node);
        refresh(pivot);
    }

    STACKDSL_HOT void rotate_right(Index node) noexcept {
        const Index pivot = left[node];
        const Index ancestor = parent[node];
        left[node] = right[pivot];
        if (right[pivot] != nil) parent[right[pivot]] = node;
        right[pivot] = node;
        parent[node] = pivot;
        parent[pivot] = ancestor;
        if (ancestor == nil) root = pivot;
        else if (left[ancestor] == node) left[ancestor] = pivot;
        else right[ancestor] = pivot;
        refresh(node);
        refresh(pivot);
    }

    STACKDSL_HOT void insert(Index node, double incoming) noexcept {
        value[node] = incoming;
        left[node] = right[node] = parent[node] = nil;
        subtree_size[node] = 1;
        active[node] = 1;
        ++count;
        if (root == nil) {
            root = node;
            return;
        }
        Index current = root;
        while (true) {
            if (key_less(incoming, node, value[current], current)) {
                if (left[current] == nil) {
                    left[current] = node;
                    parent[node] = current;
                    break;
                }
                current = left[current];
            } else {
                if (right[current] == nil) {
                    right[current] = node;
                    parent[node] = current;
                    break;
                }
                current = right[current];
            }
        }
        refresh_to_root(parent[node]);
        while (
            parent[node] != nil
            && priority_less(node, parent[node])
        ) {
            const Index ancestor = parent[node];
            if (left[ancestor] == node) rotate_right(ancestor);
            else rotate_left(ancestor);
        }
        refresh_to_root(parent[node]);
    }

    STACKDSL_HOT void erase(Index node) noexcept {
        while (left[node] != nil || right[node] != nil) {
            if (left[node] == nil) rotate_left(node);
            else if (right[node] == nil) rotate_right(node);
            else if (priority_less(left[node], right[node])) rotate_right(node);
            else rotate_left(node);
        }
        const Index ancestor = parent[node];
        if (ancestor == nil) root = nil;
        else if (left[ancestor] == node) left[ancestor] = nil;
        else right[ancestor] = nil;
        active[node] = 0;
        left[node] = right[node] = parent[node] = nil;
        subtree_size[node] = 0;
        --count;
        refresh_to_root(ancestor);
    }
};

struct EmptyOrderTree {
    STACKDSL_HOT void setup() noexcept {}
};

// Recency-ordered fixed list used by rolling kth/backfill.  A valid ring slot is
// linked once, unlinked when it leaves the row window, and kth selection walks
// only K nodes instead of rescanning Periods rows.
template <std::size_t Capacity>
struct FixedRecencyList {
    using Index = std::uint32_t;
    static constexpr Index nil = std::numeric_limits<Index>::max();
    std::array<double, Capacity> value{};
    std::array<Index, Capacity> older{};
    std::array<Index, Capacity> newer{};
    std::array<std::uint8_t, Capacity> active{};
    Index newest = nil;
    Index oldest = nil;
    Index count = 0;

    STACKDSL_HOT void setup() noexcept {
        value.fill(0.0);
        older.fill(nil);
        newer.fill(nil);
        active.fill(0);
        newest = oldest = nil;
        count = 0;
    }

    STACKDSL_HOT void erase(Index node) noexcept {
        if (!active[node]) return;
        const Index next_newer = newer[node];
        const Index next_older = older[node];
        if (next_newer == nil) newest = next_older;
        else older[next_newer] = next_older;
        if (next_older == nil) oldest = next_newer;
        else newer[next_older] = next_newer;
        active[node] = 0;
        newer[node] = older[node] = nil;
        --count;
    }

    STACKDSL_HOT void insert_newest(Index node, double incoming) noexcept {
        value[node] = incoming;
        active[node] = 1;
        newer[node] = nil;
        older[node] = newest;
        if (newest != nil) newer[newest] = node;
        else oldest = node;
        newest = node;
        ++count;
    }

    STACKDSL_HOT double kth_newest(std::size_t rank) const noexcept {
        Index node = newest;
        while (node != nil && rank != 0) {
            node = older[node];
            --rank;
        }
        return node == nil ? kNaN : value[node];
    }
};

}  // namespace stackdsl
