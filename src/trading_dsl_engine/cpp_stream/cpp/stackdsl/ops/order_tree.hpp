#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

#include "stackdsl/utils.hpp"

namespace stackdsl {

// Fixed-capacity order-statistic treap.  Ring positions are reused as node IDs,
// so insertion/erasure never allocates and every sliding update is expected
// O(log Capacity).  Sequence numbers make equal floating values distinct while
// rank queries retain upper-tie semantics.
template <std::size_t StateSize, std::size_t Capacity>
struct FixedOrderStatisticTree {
    static_assert(StateSize > 0 && Capacity > 0);

    struct Node {
        double value = 0.0;
        std::uint64_t sequence = 0;
        std::uint64_t priority = 0;
        std::int32_t left = -1;
        std::int32_t right = -1;
        std::uint32_t size = 1;
        std::uint8_t active = 0;
    };

    alignas(64) std::array<Node, StateSize * Capacity> nodes{};
    alignas(64) std::array<std::int32_t, StateSize> roots{};
    alignas(64) std::array<std::uint32_t, StateSize> counts{};

    void setup() noexcept {
        roots.fill(-1);
        counts.fill(0);
        for (auto& node : nodes) node = Node{};
    }

    STACKDSL_HOT std::size_t size(std::size_t state) const noexcept {
        return counts[state];
    }

    STACKDSL_HOT void replace(
        std::size_t state,
        std::size_t slot,
        double value,
        std::uint64_t sequence
    ) noexcept {
        Node& target = node(state, static_cast<std::int32_t>(slot));
        if (target.active) {
            roots[state] = erase(
                state, roots[state], target.value, target.sequence
            );
            --counts[state];
        }
        target = Node{};
        if (!finite(value)) return;
        target.value = value;
        target.sequence = sequence;
        target.priority = mix(
            sequence
            ^ (static_cast<std::uint64_t>(state) << 32U)
            ^ static_cast<std::uint64_t>(slot)
        );
        target.active = 1;
        roots[state] = insert(
            state, roots[state], static_cast<std::int32_t>(slot)
        );
        ++counts[state];
    }

    STACKDSL_HOT double kth(
        std::size_t state, std::size_t rank
    ) const noexcept {
        std::int32_t current = roots[state];
        while (current >= 0) {
            const Node& item = node(state, current);
            const std::size_t left_size = subtree_size(state, item.left);
            if (rank < left_size) current = item.left;
            else if (rank == left_size) return item.value;
            else {
                rank -= left_size + 1;
                current = item.right;
            }
        }
        return kNaN;
    }

    STACKDSL_HOT std::size_t count_less(
        std::size_t state, double value
    ) const noexcept {
        std::size_t result = 0;
        std::int32_t current = roots[state];
        while (current >= 0) {
            const Node& item = node(state, current);
            if (item.value < value) {
                result += subtree_size(state, item.left) + 1;
                current = item.right;
            } else {
                current = item.left;
            }
        }
        return result;
    }

    STACKDSL_HOT std::size_t count_less_equal(
        std::size_t state, double value
    ) const noexcept {
        std::size_t result = 0;
        std::int32_t current = roots[state];
        while (current >= 0) {
            const Node& item = node(state, current);
            if (item.value <= value) {
                result += subtree_size(state, item.left) + 1;
                current = item.right;
            } else {
                current = item.left;
            }
        }
        return result;
    }

private:
    static constexpr std::uint64_t mix(std::uint64_t value) noexcept {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    STACKDSL_HOT Node& node(
        std::size_t state, std::int32_t index
    ) noexcept {
        return nodes[state * Capacity + static_cast<std::size_t>(index)];
    }

    STACKDSL_HOT const Node& node(
        std::size_t state, std::int32_t index
    ) const noexcept {
        return nodes[state * Capacity + static_cast<std::size_t>(index)];
    }

    STACKDSL_HOT std::uint32_t subtree_size(
        std::size_t state, std::int32_t index
    ) const noexcept {
        return index < 0 ? 0U : node(state, index).size;
    }

    STACKDSL_HOT void pull(
        std::size_t state, std::int32_t index
    ) noexcept {
        Node& item = node(state, index);
        item.size = 1U
            + subtree_size(state, item.left)
            + subtree_size(state, item.right);
    }

    STACKDSL_HOT bool key_less(
        double value,
        std::uint64_t sequence,
        const Node& right
    ) const noexcept {
        return value < right.value
            || (value == right.value && sequence < right.sequence);
    }

    inline std::int32_t rotate_left(
        std::size_t state, std::int32_t root
    ) noexcept {
        Node& item = node(state, root);
        const std::int32_t pivot = item.right;
        Node& next = node(state, pivot);
        item.right = next.left;
        next.left = root;
        pull(state, root);
        pull(state, pivot);
        return pivot;
    }

    inline std::int32_t rotate_right(
        std::size_t state, std::int32_t root
    ) noexcept {
        Node& item = node(state, root);
        const std::int32_t pivot = item.left;
        Node& next = node(state, pivot);
        item.left = next.right;
        next.right = root;
        pull(state, root);
        pull(state, pivot);
        return pivot;
    }

    inline std::int32_t insert(
        std::size_t state,
        std::int32_t root,
        std::int32_t inserted
    ) noexcept {
        if (root < 0) return inserted;
        const Node& incoming = node(state, inserted);
        if (key_less(incoming.value, incoming.sequence, node(state, root))) {
            node(state, root).left = insert(
                state, node(state, root).left, inserted
            );
            if (
                node(state, node(state, root).left).priority
                > node(state, root).priority
            ) {
                root = rotate_right(state, root);
            }
        } else {
            node(state, root).right = insert(
                state, node(state, root).right, inserted
            );
            if (
                node(state, node(state, root).right).priority
                > node(state, root).priority
            ) {
                root = rotate_left(state, root);
            }
        }
        pull(state, root);
        return root;
    }

    inline std::int32_t merge(
        std::size_t state,
        std::int32_t left,
        std::int32_t right
    ) noexcept {
        if (left < 0) return right;
        if (right < 0) return left;
        if (node(state, left).priority > node(state, right).priority) {
            node(state, left).right = merge(
                state, node(state, left).right, right
            );
            pull(state, left);
            return left;
        }
        node(state, right).left = merge(
            state, left, node(state, right).left
        );
        pull(state, right);
        return right;
    }

    inline std::int32_t erase(
        std::size_t state,
        std::int32_t root,
        double value,
        std::uint64_t sequence
    ) noexcept {
        if (root < 0) return -1;
        Node& item = node(state, root);
        if (item.value == value && item.sequence == sequence) {
            return merge(state, item.left, item.right);
        }
        if (key_less(value, sequence, item)) {
            item.left = erase(state, item.left, value, sequence);
        } else {
            item.right = erase(state, item.right, value, sequence);
        }
        pull(state, root);
        return root;
    }
};

}  // namespace stackdsl
