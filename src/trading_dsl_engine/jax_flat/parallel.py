from __future__ import annotations

import os
from typing import Any, Callable

import jax
import jax.numpy as jnp


# Logical CPU devices share host memory. Bound the input footprint of a
# multi-device prefix so large formulas fall back to XLA's single-device scan
# instead of multiplying temporary buffers. This is one scheduler-wide memory
# budget, not an operator-specific tuning threshold.
_CPU_PREFIX_SHARD_MAX_BYTES = int(
    os.environ.get("TRADING_DSL_JAX_FLAT_CPU_PREFIX_SHARD_MAX_BYTES", str(128 * 1024 * 1024))
)


def associative_prefix(
    combine: Callable[[Any, Any], Any],
    operands: Any,
    identity: Any,
) -> Any:
    """Prefix an associative PyTree, sharding time across logical CPUs when safe."""
    leaves = jax.tree.leaves(operands)
    input_bytes = sum(leaf.size * leaf.dtype.itemsize for leaf in leaves)
    devices = tuple(jax.local_devices(backend="cpu")) if jax.default_backend() == "cpu" else ()
    if len(devices) <= 1 or leaves[0].shape[0] < len(devices) or input_bytes > _CPU_PREFIX_SHARD_MAX_BYTES:
        return jax.lax.associative_scan(combine, operands, axis=0)

    n_steps = leaves[0].shape[0]
    n_devices = len(devices)
    padding = (-n_steps) % n_devices

    def time_blocks(value, identity_value):
        value = jnp.asarray(value)
        identity_value = jnp.broadcast_to(identity_value, value.shape[1:])
        padded = jnp.concatenate(
            (value, jnp.broadcast_to(identity_value, (padding,) + value.shape[1:])),
            axis=0,
        )
        return padded.reshape((n_devices, -1) + value.shape[1:])

    blocks = jax.tree.map(time_blocks, operands, identity)

    def scan_block(block_operands):
        local = jax.lax.associative_scan(combine, block_operands, axis=0)
        totals = jax.tree.map(lambda value: jax.lax.all_gather(value[-1], "device"), local)
        total_prefix = jax.lax.associative_scan(combine, totals, axis=0)
        device_index = jax.lax.axis_index("device")
        previous = jax.tree.map(lambda value: value[jnp.maximum(device_index - 1, 0)], total_prefix)
        with_previous = combine(previous, local)
        return jax.tree.map(
            lambda local_value, prefixed_value: jnp.where(device_index == 0, local_value, prefixed_value),
            local,
            with_previous,
        )

    mapped = jax.pmap(scan_block, axis_name="device", devices=devices)(blocks)
    return jax.tree.map(
        lambda value: value.reshape((-1,) + value.shape[2:])[:n_steps],
        mapped,
    )


def _compose_affine(left, right):
    left_scale, left_bias = left
    right_scale, right_bias = right
    return right_scale * left_scale, right_scale * left_bias + right_bias


def affine_prefix(scale: jax.Array, bias: jax.Array, initial: jax.Array) -> jax.Array:
    """Evaluate y <- scale*y + bias for every prefix."""
    prefix_scale, prefix_bias = associative_prefix(
        _compose_affine,
        (scale, bias),
        (jnp.ones_like(scale[0]), jnp.zeros_like(bias[0])),
    )
    return prefix_scale * initial + prefix_bias


def _compose_shared_affine(left, right):
    left_scale, *left_biases = left
    right_scale, *right_biases = right
    return (
        right_scale * left_scale,
        *(right_scale * left_bias + right_bias for left_bias, right_bias in zip(left_biases, right_biases)),
    )


def shared_affine_prefix(
    scale: jax.Array,
    biases: tuple[jax.Array, ...],
    initials: tuple[jax.Array, ...],
) -> tuple[jax.Array, ...]:
    """Apply one affine scale to several independently biased state leaves."""
    prefixes = associative_prefix(
        _compose_shared_affine,
        (scale, *biases),
        (jnp.ones_like(scale[0]), *(jnp.zeros_like(bias[0]) for bias in biases)),
    )
    prefix_scale, *prefix_biases = prefixes
    return tuple(prefix_scale * initial + bias for initial, bias in zip(initials, prefix_biases))


def _compose_segment_product(left, right):
    left_reset, left_product = left
    right_reset, right_product = right
    return (
        left_reset | right_reset,
        jnp.where(right_reset, right_product, left_product * right_product),
    )


def segmented_product_prefix(factors: jax.Array, reset_before: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Multiply within segments whose first row is marked by reset_before."""
    return associative_prefix(
        _compose_segment_product,
        (reset_before, factors),
        (jnp.zeros_like(reset_before[0]), jnp.ones_like(factors[0])),
    )


def prefix_sum(values: jax.Array) -> jax.Array:
    return associative_prefix(jnp.add, values, jnp.zeros_like(values[0]))


def prefix_or(values: jax.Array) -> jax.Array:
    return associative_prefix(jnp.logical_or, values, jnp.zeros_like(values[0]))


def prefix_max(values: jax.Array) -> jax.Array:
    if jnp.issubdtype(values.dtype, jnp.integer):
        minimum = jnp.iinfo(values.dtype).min
    elif jnp.issubdtype(values.dtype, jnp.bool_):
        minimum = False
    else:
        minimum = -jnp.inf
    return associative_prefix(jnp.maximum, values, jnp.full_like(values[0], minimum))
