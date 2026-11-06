"""Execution of common sparse interaction-graph preparation stages."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register, workspace_sizes


@partial(jax.jit, static_argnames=("operation", "outputs", "chunk_size"))
def topology_stage(arguments, *, operation, outputs, chunk_size=16):
    """Run a fixed-shape native graph stage on CPU/CUDA.

    Integer CSR metadata uses uint8 rows of eight bytes; this transports int64
    exactly with either JAX precision setting. Dynamic graph allocation belongs
    to the caller. Sentinel updates return a new array, preserving input offsets.
    """
    target = "tmol_graph_topology"
    register(target)
    return jax.ffi.ffi_call(
        target,
        outputs,
        input_output_aliases={1: 0} if operation == 6 else None,
        vmap_method="sequential",
    )(*arguments, operation=np.int64(operation), chunk_size=np.int64(chunk_size))


@jax.jit
def prefix_counts(counts):
    """Return exact int64 exclusive offsets and total as opaque device bytes.

    Only the final eight-byte row needs a host read for a dynamic allocation.
    The common native scan retains all prefix work on the input device.
    """
    target = "tmol_graph_prefix"
    register(target)
    size = workspace_sizes(target, (counts.shape[0],), 64)[0]
    shape = jax.ShapeDtypeStruct
    return jax.ffi.ffi_call(
        target,
        (shape((counts.shape[0] + 1, 8), np.uint8), shape((size,), np.uint8)),
        vmap_method="sequential",
    )(counts, jnp.zeros(1, jnp.int32))[0]
