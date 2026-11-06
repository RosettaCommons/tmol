"""Execution of common sparse interaction-graph preparation stages."""

import ctypes
from functools import lru_cache, partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import libraries, register, workspace_sizes


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
    )(
        *arguments,
        operation=np.int64(operation),
        extent=np.int64(0),
        width=np.int64(chunk_size),
    )


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


@partial(jax.jit, static_argnames=("operation", "outputs", "extent", "width"))
def retention_stage(arguments, *, operation, outputs, extent=0, width=0):
    """Run the common retained-conformer stage with functional output masks.

    ``extent`` is the total retained block/rotamer count; ``width`` is the
    original block width (stages 9/10) or compact block width (stage 8).
    """
    target = "tmol_graph_retention"
    register(target)
    return jax.ffi.ffi_call(
        target,
        outputs,
        input_output_aliases={2: 0} if operation == 1 else None,
        vmap_method="sequential",
    )(
        *arguments,
        operation=np.int64(operation),
        extent=np.int64(extent),
        width=np.int64(width),
    )


def retained_topology(
    counts,
    block_offsets,
    pose_offsets,
    pose_for_rotamer,
    block_for_rotamer,
    *,
    best_energy_for_rotamer=None,
    best_energy_for_block=None,
):
    """Allocate and execute common retained/fixed-conformer topology stages.

    This internal eager preparation retains device arrays throughout native
    compaction; only totals and compact pose counts cross the allocation barrier.
    Bump energies must already be prepared by the common packing scorer.
    """
    if isinstance(counts, jax.core.Tracer):
        raise TypeError("dynamic retained topology must be prepared outside jit/vmap")
    poses, blocks = counts.shape
    rotamers = pose_for_rotamer.size
    shape = jax.ShapeDtypeStruct

    def wide(*dimensions):
        return shape((*dimensions, 8), np.uint8)

    def integer(*dimensions):
        return shape(dimensions, np.int32)

    def run(op, args, outputs, **attributes):
        return retention_stage(args, operation=op, outputs=outputs, **attributes)

    def compact(mask):
        offsets = prefix_counts(mask.reshape(-1, 8))
        total = int(np.asarray(offsets[-1]).copy().view(np.int64)[0])
        if total > np.iinfo(np.int32).max:
            raise OverflowError("retained topology exceeds native int32 capacity")
        return offsets[:-1], total

    with jax.default_device(counts.device):
        if best_energy_for_rotamer is None:
            if best_energy_for_block is not None:
                raise ValueError("bump retention requires both energy arrays")
            ones = np.ones(rotamers, np.int64).view(np.uint8).reshape(-1, 8)
            keep = jax.device_put(ones, counts.device)
        else:
            if best_energy_for_block is None:
                raise ValueError("bump retention requires both energy arrays")
            keep = run(
                0,
                (
                    pose_for_rotamer,
                    block_for_rotamer,
                    best_energy_for_rotamer,
                    best_energy_for_block,
                ),
                wide(rotamers),
            )
        keep, keep_blocks, fixed = run(
            1,
            (counts, block_offsets, keep),
            (wide(rotamers), wide(poses, blocks), wide(poses, blocks)),
        )
        rotamer_prefix, total_rotamers = compact(keep)
        block_prefix, total_blocks = compact(keep_blocks)
        inverse_rotamers = run(2, (keep, rotamer_prefix), wide(total_rotamers))
        inverse_blocks = run(3, (keep_blocks, block_prefix), wide(total_blocks))
        block_pose_offsets = run(4, (keep_blocks, block_prefix), integer(poses))
        block_pose_counts = run(
            5, (block_pose_offsets,), integer(poses), extent=total_blocks
        )
        rotamer_pose_offsets = run(
            6, (pose_offsets, rotamer_prefix), integer(poses), extent=total_rotamers
        )
        rotamer_pose_counts = run(
            7, (rotamer_pose_offsets,), integer(poses), extent=total_rotamers
        )
        width = int(np.asarray(block_pose_counts).max(initial=0))
        compact_offsets = run(
            8,
            (inverse_blocks, block_pose_offsets, block_offsets, rotamer_prefix),
            integer(poses, width),
            width=width,
        )
        compact_counts = run(
            9,
            (
                inverse_blocks,
                block_pose_offsets,
                block_pose_counts,
                compact_offsets,
                rotamer_pose_offsets,
            ),
            integer(poses, width),
            width=blocks,
            extent=total_rotamers,
        )
        compact_blocks = run(
            10,
            (
                inverse_rotamers,
                block_for_rotamer,
                pose_for_rotamer,
                block_prefix,
                block_pose_offsets,
            ),
            integer(total_rotamers),
            width=blocks,
        )
    return dict(
        rotamer_prefix=rotamer_prefix,
        block_prefix=block_prefix,
        inverse_rotamers=inverse_rotamers,
        inverse_blocks=inverse_blocks,
        fixed=fixed,
        block_pose_offsets=block_pose_offsets,
        block_pose_counts=block_pose_counts,
        rotamer_pose_offsets=rotamer_pose_offsets,
        rotamer_pose_counts=rotamer_pose_counts,
        block_rotamer_offsets=compact_offsets,
        block_rotamer_counts=compact_counts,
        block_for_rotamer=compact_blocks,
    )


@lru_cache(maxsize=None)
def _load_balance_workspace(total, generators):
    library = libraries().get("cuda")
    if library is None:
        return 0
    try:
        devices = jax.local_devices(backend="cuda")
    except RuntimeError:
        return 0
    query = library.tmol_load_balance_workspace
    query.argtypes = [ctypes.c_int] * 3
    query.restype = ctypes.c_uint64
    # One shape is valid on every local device, including heterogeneous GPUs.
    size = max(
        (query(total, generators, d.local_hardware_id) for d in devices), default=0
    )
    if size == (1 << 64) - 1:
        raise ValueError("invalid native load-balance workspace")
    return size


@partial(jax.jit, static_argnames=("total",))
def load_balance(offsets, *, total):
    """Expand trusted exclusive offsets to generator IDs on the input device.

    Offsets are opaque int64 rows without a terminal total. They must start at
    zero, be nondecreasing and bounded by ``total``; common count/prefix stages
    produce this layout. Empty generators are retained. The output extent is
    static, allowing jit/vmap without a hidden allocation or host count read.
    """
    generators = offsets.shape[0]
    if offsets.shape != (generators, 8) or offsets.dtype != np.uint8:
        raise TypeError("load-balance offsets require opaque int64 rows")
    if total < 0 or total + generators > np.iinfo(np.int32).max:
        raise OverflowError("load-balance extent exceeds native int32 capacity")
    if total and not generators:
        raise ValueError("nonempty load balance requires generators")
    target = "tmol_load_balance"
    register(target)
    capacity = _load_balance_workspace(total, generators)
    shape = jax.ShapeDtypeStruct
    return jax.ffi.ffi_call(
        target,
        (shape((total,), np.int32), shape((capacity, 8), np.uint8)),
        vmap_method="sequential",
    )(offsets, operation=np.int64(0), extent=np.int64(total), width=np.int64(0))[0]
