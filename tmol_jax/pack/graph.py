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


@partial(jax.jit, static_argnames=("maximum",))
def prefix_counts(counts, *, maximum=False):
    """Return exact int64 exclusive offsets and total as opaque device bytes.

    With ``maximum=True``, compute exclusive running maxima of nonnegative
    counts instead. Only the final eight-byte row needs a host read for a dynamic allocation.
    The common native scan retains all prefix work on the input device.
    """
    target = "tmol_graph_maximum_prefix" if maximum else "tmol_graph_prefix"
    register(target)
    size = workspace_sizes("tmol_graph_prefix", (counts.shape[0],), 64)[0]
    shape = jax.ShapeDtypeStruct
    return jax.ffi.ffi_call(
        target,
        (shape((counts.shape[0] + 1, 8), np.uint8), shape((size,), np.uint8)),
        vmap_method="sequential",
    )(counts, jnp.zeros(1, jnp.int32))[0]


@partial(jax.jit, static_argnames=("include_total",))
def prefix_offsets_and_total(counts, *, include_total=False):
    """Keep prefix slicing inside compiled execution around an eager barrier."""
    offsets = prefix_counts(counts)
    return offsets if include_total else offsets[:-1], offsets[-1]


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


def _wide(*dimensions):
    return jax.ShapeDtypeStruct((*dimensions, 8), np.uint8)


def _integer(*dimensions):
    return jax.ShapeDtypeStruct(dimensions, np.int32)


@jax.jit
def _retention_prefixes(counts, block_offsets, keep):
    poses, blocks = counts.shape
    keep, keep_blocks, fixed = retention_stage(
        (counts, block_offsets, keep),
        operation=1,
        outputs=(_wide(keep.shape[0]), _wide(poses, blocks), _wide(poses, blocks)),
    )
    rp, nr = prefix_offsets_and_total(keep)
    bp, nb = prefix_offsets_and_total(keep_blocks.reshape(-1, 8))
    return keep, keep_blocks, fixed, rp, bp, jnp.stack((nr, nb))


@partial(jax.jit, static_argnames=("total_rotamers", "total_blocks"))
def _retention_layout(
    keep, keep_blocks, rp, bp, pose_offsets, *, total_rotamers, total_blocks
):
    poses = keep_blocks.shape[0]

    def run(op, args, output, **attributes):
        return retention_stage(args, operation=op, outputs=output, **attributes)

    inverse_rotamers = run(2, (keep, rp), _wide(total_rotamers))
    inverse_blocks = run(3, (keep_blocks, bp), _wide(total_blocks))
    bo = run(4, (keep_blocks, bp), _integer(poses))
    bc = run(5, (bo,), _integer(poses), extent=total_blocks)
    ro = run(6, (pose_offsets, rp), _integer(poses), extent=total_rotamers)
    rc = run(7, (ro,), _integer(poses), extent=total_rotamers)
    return dict(
        inverse_rotamers=inverse_rotamers,
        inverse_blocks=inverse_blocks,
        block_pose_offsets=bo,
        block_pose_counts=bc,
        rotamer_pose_offsets=ro,
        rotamer_pose_counts=rc,
    ), jnp.max(bc, initial=0)


@partial(jax.jit, static_argnames=("width",))
def _retention_compact(
    layout, block_offsets, rp, bp, block_for_rotamer, pose_for_rotamer, *, width
):
    poses, blocks = block_offsets.shape
    total = layout["inverse_rotamers"].shape[0]

    def run(op, args, output, **attributes):
        return retention_stage(args, operation=op, outputs=output, **attributes)

    offsets = run(
        8,
        (layout["inverse_blocks"], layout["block_pose_offsets"], block_offsets, rp),
        _integer(poses, width),
        width=width,
    )
    counts = run(
        9,
        (
            layout["inverse_blocks"],
            layout["block_pose_offsets"],
            layout["block_pose_counts"],
            offsets,
            layout["rotamer_pose_offsets"],
        ),
        _integer(poses, width),
        width=blocks,
        extent=total,
    )
    compact_blocks = run(
        10,
        (
            layout["inverse_rotamers"],
            block_for_rotamer,
            pose_for_rotamer,
            bp,
            layout["block_pose_offsets"],
        ),
        _integer(total),
        width=blocks,
    )
    return dict(
        block_rotamer_offsets=offsets,
        block_rotamer_counts=counts,
        block_for_rotamer=compact_blocks,
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
    """Run common retained-conformer kernels between two allocation barriers.

    Only exact totals and the compact block width cross to the host. Bump
    energies, when provided, come from the common packing scorer.
    """
    if isinstance(counts, jax.core.Tracer):
        raise TypeError("dynamic retained topology must be prepared outside jit/vmap")
    if (best_energy_for_rotamer is None) != (best_energy_for_block is None):
        raise ValueError("bump retention requires both energy arrays")
    with jax.default_device(counts.device):
        if best_energy_for_rotamer is None:
            keep = jax.device_put(
                np.ones(pose_for_rotamer.size, np.int64).view(np.uint8).reshape(-1, 8),
                counts.device,
            )
        else:
            keep = retention_stage(
                (
                    pose_for_rotamer,
                    block_for_rotamer,
                    best_energy_for_rotamer,
                    best_energy_for_block,
                ),
                operation=0,
                outputs=_wide(pose_for_rotamer.size),
            )
        keep, keep_blocks, fixed, rp, bp, totals = _retention_prefixes(
            counts, block_offsets, keep
        )
        nr, nb = np.asarray(totals).copy().view(np.int64).reshape(-1)
        if max(nr, nb) > np.iinfo(np.int32).max:
            raise OverflowError("retained topology exceeds native int32 capacity")
        layout, maximum = _retention_layout(
            keep,
            keep_blocks,
            rp,
            bp,
            pose_offsets,
            total_rotamers=int(nr),
            total_blocks=int(nb),
        )
        compact = _retention_compact(
            layout,
            block_offsets,
            rp,
            bp,
            block_for_rotamer,
            pose_for_rotamer,
            width=int(np.asarray(maximum)),
        )
    return dict(rotamer_prefix=rp, block_prefix=bp, fixed=fixed, **layout, **compact)


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
