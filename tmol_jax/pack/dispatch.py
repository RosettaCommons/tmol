"""Native sparse candidate dispatch for conformer scoring."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register
from .graph import prefix_counts


@partial(jax.jit, static_argnames=("operation", "outputs", "begin", "count", "reach"))
def _stage(arguments, *, operation, outputs, begin=0, count=0, reach=0.0):
    target = "tmol_rotamer_dispatch"
    register(target)
    return jax.ffi.ffi_call(target, outputs, vmap_method="sequential")(
        *arguments,
        operation=np.int64(operation),
        extent=np.int64(begin),
        width=np.int64(count),
        reach=np.float64(reach),
    )


def rotamer_spheres(coords, coord_offsets, types, n_atoms):
    """Compute the canonical atom-center bounding sphere of each conformer."""
    return _stage(
        (coords, coord_offsets, types, n_atoms),
        operation=0,
        outputs=jax.ShapeDtypeStruct((types.size, 4), coords.dtype),
    )


def block_spheres(spheres, counts, offsets):
    """Bound the ensemble of conformer spheres at each molecular site."""
    return _stage(
        (spheres, counts, offsets),
        operation=1,
        outputs=jax.ShapeDtypeStruct((*counts.shape, 4), spheres.dtype),
    )


@jax.jit
def _wide_nonnegative(counts):
    # Native dispatch counts are nonnegative int32, with overflow reported
    # separately. Preserve exact int64 capacity while JAX x64 is disabled.
    return jnp.stack((counts, jnp.zeros_like(counts)), axis=-1).view(jnp.uint8)


@jax.jit
def _narrow_prefix(offsets):
    # The caller checks the wide total before consuming these int32 offsets.
    return offsets[:, :4].view(jnp.int32).reshape(-1)


@partial(jax.jit, static_argnames=("begin", "count", "reach"))
def _candidate_layout(arguments, *, begin, count, reach):
    shape = jax.ShapeDtypeStruct
    counts, overflow = _stage(
        arguments,
        operation=2,
        begin=begin,
        count=count,
        reach=reach,
        outputs=(shape((count,), np.int32), shape((1, 8), np.uint8)),
    )
    offsets = prefix_counts(_wide_nonnegative(counts))
    # One allocation barrier observes both overflow and the exact wide total.
    # Overflowed native counts remain zero; no fill consumes truncated offsets.
    info = jnp.concatenate((overflow, offsets[-1:]), axis=0)
    return counts, _narrow_prefix(offsets[:-1]), info


def candidate_pairs(
    pose_types,
    spheres,
    site_spheres,
    counts,
    offsets,
    groups,
    *,
    reach,
    begin=0,
    count=None,
):
    """Enumerate native sparse pairs, retaining lockstep and paging policies.

    Pair indices are (pose, global conformer i, global conformer j). This is
    eager dynamic preparation: count totals cross a host allocation barrier;
    sphere evaluation and pair enumeration stay on the input device. Individual
    fixed-shape stages support jit/vmap, but candidate identities are discrete.
    """
    if isinstance(spheres, jax.core.Tracer):
        raise TypeError("dynamic candidate pairs must be prepared outside jit/vmap")
    poses, blocks = pose_types.shape
    candidates = poses * blocks * (blocks + 1) // 2
    count = candidates - begin if count is None else count
    if not 0 <= begin <= candidates or not 0 <= count <= candidates - begin:
        raise ValueError("invalid conformer candidate window")
    arguments = (site_spheres, pose_types, counts, offsets, spheres, groups)
    shape = jax.ShapeDtypeStruct
    pair_counts, pair_offsets, info = _candidate_layout(
        arguments, begin=begin, count=count, reach=reach
    )
    overflow, total = np.asarray(info).copy().view(np.int64).reshape(-1)
    if overflow:
        raise OverflowError("conformer pair count exceeds native int32 capacity")
    if total > np.iinfo(np.int32).max:
        raise OverflowError("conformer pair indices exceed native int32 capacity")
    return _stage(
        (*arguments, pair_counts, pair_offsets),
        operation=3,
        begin=begin,
        count=count,
        reach=reach,
        outputs=shape((3, total), np.int32),
    )
