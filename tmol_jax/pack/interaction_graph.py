"""Caller-owned arrays for the common streaming interaction graph."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register
from .graph import (
    retained_topology,
    topology_stage,
    prefix_offsets_and_total,
    _wide,
    _integer,
)


@partial(jax.jit, static_argnames=("operation", "outputs", "extent", "chunk_size"))
def entry_stage(arguments, *, operation, outputs, extent=0, chunk_size=16):
    register("tmol_graph_entries")
    return jax.ffi.ffi_call("tmol_graph_entries", outputs, vmap_method="sequential")(
        *arguments,
        operation=np.int64(operation),
        extent=np.int64(extent),
        width=np.int64(chunk_size),
    )


@jax.jit
def _valid_entries(values, indices, pose_for_rotamer, pose_counts):
    valid = jnp.all(jnp.isfinite(values)) & jnp.all(indices >= 0)
    valid &= jnp.all(indices[0] < pose_counts.shape[0])
    valid &= jnp.all(indices[1:] < pose_for_rotamer.size)
    same_pose = jnp.all(pose_for_rotamer[indices[1:]] == indices[0])
    return jnp.stack((valid, same_pose))


@partial(jax.jit, static_argnames=("chunk_size",))
def _graph_adjacency(
    original_counts, counts, indices, block_for_rotamer, *, chunk_size
):
    p, b = original_counts.shape
    m = counts.shape[1]
    mapping = topology_stage(
        (original_counts,), operation=0, outputs=_wide(p, b), chunk_size=chunk_size
    )
    adjacency = entry_stage(
        (indices, block_for_rotamer, mapping, counts),
        operation=0,
        outputs=_integer(p, m, (m + 31) // 32),
        chunk_size=chunk_size,
    )
    neighbors = topology_stage(
        (adjacency,), operation=1, outputs=_wide(p * m + 1), chunk_size=chunk_size
    )
    rows, total = prefix_offsets_and_total(neighbors[:-1], include_total=True)
    return mapping, adjacency, rows, total


@partial(jax.jit, static_argnames=("n_edges", "chunk_size"))
def _graph_edge_layout(counts, adjacency, rows, *, n_edges, chunk_size):
    neighbors, pairs = topology_stage(
        (counts, adjacency, rows),
        operation=2,
        outputs=(_integer(n_edges), _wide(n_edges)),
        chunk_size=chunk_size,
    )
    words = topology_stage(
        (pairs,), operation=3, outputs=_wide(n_edges), chunk_size=chunk_size
    )
    wo, nw = prefix_offsets_and_total(words)
    po, npairs = prefix_offsets_and_total(pairs)
    return neighbors, wo, po, jnp.stack((nw, npairs))


@partial(jax.jit, static_argnames=("n_words", "n_pairs", "chunk_size"))
def _graph_chunk_layout(
    common,
    block_offsets,
    counts,
    rows,
    neighbors,
    word_offsets,
    pair_offsets,
    *,
    n_words,
    n_pairs,
    chunk_size,
):
    adjacency = entry_stage(
        (*common, block_offsets, counts, rows, neighbors, word_offsets),
        operation=1,
        outputs=_integer(n_words),
        chunk_size=chunk_size,
    )
    sizes = topology_stage(
        (counts, rows, neighbors, pair_offsets, word_offsets, adjacency),
        operation=5,
        outputs=_wide(n_pairs),
        chunk_size=chunk_size,
    )
    offsets, total = prefix_offsets_and_total(sizes)
    offsets = topology_stage(
        (sizes, offsets), operation=6, outputs=_wide(n_pairs), chunk_size=chunk_size
    )
    return offsets, total


def build_interaction_graph(rotamers, values, indices, *, chunk_size=16):
    """Build native background, one-body and chunked two-body energy tables.

    Inputs are uncoalesced scalar entries and [3, entries] pose/rotamer indices.
    Duplicate entries accumulate. All conformers are retained; bump pruning and
    correlated-group collapse are not implemented here. Discrete allocation is
    eager and energy-table construction currently has no AD rule.
    """
    if any(isinstance(a, jax.core.Tracer) for a in (values, indices)):
        raise TypeError("dynamic interaction graphs must be built outside jit/vmap/AD")
    if rotamers.correlated_groups:
        raise NotImplementedError(
            "interaction graph lockstep collapse is not implemented"
        )
    if not isinstance(chunk_size, (int, np.integer)) or not 0 < chunk_size < 1 << 31:
        raise ValueError("chunk_size must be a positive int32")
    if values.ndim != 1 or values.dtype not in (jnp.float32, jnp.float64):
        raise TypeError("interaction graph values must be a float32/64 vector")
    if indices.shape != (3, values.size) or indices.dtype != jnp.int32:
        raise TypeError("interaction graph indices must be int32 [3, entries]")
    if values.device != rotamers.device or indices.device != rotamers.device:
        raise ValueError("interaction graph inputs must share the conformer device")
    p = rotamers.n_rots_for_block.shape[0]
    if values.size:
        valid, same_pose = np.asarray(
            _valid_entries(
                values, indices, rotamers.pose_for_rot, rotamers.n_rots_for_block
            )
        )
        if not bool(valid):
            raise ValueError("invalid energy or sparse conformer index")
        if not same_pose:
            raise ValueError("sparse entries cannot join different poses")
    shape = jax.ShapeDtypeStruct

    def total(value):
        return int(np.asarray(value).copy().view(np.int64).reshape(-1)[0])

    with jax.default_device(rotamers.device):
        retained = retained_topology(
            rotamers.n_rots_for_block,
            rotamers.rot_offset_for_block,
            rotamers.rot_offset_for_pose,
            rotamers.pose_for_rot,
            rotamers.block_ind_for_rot,
        )
        counts = retained["block_rotamer_counts"]
        mapping, adjacency, rows, size = _graph_adjacency(
            rotamers.n_rots_for_block,
            counts,
            indices,
            rotamers.block_ind_for_rot,
            chunk_size=chunk_size,
        )
        neighbors, word_offsets, pair_offsets, sizes = _graph_edge_layout(
            counts,
            adjacency,
            rows,
            n_edges=total(size),
            chunk_size=chunk_size,
        )
        n_words, n_pairs = np.asarray(sizes).copy().view(np.int64).reshape(-1)
        chunk_offsets, size = _graph_chunk_layout(
            (indices, rotamers.block_ind_for_rot, mapping),
            rotamers.rot_offset_for_block,
            counts,
            rows,
            neighbors,
            word_offsets,
            pair_offsets,
            n_words=int(n_words),
            n_pairs=int(n_pairs),
            chunk_size=chunk_size,
        )
        n_energies = total(size)
        n_rotamers = retained["inverse_rotamers"].shape[0]
        background, one_body, two_body = entry_stage(
            (
                indices,
                values,
                rotamers.block_ind_for_rot,
                mapping,
                rotamers.rot_offset_for_block,
                retained["fixed"],
                counts,
                retained["block_rotamer_offsets"],
                rows,
                neighbors,
                pair_offsets,
                chunk_offsets,
            ),
            outputs=(
                shape((p,), values.dtype),
                shape((n_rotamers,), values.dtype),
                shape((n_energies,), values.dtype),
            ),
            operation=2,
            extent=n_rotamers,
            chunk_size=chunk_size,
        )
    return dict(
        retained=retained,
        chunk_size=chunk_size,
        background=background,
        energy1b=one_body,
        energy2b=two_body,
        neighbor_row_offsets=rows,
        neighbor_blocks=neighbors,
        neighbor_chunk_offset_offsets=pair_offsets,
        chunk_offsets=chunk_offsets,
    )
