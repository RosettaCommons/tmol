"""Caller-owned arrays for the common streaming interaction graph."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register
from .graph import retained_topology, topology_stage, prefix_offsets_and_total


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
    p, b = rotamers.n_rots_for_block.shape
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

    def wide(*size):
        return shape((*size, 8), np.uint8)

    def integer(*size):
        return shape(size, np.int32)

    def prefix(counts, *, include_total=False):
        offsets, total = prefix_offsets_and_total(counts, include_total=include_total)
        total = int(np.asarray(total).copy().view(np.int64)[0])
        return offsets, total

    def topology(op, args, outputs):
        return topology_stage(
            args, operation=op, outputs=outputs, chunk_size=chunk_size
        )

    def entries(op, args, outputs, extent=0):
        return entry_stage(
            args, operation=op, outputs=outputs, extent=extent, chunk_size=chunk_size
        )

    with jax.default_device(rotamers.device):
        retained = retained_topology(
            rotamers.n_rots_for_block,
            rotamers.rot_offset_for_block,
            rotamers.rot_offset_for_pose,
            rotamers.pose_for_rot,
            rotamers.block_ind_for_rot,
        )
        counts = retained["block_rotamer_counts"]
        m = counts.shape[1]
        mapping = topology(0, (rotamers.n_rots_for_block,), wide(p, b))
        common = (indices, rotamers.block_ind_for_rot, mapping)
        adjacency = entries(0, (*common, counts), integer(p, m, (m + 31) // 32))
        neighbor_counts = topology(1, (adjacency,), wide(p * m + 1))
        rows, n_edges = prefix(neighbor_counts[:-1], include_total=True)
        neighbors, pair_counts = topology(
            2, (counts, adjacency, rows), (integer(n_edges), wide(n_edges))
        )
        words = topology(3, (pair_counts,), wide(n_edges))
        word_offsets, n_words = prefix(words)
        pair_offsets, n_pairs = prefix(pair_counts)
        chunk_adjacency = entries(
            1,
            (
                *common,
                rotamers.rot_offset_for_block,
                counts,
                rows,
                neighbors,
                word_offsets,
            ),
            integer(n_words),
        )
        pair_sizes = topology(
            5,
            (
                counts,
                rows,
                neighbors,
                pair_offsets,
                word_offsets,
                chunk_adjacency,
            ),
            wide(n_pairs),
        )
        chunk_offsets, n_energies = prefix(pair_sizes)
        chunk_offsets = topology(6, (pair_sizes, chunk_offsets), wide(n_pairs))
        n_rotamers = retained["inverse_rotamers"].shape[0]
        background, one_body, two_body = entries(
            2,
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
            (
                shape((p,), values.dtype),
                shape((n_rotamers,), values.dtype),
                shape((n_energies,), values.dtype),
            ),
            extent=n_rotamers,
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
