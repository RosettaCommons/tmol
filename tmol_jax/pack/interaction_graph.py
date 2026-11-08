"""Caller-owned arrays for the common streaming interaction graph."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .graph import (
    retained_topology,
    topology_stage,
    prefix_offsets_and_total,
    _wide,
    _integer,
)

from ._ffi import stage


@partial(jax.jit, static_argnames=("operation", "outputs", "extent", "chunk_size"))
def entry_stage(arguments, *, operation, outputs, extent=0, chunk_size=16):
    target = "tmol_graph_entries"
    return stage(arguments, target, outputs, operation, extent, chunk_size)


@partial(jax.jit, static_argnames=("operation", "extent", "chunk_size"))
def additive_entry_stage(arguments, previous, *, operation, extent=0, chunk_size=16):
    """Accumulate one native page, retaining JAX input value semantics."""
    outputs = tuple(jax.ShapeDtypeStruct(a.shape, a.dtype) for a in previous)
    return stage(
        (*arguments, *previous),
        "tmol_graph_entries_additive",
        outputs,
        operation,
        extent,
        chunk_size,
        aliases={len(arguments) + i: i for i in range(len(previous))},
    )


@jax.jit
def _valid_entries(values, indices, pose_for_rotamer, block_for_rotamer, pose_counts):
    valid = jnp.all(jnp.isfinite(values)) & jnp.all(indices >= 0)
    valid &= jnp.all(indices[0] < pose_counts.shape[0])
    valid &= jnp.all(indices[1:] < pose_for_rotamer.size)
    same_pose = jnp.all(pose_for_rotamer[indices[1:]] == indices[0])
    compatible = jnp.all(
        (indices[1] == indices[2])
        | (block_for_rotamer[indices[1]] != block_for_rotamer[indices[2]])
    )
    return jnp.stack((valid, same_pose, compatible))


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


@jax.jit
def _canonical_indices(indices, block_of):
    swap = block_of[indices[1]] > block_of[indices[2]]
    return jnp.stack(
        (
            indices[0],
            jnp.where(swap, indices[2], indices[1]),
            jnp.where(swap, indices[1], indices[2]),
        )
    )


def build_interaction_graph(
    rotamers, values=None, indices=None, *, scorer=None, chunk_size=16, bump_check=False
):
    """Build native background, one-body and chunked two-body energy tables.

    Inputs are uncoalesced scalar entries and [3, entries] pose/rotamer indices.
    Alternatively, supply a scorer to produce those entries here, allowing
    obsolete entry arrays to be released as the graph is constructed.
    Duplicate entries accumulate. Distinct alternatives of one block cannot
    share an energy entry. Producer-declared joint conformers are collapsed
    through the shared correspondence plan. Optional bump pruning uses the common
    native bounds and preserves original retained identities.
    Discrete allocation is
    eager and energy-table construction currently has no AD rule.
    """
    if scorer is not None:
        if values is not None or indices is not None:
            raise ValueError("supply either sparse entries or a scorer")
        values, indices = scorer(rotamers.coords)
    elif values is None or indices is None:
        raise ValueError("supply both values and indices, or a scorer")
    if any(isinstance(a, jax.core.Tracer) for a in (values, indices)):
        raise TypeError("dynamic interaction graphs must be built outside jit/vmap/AD")
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
        valid, same_pose, compatible = np.asarray(
            _valid_entries(
                values,
                indices,
                rotamers.pose_for_rot,
                rotamers.block_ind_for_rot,
                rotamers.n_rots_for_block,
            )
        )
        if not bool(valid):
            raise ValueError("invalid energy or sparse conformer index")
        if not same_pose:
            raise ValueError("sparse entries cannot join different poses")
        if not compatible:
            raise ValueError(
                "sparse entries cannot join distinct alternatives of one block"
            )
    shape = jax.ShapeDtypeStruct
    collapse = None
    grouped = bool(rotamers.correlated_groups)
    if grouped:
        from ._correlation import collapse_for_graph

        rotamers, values, indices, collapse = collapse_for_graph(
            rotamers, values, indices
        )

    indices = _canonical_indices(indices, rotamers.block_ind_for_rot)

    def total(value):
        return int(np.asarray(value).copy().view(np.int64).reshape(-1)[0])

    with jax.default_device(rotamers.device):
        bounds = {}
        if bump_check:
            from .bump import bump_bounds

            best_rotamer, best_block = bump_bounds(
                rotamers, values, indices, chunk_size=chunk_size
            )
            bounds = dict(
                best_energy_for_rotamer=best_rotamer, best_energy_for_block=best_block
            )
        retained = retained_topology(
            rotamers.n_rots_for_block,
            rotamers.rot_offset_for_block,
            rotamers.rot_offset_for_pose,
            rotamers.pose_for_rot,
            rotamers.block_ind_for_rot,
            **bounds,
        )
        source_counts, source_offsets = (
            rotamers.n_rots_for_block,
            rotamers.rot_offset_for_block,
        )
        source_blocks, fixed = rotamers.block_ind_for_rot, retained["fixed"]
        if bump_check:
            from .bump import retained_entry_mapping, compact_entries
            from .imposition import _wide as wide_values

            keep, entry_keep, mapping, source_offsets, source_counts, new_fixed = (
                retained_entry_mapping(
                    retained["inverse_rotamers"],
                    fixed,
                    indices,
                    source_blocks,
                    source_offsets,
                    source_counts,
                )
            )
            sizes = np.asarray(jnp.stack((jnp.sum(keep), jnp.sum(entry_keep))))
            values, indices, source_blocks = compact_entries(
                values,
                indices,
                keep,
                entry_keep,
                mapping,
                source_blocks,
                rotamers=int(sizes[0]),
                entries=int(sizes[1]),
            )
            fixed = wide_values(new_fixed)
        counts = retained["block_rotamer_counts"]
        mapping, adjacency, rows, size = _graph_adjacency(
            source_counts,
            counts,
            indices,
            source_blocks,
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
            (indices, source_blocks, mapping),
            source_offsets,
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
                source_blocks,
                mapping,
                source_offsets,
                fixed,
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
    if grouped:
        retained["group_collapse"] = collapse
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
