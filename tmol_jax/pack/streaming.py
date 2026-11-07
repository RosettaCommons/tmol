"""JAX buffer execution for the common three-pass interaction-graph driver."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.streaming import (
    build_streaming_interaction_graph,
    PACK_ROTAMER_BLOCK_PAIR_WINDOW,
    PACK_FUSED_ROTAMER_SCORE_WINDOW,
)
from .graph import topology_stage, prefix_offsets_and_total, _wide, _integer
from .interaction_graph import (
    build_interaction_graph,
    additive_entry_stage,
    _graph_edge_layout,
)

# These arrays belong exclusively to the common driver. Consume their old
# storage rather than copying the complete graph for each bounded score page.
_consume_entries = partial(
    jax.jit,
    donate_argnums=(1,),
    static_argnames=("operation", "extent", "chunk_size"),
)(additive_entry_stage)


def _total(value):
    return int(np.asarray(value).copy().view(np.int64).reshape(-1)[0])


class _GraphExecution:
    def __init__(self, rotamers):
        self.rotamers = rotamers

    def _build_interaction_graph(self, bump, chunk_size, *arguments):
        indices, values = arguments[-4:-2]
        self.graph = build_interaction_graph(
            self.rotamers, values, indices, chunk_size=chunk_size, bump_check=bump
        )
        r = self.graph["retained"]
        return (
            jnp.max(r["rotamer_pose_counts"], initial=0),
            *(
                r[name]
                for name in (
                    "block_pose_counts",
                    "rotamer_pose_counts",
                    "rotamer_pose_offsets",
                    "block_rotamer_counts",
                    "block_rotamer_offsets",
                    "block_for_rotamer",
                    "fixed",
                    "inverse_rotamers",
                )
            ),
            *(
                self.graph[name]
                for name in (
                    "background",
                    "energy1b",
                    "neighbor_row_offsets",
                    "neighbor_blocks",
                    "neighbor_chunk_offset_offsets",
                    "chunk_offsets",
                    "energy2b",
                )
            ),
        )

    def _initialize_interaction_graph_topology(self, original, counts, empty):
        self.counts = counts
        poses, blocks = counts.shape
        mapping = topology_stage(
            (original,), operation=0, outputs=_wide(*original.shape)
        )
        return jnp.zeros((poses, blocks, (blocks + 31) // 32), jnp.int32), mapping

    def _note_interaction_graph_topology(
        self, blocks, mapping, adjacency, indices, values
    ):
        return _consume_entries(
            (indices, blocks, mapping, self.counts), (adjacency,), operation=0
        )

    def _finalize_interaction_graph_topology(self, width, counts, adjacency, empty):
        neighbors = topology_stage(
            (adjacency,), operation=1, outputs=_wide(counts.size + 1), chunk_size=width
        )
        rows, total = prefix_offsets_and_total(neighbors[:-1], include_total=True)
        neighbors, word_offsets, pair_offsets, sizes = _graph_edge_layout(
            counts, adjacency, rows, n_edges=_total(total), chunk_size=width
        )
        words, self.pairs = map(
            int, np.asarray(sizes).copy().view(np.int64).reshape(-1)
        )
        return rows, neighbors, pair_offsets, word_offsets, jnp.zeros(words, jnp.int32)

    def _note_interaction_graph_chunk_topology(
        self,
        width,
        original,
        offsets,
        blocks,
        mapping,
        counts,
        rows,
        neighbors,
        word_offsets,
        adjacency,
        indices,
        values,
    ):
        return _consume_entries(
            (indices, blocks, mapping, offsets, counts, rows, neighbors, word_offsets),
            (adjacency,),
            operation=1,
            chunk_size=width,
        )

    def _finalize_interaction_graph_chunk_topology(
        self,
        width,
        counts,
        rows,
        neighbors,
        pair_offsets,
        word_offsets,
        adjacency,
        empty,
    ):
        sizes = topology_stage(
            (counts, rows, neighbors, pair_offsets, word_offsets, adjacency),
            operation=5,
            outputs=_wide(self.pairs),
            chunk_size=width,
        )
        offsets, total = prefix_offsets_and_total(sizes)
        offsets = topology_stage(
            (sizes, offsets),
            operation=6,
            outputs=_wide(self.pairs),
            chunk_size=width,
        )
        return offsets, jnp.zeros(_total(total), empty.dtype)

    def _accumulate_interaction_graph_entries(
        self,
        width,
        original,
        offsets,
        blocks,
        mapping,
        fixed,
        counts,
        compact_offsets,
        rows,
        neighbors,
        pair_offsets,
        chunk_offsets,
        background,
        one_body,
        two_body,
        indices,
        values,
    ):
        return _consume_entries(
            (
                indices,
                values,
                blocks,
                mapping,
                offsets,
                fixed,
                counts,
                compact_offsets,
                rows,
                neighbors,
                pair_offsets,
                chunk_offsets,
            ),
            (background, one_body, two_body),
            operation=2,
            extent=one_body.size,
            chunk_size=width,
        )


def streaming_interaction_graph(
    scorer,
    coords,
    *,
    chunk_size,
    block_pair_window=PACK_ROTAMER_BLOCK_PAIR_WINDOW,
    score_window=PACK_FUSED_ROTAMER_SCORE_WINDOW,
):
    """Prepare a graph by replaying bounded pages through common native physics."""
    scorer._validate_coordinates(coords)
    if not scorer.sum_terms:
        raise ValueError("packing graphs require summed scalar entries")
    rotamers = scorer.rotamers
    if rotamers.correlated_groups:
        raise ValueError("correlated packing requires the complete score table")
    execution = _GraphExecution(rotamers)

    def entries(*, topology_only):
        for values, indices in scorer.iter_packing_entries(
            coords,
            block_pair_window=block_pair_window,
            score_window=score_window,
            topology_only=topology_only,
        ):
            yield None, indices, values

    graph_inputs = (
        scorer.packed_block_types.n_types,
        rotamers.n_rots_for_pose,
        rotamers.rot_offset_for_pose,
        rotamers.n_rots_for_block,
        rotamers.rot_offset_for_block,
        rotamers.pose_for_rot,
        rotamers.block_type_ind_for_rot,
        rotamers.block_ind_for_rot,
    )
    with jax.default_device(coords.device):
        result = build_streaming_interaction_graph(
            entries,
            graph_inputs,
            chunk_size,
            jnp.empty((3, 0), jnp.int32),
            jnp.empty((0,), coords.dtype),
            execution,
        )
    execution.graph.update(
        zip(
            (
                "background",
                "energy1b",
                "neighbor_row_offsets",
                "neighbor_blocks",
                "neighbor_chunk_offset_offsets",
                "chunk_offsets",
                "energy2b",
            ),
            result[9:],
        )
    )
    return execution.graph
