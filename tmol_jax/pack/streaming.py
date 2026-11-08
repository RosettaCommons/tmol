"""JAX buffer execution for the common three-pass interaction-graph driver."""

from functools import partial, lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.streaming import (
    build_streaming_interaction_graph,
    complete_table_fits_window,
    PACK_ROTAMER_BLOCK_PAIR_WINDOW,
    PACK_FUSED_ROTAMER_SCORE_WINDOW,
)
from .graph import topology_stage, prefix_offsets_and_total, _wide
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


def _explicit_intermediate_bytes(program):
    """Overcount explicit JAX buffers, including nested native FFI workspaces.

    Equation results are counted even when they may alias inputs or are dead.
    stop_gradient is an identity at execution and needs no additional buffer.
    Input chemistry/coordinate arrays are already resident and are excluded.
    Unknown control-flow multiplicity cannot establish a complete-table bound.
    """
    if hasattr(program, "jaxpr"):
        program = program.jaxpr
    total = sum(
        int(np.prod(value.aval.shape)) * np.dtype(value.aval.dtype).itemsize
        for value in program.constvars
        if hasattr(value.aval, "shape") and hasattr(value.aval, "dtype")
    )
    for equation in program.eqns:
        if equation.primitive.name == "stop_gradient":
            continue
        if equation.primitive.name in ("while", "scan", "pure_callback", "io_callback"):
            return None
        for value in equation.outvars:
            aval = value.aval
            if hasattr(aval, "shape") and hasattr(aval, "dtype"):
                total += int(np.prod(aval.shape)) * np.dtype(aval.dtype).itemsize
        pending = list(equation.params.values())
        while pending:
            value = pending.pop()
            if hasattr(value, "jaxpr") or hasattr(value, "eqns"):
                size = _explicit_intermediate_bytes(value)
                if size is None:
                    return None
                total += size
            elif isinstance(value, (tuple, list)):
                pending.extend(value)
            elif isinstance(value, dict):
                pending.extend(value.values())
    return total


@lru_cache(maxsize=128)
def _compiled_buffer_bound(evaluate, coordinate_shape, arrays, candidates, x64):
    # Keys retain only compiled-call identity and shapes, never device chemistry
    # buffers. A new layout, dtype or callback gets a fresh conservative trace.
    program = jax.make_jaxpr(evaluate)(coordinate_shape, arrays, candidates)
    return program.out_avals[0].shape[-1], _explicit_intermediate_bytes(program)


def complete_table_bound(scorer, coords):
    """Conservatively bound the existing fused evaluator without running physics."""
    counts = np.asarray(scorer.rotamers.n_rots_for_pose, dtype=np.int64)
    pair_bound = sum(int(n) * (int(n) + 1) // 2 for n in counts)
    poses, blocks = scorer.rotamers.n_rots_for_block.shape
    block_pairs = poses * blocks * (blocks + 1) // 2
    reaches = {entry[4][3] for entry in scorer.entries if entry[5]}
    candidates = tuple(
        jax.ShapeDtypeStruct((3, pair_bound), np.int32) if entry[5] else None
        for entry in scorer.entries
    )
    # Abort large cases before tracing: even candidate identities alone exceed
    # the common page budget. Python integers keep this bound overflow-free.
    if not complete_table_fits_window(
        block_pairs,
        pair_bound * len(reaches),
        pair_bound * len(reaches) * 12,
        coords.dtype.itemsize,
    ):
        return None
    arrays = jax.tree.map(
        lambda a: jax.ShapeDtypeStruct(a.shape, a.dtype), scorer._arrays
    )
    coordinate_shape = jax.ShapeDtypeStruct(coords.shape, coords.dtype)
    score_count, temporary = _compiled_buffer_bound(
        scorer._evaluate, coordinate_shape, arrays, candidates, jax.config.x64_enabled
    )
    if temporary is None:
        return None
    # Sphere outputs, each reach's count/prefix/layout temporaries, and all
    # candidate buffers can coexist before the fused score call. The layout
    # allowance overcounts its int32 counts/offsets and wide prefix temporaries.
    temporary += 8 * coords.dtype.itemsize * (int(counts.sum()) + poses * blocks)
    temporary += len(reaches) * (64 * (block_pairs + 1) + 12 * pair_bound)
    return block_pairs, score_count, temporary


def use_complete_table(scorer, coords, **windows):
    bound = complete_table_bound(scorer, coords)
    return bound is not None and complete_table_fits_window(
        *bound, coords.dtype.itemsize, **windows
    )
