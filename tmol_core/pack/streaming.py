"""Bounded packing policy and the shared native three-pass graph lifecycle.

Execution owns buffer allocation, device queries and native bindings. The driver
owns replay order and releases each page before requesting the next one. Graph
inputs and outputs follow the existing native interaction-graph ABI.
"""

PACK_ROTAMER_BLOCK_PAIR_WINDOW = 32 * 1024
PACK_FUSED_ROTAMER_SCORE_WINDOW = 4 * 1024 * 1024

_SMALL_PACKING_POSE_CHUNK = 25
_DEFAULT_PACKING_POSE_CHUNK = 10
_ESTIMATED_PACKING_BYTES_PER_BLOCK = 2 * 1024 * 1024
_PACKING_FREE_MEMORY_FRACTION = 0.25
_CPU_INTERACTION_GRAPH_CHUNK_SIZE = 16
_CUDA_INTERACTION_GRAPH_CHUNK_SIZE = 32


def interaction_graph_chunk_size(device_kind):
    """Return the mature benchmarked sparse-table chunk width."""
    return (
        _CUDA_INTERACTION_GRAPH_CHUNK_SIZE
        if device_kind == "cuda"
        else _CPU_INTERACTION_GRAPH_CHUNK_SIZE
    )


def packing_pose_chunk_size(max_n_blocks, free_memory_bytes=None, override=None) -> int:
    """Return a bounded, memory-aware native-packer batch size.

    Native score dispatches use signed 32-bit per-chunk indices. Keeping the
    batch bounded prevents a wide pose stack from overflowing those indices
    and also limits transient score/index storage. The size cutoff and memory
    estimate are deliberately conservative: packing scratch grows with both
    residue count and rotamer density. The environment override is primarily
    for deterministic tests and performance tuning.
    """
    configured = override
    if configured is not None:
        try:
            chunk_size = int(configured)
        except ValueError as error:
            raise ValueError(
                "TMOL_PACK_MAX_POSES_PER_CHUNK must be a positive integer; "
                f"got {configured!r}"
            ) from error
        if chunk_size <= 0:
            raise ValueError(
                "TMOL_PACK_MAX_POSES_PER_CHUNK must be a positive integer; "
                f"got {configured!r}"
            )
        return chunk_size

    chunk_size = (
        _SMALL_PACKING_POSE_CHUNK
        if max_n_blocks <= 256
        else _DEFAULT_PACKING_POSE_CHUNK
    )

    if free_memory_bytes is not None:
        memory_budget = int(free_memory_bytes * _PACKING_FREE_MEMORY_FRACTION)
        estimated_bytes_per_pose = (
            max(1, max_n_blocks) * _ESTIMATED_PACKING_BYTES_PER_BLOCK
        )
        memory_limited_chunk = max(1, memory_budget // estimated_bytes_per_pose)
        chunk_size = min(chunk_size, memory_limited_chunk)
    return chunk_size


PACKER_TASK_POSE_TENSORS = (
    "is_real_block",
    "per_block_orig_block_type",
    "per_block_n_considered_block_types",
    "per_block_considered_block_types",
    "per_block_considered_block_types_is_orig",
    "restrict_to_repacking_masks",
    "per_block_is_block_type_allowed",
    "per_block_conformer_sampler_allowed",
    "per_block_chi_expansion",
)


def should_stream_interaction_graph(device_kind, correlated_groups, bump_check):
    """Use the mature inference-only GPU path when pruning permits replay."""
    return device_kind == "cuda" and not correlated_groups and not bump_check


def packing_windows(count, window):
    """Yield bounded (start, count) ranges in canonical candidate order."""
    if window <= 0:
        raise ValueError("packing window must be positive")
    for begin in range(0, count, window):
        yield begin, min(window, count - begin)


def build_streaming_interaction_graph(
    iter_entries,
    graph_inputs,
    chunk_size,
    empty_indices,
    empty_values,
    execution,
    verbose=False,
):
    """Replay bounded pages for block support, chunk support, then energies.

    ``iter_entries(topology_only=...)`` returns fresh ordered (term, indices,
    values) pages on every call. Execution provides the seven native graph
    functions with their existing positional ABI. It must accumulate energy
    pages additively, preserving duplicate entries and transpose symmetry.
    The caller disables autodiff retention for this inference-only operation.
    """
    (
        max_n_block_types,
        n_rots_for_pose,
        rot_offset_for_pose,
        n_rots_for_block,
        rot_offset_for_block,
        pose_for_rot,
        block_type_ind_for_rot,
        block_ind_for_rot,
    ) = graph_inputs

    # Empty no-bump construction establishes the canonical kept/fixed rotamer
    # numbering without retaining any score layout.
    base = list(
        execution._build_interaction_graph(
            False,
            chunk_size,
            max_n_block_types,
            n_rots_for_pose,
            rot_offset_for_pose,
            n_rots_for_block,
            rot_offset_for_block,
            pose_for_rot,
            block_type_ind_for_rot,
            block_ind_for_rot,
            empty_indices,
            empty_values,
            # The table is empty: this call establishes the kept/fixed rotamer
            # numbering and retains no score layout, so "all entries unique"
            # is both true and irrelevant.
            -1,
            verbose,
        )
    )
    block_adjacency, orig_block_to_molten = (
        execution._initialize_interaction_graph_topology(
            n_rots_for_block,
            base[4],
            empty_values,
        )
    )

    # Pass one records block topology only. Chunk support is allocated after
    # the sparse block graph is known, avoiding a global n_chunks**2 bitset.
    for _, indices, values in iter_entries(topology_only=True):
        (block_adjacency,) = execution._note_interaction_graph_topology(
            block_ind_for_rot,
            orig_block_to_molten,
            block_adjacency,
            indices,
            values,
        )
        del indices, values

    chunk_topology = list(
        execution._finalize_interaction_graph_topology(
            chunk_size,
            base[4],
            block_adjacency,
            empty_values,
        )
    )
    del block_adjacency

    # Pass two records chunk support inside each observed block edge.
    for _, indices, values in iter_entries(topology_only=True):
        (chunk_topology[4],) = execution._note_interaction_graph_chunk_topology(
            chunk_size,
            n_rots_for_block,
            rot_offset_for_block,
            block_ind_for_rot,
            orig_block_to_molten,
            base[4],
            chunk_topology[0],
            chunk_topology[1],
            chunk_topology[3],
            chunk_topology[4],
            indices,
            values,
        )
        del indices, values

    base[11], base[12], base[13] = chunk_topology[:3]
    base[14], base[15] = execution._finalize_interaction_graph_chunk_topology(
        chunk_size,
        base[4],
        chunk_topology[0],
        chunk_topology[1],
        chunk_topology[2],
        chunk_topology[3],
        chunk_topology[4],
        empty_values,
    )
    del chunk_topology

    # Pass three adds terms in canonical order. Native accumulation is
    # additive, preserving duplicate coordinates and CSR transpose symmetry.
    for _, indices, values in iter_entries(topology_only=False):
        base[9], base[10], base[15] = execution._accumulate_interaction_graph_entries(
            chunk_size,
            n_rots_for_block,
            rot_offset_for_block,
            block_ind_for_rot,
            orig_block_to_molten,
            base[7],
            base[4],
            base[5],
            base[11],
            base[12],
            base[13],
            base[14],
            base[9],
            base[10],
            base[15],
            indices,
            values,
        )
        del indices, values
    return tuple(base)
