"""Eager packing orchestration over shared conformer, score, graph and search kernels."""

import jax
import numpy as np

from .rotamers import build_rotamers
from .interaction_graph import build_interaction_graph
from .annealing import anneal_cpu, anneal_cuda
from .imposition import impose_top_rotamer_assignments


def pack_rotamers(pose_stack, sfxn, task, *, seed, chunk_size=16):
    """Select conformers using the complete sparse score and native search.

    Returns a new pose. The task must register its samplers, including fallback
    for uncovered blocks. All enabled score terms are included. This eager,
    discrete operation requires float32 coordinates and an explicit uint64
    seed. Candidate pruning and correlated-group search are not implemented.
    CPU and CUDA use their respective mature search schedules. The CUDA phase
    seeds are drawn locally from NumPy PCG64; no global RNG state is changed.
    """
    if isinstance(pose_stack.coords, jax.core.Tracer):
        raise TypeError("discrete packing requires eager coordinates and choices")
    if pose_stack.coords.dtype != np.float32:
        raise TypeError("native packing search requires float32 coordinates")
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 1 << 64:
        raise ValueError("packing seed must be an unsigned 64-bit integer")
    pose, rotamers = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    scorer = sfxn.render_rotamer_scoring_module(pose, rotamers)
    values, indices = scorer(rotamers.coords)
    graph = build_interaction_graph(rotamers, values, indices, chunk_size=chunk_size)
    r = graph["retained"]
    arguments = tuple(
        r[name]
        for name in (
            "block_pose_counts",
            "rotamer_pose_counts",
            "rotamer_pose_offsets",
            "block_rotamer_counts",
            "block_rotamer_offsets",
            "block_for_rotamer",
        )
    )
    arguments += tuple(
        graph[name]
        for name in (
            "neighbor_row_offsets",
            "neighbor_blocks",
            "neighbor_chunk_offset_offsets",
            "chunk_offsets",
            "energy1b",
            "energy2b",
        )
    )
    if pose.device.platform == "cpu":
        _, assignments = anneal_cpu(*arguments, chunk_size=chunk_size, seed=seed)
    else:
        seeds = np.random.PCG64(int(seed)).random_raw(3).view(np.uint8).reshape(3, 8)
        _, assignments = anneal_cuda(
            *arguments,
            chunk_size=chunk_size,
            phase_seeds=jax.device_put(seeds, pose.device),
            max_rotamers=int(np.asarray(r["rotamer_pose_counts"]).max(initial=0)),
        )
    return impose_top_rotamer_assignments(pose, rotamers, graph, assignments)
