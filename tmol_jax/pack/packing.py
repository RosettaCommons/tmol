"""Eager packing orchestration over shared conformer, score, graph and search kernels."""

import jax
import numpy as np

from .rotamers import build_rotamers
from .interaction_graph import build_interaction_graph
from .annealing import anneal_cpu, anneal_cuda
from .imposition import impose_top_rotamer_assignments


def pack_rotamers(
    pose_stack, sfxn, task, *, seed=None, chunk_size=None, anneal_device=None
):
    """Select conformers using the complete sparse score and native search.

    Returns a new pose. The task must register its samplers, including fallback
    for uncovered blocks. All enabled score terms are included. This eager,
    discrete operation requires float32 coordinates. An explicit uint64 seed
    makes search reproducible; None draws a fresh local seed. Optional task bump
    checks use the mature native pruning rule. Correlated groups retain shared
    lockstep context; independently varying third-backbone context is unsupported.
    CPU and CUDA use their respective mature search schedules. The CUDA phase
    seeds are drawn locally from NumPy PCG64; no global RNG state is changed.
    ``anneal_device`` may select a CPU/CUDA device independently of scoring.
    Only prepared graph arrays move; the returned pose stays on its input device.
    """
    if isinstance(pose_stack.coords, jax.core.Tracer):
        raise TypeError("discrete packing requires eager coordinates and choices")
    if pose_stack.coords.dtype != np.float32:
        raise TypeError("native packing search requires float32 coordinates")
    if seed is None:
        seed = int(np.random.SeedSequence().generate_state(1, np.uint64)[0])
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 1 << 64:
        raise ValueError("packing seed must be an unsigned 64-bit integer")
    device = pose_stack.device if anneal_device is None else anneal_device
    if isinstance(device, str):
        platform, _, index = device.partition(":")
        if platform not in ("cpu", "cuda", "gpu"):
            raise ValueError("annealing supports native CPU and CUDA devices")
        index = int(index or 0)
        devices = jax.devices(platform)
        if not 0 <= index < len(devices):
            raise ValueError("annealing device index is out of range")
        device = devices[index]
    if not isinstance(device, jax.Device):
        raise TypeError("anneal_device must be a JAX device or a device string")
    if device.platform not in ("cpu", "gpu"):
        raise ValueError("annealing supports native CPU and CUDA devices")
    task._validate_pose(pose_stack)
    from .batching import max_poses_per_chunk, slice_task

    limit = max_poses_per_chunk(pose_stack)
    if pose_stack.n_poses > limit:
        from tmol_core.pack.streaming import packing_windows
        from tmol_jax.pose.pose_stack_builder import PoseStackBuilder

        seeds = np.random.PCG64(int(seed))
        chunks = []
        for first, count in packing_windows(pose_stack.n_poses, limit):
            pose = pose_stack[first : first + count]
            chunks.append(
                pack_rotamers(
                    pose,
                    sfxn,
                    slice_task(task, pose, first, first + count),
                    seed=int(seeds.random_raw()),
                    chunk_size=chunk_size,
                    anneal_device=device,
                )
            )
        return PoseStackBuilder.from_poses(chunks, pose_stack.device)
    from tmol_core.pack.streaming import (
        interaction_graph_chunk_size,
        should_stream_interaction_graph,
    )

    kind = "cuda" if pose_stack.device.platform == "gpu" else pose_stack.device.platform
    if chunk_size is None:
        chunk_size = interaction_graph_chunk_size(kind)
    pose, rotamers = build_rotamers(
        pose_stack, task, pose_stack.packed_block_types.chem_db
    )
    scorer = sfxn.render_rotamer_scoring_module(pose, rotamers)
    from .streaming import streaming_interaction_graph, use_complete_table

    stream = should_stream_interaction_graph(
        kind, rotamers.correlated_groups, task.bump_check
    )
    if stream and not use_complete_table(scorer, rotamers.coords):

        graph = streaming_interaction_graph(
            scorer, rotamers.coords, chunk_size=chunk_size
        )
    else:
        values, indices = scorer(rotamers.coords)
        graph = build_interaction_graph(
            rotamers, values, indices, chunk_size=chunk_size, bump_check=task.bump_check
        )
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
    if device != pose.device:
        arguments = jax.device_put(arguments, device)
    if device.platform == "cpu":
        _, assignments = anneal_cpu(*arguments, chunk_size=chunk_size, seed=seed)
    else:
        seeds = np.random.PCG64(int(seed)).random_raw(3).view(np.uint8).reshape(3, 8)
        _, assignments = anneal_cuda(
            *arguments,
            chunk_size=chunk_size,
            phase_seeds=jax.device_put(seeds, device),
            max_rotamers=int(np.asarray(r["rotamer_pose_counts"]).max(initial=0)),
        )
    if device != pose.device:
        assignments = jax.device_put(assignments, pose.device)
    return impose_top_rotamer_assignments(pose, rotamers, graph, assignments)
