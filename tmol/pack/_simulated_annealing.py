import torch

from tmol.pack._datatypes import PackerEnergyTables
from tmol.types._torch import Tensor

DEFAULT_ANNEAL_TRAJECTORIES = 128


def _run_simulated_annealing(
    energy_tables: PackerEnergyTables,
    n_trajectories: int = DEFAULT_ANNEAL_TRAJECTORIES,
) -> tuple[Tensor[torch.float32][:, :], Tensor[torch.int32][:, :, :]]:
    """Rank rotamer assignments by simulated annealing on the tables' device.

    Args:
        energy_tables: One- and two-body energies plus their rotamer layout.
        n_trajectories: High-temperature trajectories per pose; at least 4.
            The single-trajectory CPU annealer, used below 8 threads, ignores it.

    Returns:
        Scores shaped ``[n_poses, n_final_trajectories]`` and local rotamer
        assignments shaped ``[n_poses, n_final_trajectories, max_n_res]``.

    Notes:
        The best 25% of the high-temperature trajectories seed ten
        low-temperature trajectories each, and the best 25% of those are fully
        quenched: 128 trajectories leave 80 ranked assignments per pose.
    """
    from tmol.pack.compiled._compiled import _pack_anneal

    if n_trajectories < 4:
        raise ValueError(f"n_trajectories must be at least 4; got {n_trajectories}")

    return _pack_anneal(
        energy_tables.max_n_rotamers_per_pose,
        energy_tables.pose_n_res,
        energy_tables.pose_n_rotamers,
        energy_tables.pose_rotamer_offset,
        energy_tables.nrotamers_for_res,
        energy_tables.oneb_offsets,
        energy_tables.res_for_rot,
        energy_tables.chunk_size,
        energy_tables.neighbor_row_offsets,
        energy_tables.neighbor_blocks,
        energy_tables.neighbor_chunk_offset_offsets,
        energy_tables.chunk_offsets,
        energy_tables.energy1b,
        energy_tables.energy2b,
        n_trajectories,
    )
