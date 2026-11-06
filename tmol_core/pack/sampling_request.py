"""Concrete molecular/task selection shared by Dunbrack sampling frontends."""

from dataclasses import dataclass
import numpy as np

from tmol_core.chemical.atom_resolution import resolve_uaids


@dataclass(frozen=True)
class DunbrackSamplingRequest:
    n_gbt: int
    max_n_chi: int
    has_allowed: bool
    bbt_to_gbt: np.ndarray
    block_type_ind_for_bbt: np.ndarray
    n_dihe_for_block: np.ndarray
    dihedral_offset_for_block: np.ndarray
    dihedral_atom_inds: np.ndarray
    bubl_and_rottable_set_for_bbt: np.ndarray
    chi_expansion_for_bbt: np.ndarray
    non_dunbrack_expansion_for_bbt: np.ndarray
    non_dunbrack_expansion_counts_for_bbt: np.ndarray
    prob_cumsum_limit_for_bbt: np.ndarray
    n_chi_for_bbt: np.ndarray
    frozen_requests: tuple
    frozen_destinations: np.ndarray
    validate_frozen_chi: bool


def backbone_dihedral_indices(
    metadata,
    block_type_ind,
    block_coord_offset,
    inter_residue_connections,
    atom_downstream_of_conn,
    max_n_pose_atoms,
):
    """Padded pose/block/phi-psi/atom indices; missing atoms remain -1."""
    shape = block_type_ind.shape + (2, 4)
    if not np.any(block_type_ind >= 0):
        return np.full(shape, -1, dtype=np.int32)
    if block_type_ind.shape[0] * max_n_pose_atoms > np.iinfo(np.int32).max:
        raise OverflowError("Dunbrack coordinate indices exceed int32 capacity")
    uaids = metadata.bbdihe_uaids[np.maximum(block_type_ind, 0)]
    pose, local, ok = resolve_uaids(
        uaids,
        block_type_ind,
        block_coord_offset,
        inter_residue_connections,
        atom_downstream_of_conn,
    )
    return np.where(ok, pose * max_n_pose_atoms + local, -1).astype(np.int32)


def prepare_dunbrack_sampling_request(
    block_types,
    metadata,
    block_type_ind,
    block_coord_offset,
    inter_residue_connections,
    atom_downstream_of_conn,
    max_n_pose_atoms,
    considered_pose,
    considered_block,
    considered_type,
    considered_slot,
    considered_allowed,
    sampler_allowed,
    chi_expansion,
):
    """Select concrete task rows and prepare one canonical native sampling call.

    Chemistry and topology are static inputs, while frozen torsion values are
    supplied separately by the execution adapter. A changed chemical identity
    keeps its generated ideal chi. No canonical-residue-name assumption is made.
    """
    allowed = sampler_allowed[considered_pose, considered_block] & considered_allowed
    allowed_gbt = np.flatnonzero(allowed)
    selected = allowed_gbt[
        metadata.defines_rotamers_for_bts[considered_type[allowed_gbt]]
    ]
    types = considered_type[selected]
    poses, blocks = considered_pose[selected], considered_block[selected]
    global_blocks = poses * block_type_ind.shape[1] + blocks
    unique_blocks, bubl = np.unique(global_blocks, return_inverse=True)
    n_blocks, n_types = len(unique_blocks), len(selected)
    table_sets = metadata.rottable_set_for_bt[types]
    bubl_and_tables = np.column_stack((bubl, table_sets)).astype(np.int32)

    # Resolve source and destination chemical types independently. A connection
    # can lead to a different residue type or an unresolved boundary.
    atoms = np.empty((0, 4), dtype=np.int32)
    if n_blocks:
        all_backbone = backbone_dihedral_indices(
            metadata,
            block_type_ind,
            block_coord_offset,
            inter_residue_connections,
            atom_downstream_of_conn,
            max_n_pose_atoms,
        )
        atoms = all_backbone.reshape(-1, 2, 4)[unique_blocks].reshape(-1, 4)
    expansions = chi_expansion[poses, blocks, considered_slot[selected]]
    # Burial classification is not implemented; both frontends preserve the
    # conservative buried-state library/proton sampling policy.
    counts = metadata.non_dunbrack_sample_counts[types, :, 1]
    samples = metadata.non_dunbrack_samples[types, :, 1].copy()
    # Frozen gaps are real chi slots; only -1 marks padding.
    n_chi = np.sum(counts >= 0, axis=1, dtype=np.int32)

    requests, destinations = [], []
    if any(metadata.frozen_chi):
        for row, (type_index, pose, block) in enumerate(zip(types, poses, blocks)):
            if type_index != block_type_ind[pose, block]:
                continue
            torsions = {
                t.name: i for i, t in enumerate(block_types[type_index].torsions)
            }
            for name in metadata.frozen_chi[type_index]:
                requests.append((int(pose), int(block), torsions[name], name))
                destinations.append((row, int(name[3:]) - 1))
    return DunbrackSamplingRequest(
        len(considered_pose),
        metadata.max_n_chi,
        bool(len(allowed_gbt)),
        selected,
        types,
        np.full(n_blocks, 2, dtype=np.int32),
        2 * np.arange(n_blocks, dtype=np.int32),
        atoms,
        bubl_and_tables,
        expansions,
        samples,
        counts,
        np.full(n_types, 0.98, dtype=np.float32),
        n_chi,
        tuple(requests),
        np.asarray(destinations, dtype=np.int64).reshape(-1, 2),
        any(metadata.frozen_chi),
    )


def apply_frozen_chi(request, measured_values):
    """Apply resolved input torsions, keeping ideal values for missing atoms."""
    values = np.asarray(measured_values)
    if values.shape != (len(request.frozen_destinations),):
        raise ValueError("Frozen chi measurement count does not match the request")
    samples = request.non_dunbrack_expansion_for_bbt
    if len(values):
        row, chi = request.frozen_destinations.T
        samples[row, chi, 0] = np.where(
            np.isfinite(values), values, samples[row, chi, 0]
        )
    if request.validate_frozen_chi and not np.isfinite(samples).all():
        raise ValueError("Frozen chi lack resolved input or generated ideal geometry")
    return samples
