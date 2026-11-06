"""Project optional source annotations into common canonical atom order."""

import numpy

from .pdb_info import DEFAULT_ATOM_B_FACTOR, DEFAULT_ATOM_OCCUPANCY


def populate_optional_atom_metadata(
    biotite_structure,
    n_poses: int,
    n_residues: int,
    max_n_canonical_atoms: int,
    valid_res_inds,
    valid_atom_inds,
    valid_atom_mask,
):
    biotite_b_factors = None
    biotite_occupancy = None

    if hasattr(biotite_structure, "b_factor"):
        b_factor = numpy.asarray(biotite_structure.b_factor)
        biotite_b_factors = numpy.full(
            (n_poses, n_residues, max_n_canonical_atoms),
            DEFAULT_ATOM_B_FACTOR,
            dtype=numpy.float32,
        )
        if n_poses == 1 or b_factor.ndim == 1:
            biotite_b_factors[:, valid_res_inds, valid_atom_inds] = b_factor[
                valid_atom_mask
            ]
        else:
            for pose_ind in range(n_poses):
                biotite_b_factors[pose_ind, valid_res_inds, valid_atom_inds] = b_factor[
                    pose_ind
                ][valid_atom_mask]

    if hasattr(biotite_structure, "occupancy"):
        occupancy = numpy.asarray(biotite_structure.occupancy)
        biotite_occupancy = numpy.full(
            (n_poses, n_residues, max_n_canonical_atoms),
            DEFAULT_ATOM_OCCUPANCY,
            dtype=numpy.float32,
        )
        if n_poses == 1 or occupancy.ndim == 1:
            biotite_occupancy[:, valid_res_inds, valid_atom_inds] = occupancy[
                valid_atom_mask
            ]
        else:
            for pose_ind in range(n_poses):
                biotite_occupancy[pose_ind, valid_res_inds, valid_atom_inds] = (
                    occupancy[pose_ind][valid_atom_mask]
                )
    return biotite_b_factors, biotite_occupancy
