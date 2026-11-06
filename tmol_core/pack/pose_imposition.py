"""Host metadata for imposing conformers on an existing pose topology."""

import numpy as np


def impose_pdb_atom_metadata(
    original_types,
    new_types,
    original_offsets,
    new_offsets,
    type_natoms,
    original_occupancy,
    original_b_factor,
    capacity,
    default_occupancy=1.0,
    default_b_factor=30.0,
):
    """Keep unchanged residue metadata and default newly designed residues.

    Coordinates and topology selection stay on the execution device. These
    arrays are host PDB metadata; padded atoms retain zero occupancy/B factor.
    """
    occupancy = np.zeros((new_types.shape[0], capacity), dtype=np.float32)
    b_factor = np.zeros_like(occupancy)
    for pose, block in zip(*np.nonzero(new_types >= 0)):
        kind = new_types[pose, block]
        count = type_natoms[kind]
        dest = slice(new_offsets[pose, block], new_offsets[pose, block] + count)
        if kind == original_types[pose, block]:
            source = slice(
                original_offsets[pose, block], original_offsets[pose, block] + count
            )
            occupancy[pose, dest] = original_occupancy[pose, source]
            b_factor[pose, dest] = original_b_factor[pose, source]
        else:
            occupancy[pose, dest] = default_occupancy
            b_factor[pose, dest] = default_b_factor
    return occupancy, b_factor
