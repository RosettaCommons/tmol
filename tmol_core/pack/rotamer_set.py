"""Host projection of packed rotamers into Biotite structure annotations."""

import numpy as np

from tmol_core.chemical.restypes import _get_element_from_atom_name


def atom_array_from_rotamers(
    active_block_types,
    coords,
    block_type_ind_for_rot,
    block_ind_for_rot,
    coord_offset_for_rot,
):
    """Export each rotamer as a numbered model using the mature atom-name policy.

    Inputs are host arrays; frontends own device transfer. Residue numbers are
    one-based source block indices, independent of model (rotamer) indices.
    Biotite retains its existing behavior for an empty atom sequence.
    """
    import biotite.structure as struc

    atoms = []
    model_ids = []
    for i in range(len(block_ind_for_rot)):
        offset = int(coord_offset_for_rot[i])
        bt = active_block_types[int(block_type_ind_for_rot[i])]
        res_id = int(block_ind_for_rot[i]) + 1
        for j, atom in enumerate(bt.atoms):
            atoms.append(
                struc.Atom(
                    coords[offset + j],
                    chain_id="A",
                    res_id=res_id,
                    res_name=bt.name3,
                    atom_name=atom.name,
                    element=_get_element_from_atom_name(atom.name),
                )
            )
            model_ids.append(i)
    result = struc.array(atoms)
    result.set_annotation("model", np.array(model_ids, dtype=np.int32))
    return result
