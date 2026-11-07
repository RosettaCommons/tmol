"""Mature canonical atom identities, validation and neutral-terminal mapping."""

import numpy


def map_atoms_to_canonical(co, atom_res_inds, res_names, atom_names, elements):
    """Map Biotite atom names to canonical ordering indices.

    Returns (valid_atom_mask, valid_atom_inds, valid_res_inds).
    """

    atom_inds = []
    valid = []
    unmapped = set()
    destinations = set()
    for i, (resname, atname) in enumerate(zip(res_names, atom_names)):
        mapping = co.restypes_atom_index_mapping.get(resname, {})
        idx = mapping.get(atname, -1)
        atom_inds.append(idx)
        valid.append(idx >= 0)
        if idx >= 0:
            destination = (int(atom_res_inds[i]), idx)
            if destination in destinations:
                raise ValueError(
                    f"Multiple input atoms map to canonical atom {idx} of "
                    f"{resname} at residue index {destination[0]} ({atname}). "
                    "Resolve alternate locations and preserve residue identifiers "
                    "and insertion codes before constructing a pose."
                )
            destinations.add(destination)
        if idx < 0 and str(elements[i]).strip().upper() not in ("H", "D"):
            unmapped.add((int(atom_res_inds[i]), str(resname), str(atname)))

    if unmapped:
        details = ", ".join(
            f"{resname} at residue index {res}: {name}"
            for res, resname, name in sorted(unmapped)
        )
        raise ValueError(
            "Heavy atoms are absent from the selected chemical definitions: "
            f"{details}. Supply a matching chemical definition or correct the "
            "input atom names; these atoms cannot be silently discarded."
        )

    valid_atom_mask = numpy.array(valid, dtype=bool)
    atom_inds_arr = numpy.array(atom_inds, dtype=numpy.int64)
    return (
        valid_atom_mask,
        atom_inds_arr[valid_atom_mask],
        atom_res_inds[valid_atom_mask],
    )


def map_neutral_terminal_hydrogens(
    co,
    atom_res_inds,
    res_names,
    atom_names,
    variants,
    valid_atom_mask,
    valid_atom_inds,
):
    """Map supplied CCD H into neutral amino H1 unless H1 was supplied itself."""
    from tmol_core.chemical import NEUTRAL_TERMINUS_VAR_BASE

    neutral = variants >= NEUTRAL_TERMINUS_VAR_BASE
    for atom in numpy.flatnonzero(neutral & (atom_names == "H")):
        mapping = co.restypes_atom_index_mapping[res_names[atom]]
        residue = atom_res_inds == atom_res_inds[atom]
        if "H1" in mapping and not numpy.any(residue & (atom_names == "H1")):
            valid_atom_inds[valid_atom_mask.nonzero()[0] == atom] = mapping["H1"]
    return valid_atom_inds
