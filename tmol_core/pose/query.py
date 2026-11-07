"""Shared selection and result structure for named molecular torsion queries."""

import re

import numpy

from tmol_core.chemical.restypes import _three2one


def _is_scalar(x):
    return isinstance(x, (int, numpy.integer, str))


def _as_list(x, default):
    if x is None:
        return list(default)
    return [x] if _is_scalar(x) else list(x)


def torsion_requests(
    active_types, block_type_ind, poses=None, blocks=None, names=None
):  # noqa: C901
    """Expand a selection into (pose, block, torsion index, name) tuples."""
    bti = numpy.asarray(block_type_ind)
    n_poses, max_n_blocks = bti.shape

    pose_list = _as_list(poses, range(n_poses))
    block_list = _as_list(blocks, range(max_n_blocks))
    name_list = None if names is None else _as_list(names, ())

    paired = poses is not None and blocks is not None
    if paired and not _is_scalar(poses) and not _is_scalar(blocks):
        if len(pose_list) != len(block_list):
            raise ValueError("poses and blocks must be the same length")
        pairs = list(zip(pose_list, block_list))
    else:
        pairs = [(p, b) for p in pose_list for b in block_list]

    if name_list is not None and len(name_list) == len(pairs) and len(pairs) > 1:
        paired_names = name_list
    else:
        paired_names = None

    requests = []
    for i, (pose, block) in enumerate(pairs):
        bt_ind = int(bti[pose, block])
        if bt_ind == -1:
            if blocks is None:
                continue
            raise ValueError(f"block {block} of pose {pose} is not a real block")
        bt = active_types[bt_ind]
        tor_inds = {tor.name: j for j, tor in enumerate(bt.torsions)}
        if paired_names is not None:
            wanted = [paired_names[i]]
        elif name_list is not None:
            wanted = name_list
        else:
            wanted = list(tor_inds)
        for name in wanted:
            if name not in tor_inds:
                if name_list is None or blocks is None:
                    continue
                raise ValueError(f"block type {bt.name} has no torsion {name!r}")
            requests.append((pose, block, tor_inds[name], name))
    return requests


def format_named_torsions(requests, values, shape, poses, blocks, names):
    """Preserve scalar, name-dictionary and pose/block nested output contracts."""
    if _is_scalar(poses) and _is_scalar(blocks):
        if _is_scalar(names):
            return float(values[0])
        return {
            name: value for (_, _, _, name), value in zip(requests, values.tolist())
        }
    out = [[{} for _ in range(shape[1])] for _ in range(shape[0])]
    for (pose, block, _, name), value in zip(requests, values.tolist()):
        out[pose][block][name] = value
    return out


def sequence_from_types(active_types, block_type_ind_row):
    """One letter per real block; unknown identities remain X, including ligands."""
    return "".join(
        _three2one(active_types[int(i)].name3) or "X"
        for i in block_type_ind_row
        if i >= 0
    )


def ligand_flags(active_types):
    """Per-type non-polymer flags; frontends mask padding when gathering."""
    return [not bt.properties.polymer.is_polymer for bt in active_types]


def polymer_type_flags(active_types, polymer_type):
    return [bt.properties.polymer.polymer_type == polymer_type for bt in active_types]


def metal_blocks(active_types, block_type_ind_row):
    return [
        block
        for block, i in enumerate(block_type_ind_row)
        if i >= 0 and active_types[int(i)].metal_sites
    ]


def residue_label(block_type, pdb_info, pose, block):
    """Preserve the public chain/residue-number/name3 display format."""
    return (
        f"{pdb_info.chain_labels[pose, block]} "
        f"{int(pdb_info.residue_labels[pose, block])} {block_type.name3}"
    )


def atom_label(active_types, block_type_ind_row, offsets, pdb_info, pose, atom):
    real = numpy.asarray(block_type_ind_row) >= 0
    offsets = numpy.asarray(offsets)
    block = int(numpy.flatnonzero((offsets <= atom) & real)[-1])
    bt = active_types[int(block_type_ind_row[block])]
    name = bt.atoms[atom - int(offsets[block])].name
    return f"{residue_label(bt, pdb_info, pose, block)} {name}"


_RESIDUE_ID = re.compile(r"^(?P<number>\d+)(?P<insertion_code>[A-Za-z]*)$")


def block_for_residue_label(pdb_info, pose_index, chain, residue_id):
    """Resolve a chain/author number/insertion-code label, using the first match."""
    match = _RESIDUE_ID.match(residue_id)
    if not match:
        raise ValueError(f"invalid residue id: {residue_id!r}")
    matches = numpy.flatnonzero(
        (pdb_info.chain_labels[pose_index] == chain)
        & (pdb_info.residue_labels[pose_index] == int(match.group("number")))
        & (
            pdb_info.residue_insertion_codes[pose_index]
            == match.group("insertion_code")
        )
    )
    if len(matches) == 0:
        raise ValueError(
            f"no block found for chain {chain!r} residue {residue_id!r} "
            f"in pose {pose_index}"
        )
    return int(matches[0])


def blocks_for_residue_labels(pdb_info, pose_index, chain, residue_numbers):
    """Batch legacy numeric labels; insertion codes are deliberately not queried."""
    matches = (
        pdb_info.residue_labels[pose_index] == numpy.asarray(residue_numbers)[..., None]
    ) & (pdb_info.chain_labels[pose_index] == chain)
    return numpy.where(matches.any(axis=-1), matches.argmax(axis=-1), -1)


def atom_for_residue_label(
    active_types, block_type_ind, pdb_info, pose_ind, chain_id, res_id, atom_id
):
    block = block_for_residue_label(pdb_info, pose_ind, chain_id, res_id)
    block_type = active_types[int(block_type_ind[pose_ind, block])]
    if atom_id not in block_type.atom_to_idx:
        raise ValueError(
            f"residue type {block_type.name} (chain {chain_id!r} residue "
            f"{res_id!r} in pose {pose_ind}) has no atom {atom_id!r}"
        )
    return numpy.array(
        [pose_ind, block, block_type.atom_to_idx[atom_id]], dtype=numpy.int32
    )


def constraint_atoms_for_labels(active_types, block_type_ind, pdb_info, atoms):
    if not 1 <= len(atoms) <= 4:
        raise ValueError(f"atoms must specify 1 to 4 atoms, got {len(atoms)}")
    return numpy.stack(
        [
            atom_for_residue_label(active_types, block_type_ind, pdb_info, *label)
            for label in atoms
        ]
    )
