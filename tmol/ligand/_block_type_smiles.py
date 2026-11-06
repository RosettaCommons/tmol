"""Torch pose assembly over shared ligand fragment chemistry."""

from tmol_core.ligand import _block_type_smiles as _core
import numpy
from tmol_core.ligand._block_type_smiles import _site_connections, _smiles


def __getattr__(name):
    return getattr(_core, name)


def smiles_from_pose_stack(
    pose_stack,
    pose_ind: int,
    block_ind: int,
    *,
    hydrogens: bool = False,
    atom_map: bool = False,
) -> str:
    """Canonical SMILES of one residue of a PoseStack, with stereochemistry from its
    coordinates and a [*] for each connection that is not a metal bond.

    Args:
        pose_stack: The PoseStack.
        pose_ind: Pose index.
        block_ind: Residue (block) index within the pose; must be a real block.
        hydrogens: Keep explicit hydrogens in the SMILES.
        atom_map: Number each atom by its block-type atom index + 1 (hydrogens
            only when kept).
    """
    pbt = pose_stack.packed_block_types
    block_type = pose_stack.block_type(pose_ind, block_ind)
    coords = pose_stack.coords[pose_ind].detach().cpu().numpy()
    offsets = pose_stack.block_coord_offset[pose_ind].tolist()
    block_types = pose_stack.block_type_ind64[pose_ind].tolist()
    partners = pose_stack.inter_residue_connections64[pose_ind, block_ind].tolist()
    offset = offsets[block_ind]

    own_sites = set(_site_connections(block_type))
    attachments = []
    for c, conn in enumerate(block_type.connections):
        if conn.name in own_sites:
            continue
        partner, partner_conn = partners[c]
        if partner < 0:
            attachments.append((c, numpy.full(3, numpy.nan)))
            continue
        partner_type = pbt.active_block_types[block_types[partner]]
        if partner_type.connections[partner_conn].name in _site_connections(
            partner_type
        ):
            continue
        partner_atom = partner_type.ordered_connection_atoms[partner_conn]
        attachments.append((c, coords[offsets[partner] + partner_atom]))

    return _smiles(
        block_type,
        chem_db=pbt.chem_db,
        atom_coords=coords[offset : offset + block_type.n_atoms],
        attachments=attachments,
        hydrogens=hydrogens,
        atom_map=atom_map,
    )
