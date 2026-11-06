import attr
import torch

from tmol.types._torch import Tensor
from tmol.pose._pdb_info import DEFAULT_ATOM_B_FACTOR, DEFAULT_ATOM_OCCUPANCY
from tmol.pose._pose_stack import PoseStack
from tmol.pack.rotamer._rotamer_set import RotamerSet
from tmol_core.pack.pose_imposition import impose_pdb_atom_metadata


def _imposition_inputs(orig, fixed, n_molten, offsets, mapping, choices):
    inputs = [orig.block_type_ind64, fixed, n_molten, offsets, mapping, choices]
    # Metal retains native CPU metadata planning until it has a device binding.
    device = orig.device if orig.device.type in ("cpu", "cuda") else torch.device("cpu")
    return [value.to(device).contiguous() for value in inputs], device


def _chosen_rotamer_for_block(
    orig_pose_stack,
    rotamer_for_nonmolten_block,
    n_molten_blocks_per_pose,
    bc_rot_offset_for_molten_block,
    bc_rot_to_orig_rot,
    bc_assignment,
):
    """Map retained choices to original conformers, preserving fixed blocks."""
    from tmol.pack.rotamer._conformer_compiled import select_pose_rotamers

    inputs, _ = _imposition_inputs(
        orig_pose_stack,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bc_assignment,
    )
    return select_pose_rotamers(inputs).to(orig_pose_stack.device)


def _impose_top_rotamer_assignments(
    orig_pose_stack: PoseStack,
    rotamer_set: RotamerSet,
    rotamer_for_nonmolten_block: Tensor[torch.int64][:, :],
    n_molten_blocks_per_pose: Tensor[torch.int64][:],
    bc_rot_offset_for_molten_block: Tensor[torch.int64][:, :],
    bc_rot_to_orig_rot: Tensor[torch.int64][:],
    bc_assignment: Tensor[torch.int32][:, :, :],
):
    """Impose the lowest-energy assignment using the common native layout."""
    from tmol.pack.rotamer._conformer_compiled import plan_pose_imposition

    pbt = orig_pose_stack.packed_block_types
    device = orig_pose_stack.device
    n_poses = orig_pose_stack.n_poses
    inputs, plan_device = _imposition_inputs(
        orig_pose_stack,
        rotamer_for_nonmolten_block,
        n_molten_blocks_per_pose,
        bc_rot_offset_for_molten_block,
        bc_rot_to_orig_rot,
        bc_assignment[:, 0, :],
    )
    inputs.extend(
        value.to(plan_device).contiguous()
        for value in (
            rotamer_set.block_type_ind_for_rot,
            pbt.n_atoms,
            rotamer_set.coord_offset_for_rot,
            orig_pose_stack.block_coord_offset,
        )
    )
    _, types, offsets, src, dst, metadata = plan_pose_imposition(
        inputs,
        orig_pose_stack.max_n_pose_atoms,
        rotamer_set.coords.shape[0],
    )
    capacity, changed = metadata.tolist()
    new_block_type_ind64 = types.to(device)
    if changed:
        new_n_atoms_offset32 = offsets.to(device)
        new_n_atoms_offset64 = new_n_atoms_offset32.to(torch.int64)
        occupancy, b_factor = impose_pdb_atom_metadata(
            orig_pose_stack.block_type_ind64.cpu().numpy(),
            types.cpu().numpy(),
            orig_pose_stack.block_coord_offset.cpu().numpy(),
            offsets.cpu().numpy(),
            pbt.n_atoms.cpu().numpy(),
            orig_pose_stack.pdb_info.atom_occupancy,
            orig_pose_stack.pdb_info.atom_b_factor,
            capacity,
            DEFAULT_ATOM_OCCUPANCY,
            DEFAULT_ATOM_B_FACTOR,
        )
        new_pdb_info = attr.evolve(
            orig_pose_stack.pdb_info,
            atom_occupancy=occupancy,
            atom_b_factor=b_factor,
        )
    else:
        new_n_atoms_offset32 = orig_pose_stack.block_coord_offset
        new_n_atoms_offset64 = orig_pose_stack.block_coord_offset64
        new_pdb_info = orig_pose_stack.pdb_info
    new_coords = torch.zeros(
        (n_poses * capacity, 3), dtype=torch.float32, device=device
    )
    new_coords[dst.to(device)] = rotamer_set.coords[src.to(device)]
    new_coords = new_coords.view(n_poses, capacity, 3)

    # now construct the new PoseStack
    return PoseStack(
        packed_block_types=pbt,
        coords=new_coords,
        block_coord_offset=new_n_atoms_offset32,
        block_coord_offset64=new_n_atoms_offset64,
        inter_residue_connections=orig_pose_stack.inter_residue_connections,
        inter_residue_connections64=orig_pose_stack.inter_residue_connections64,
        inter_block_bondsep=orig_pose_stack.inter_block_bondsep,
        block_type_ind=new_block_type_ind64.to(torch.int32),
        block_type_ind64=new_block_type_ind64,
        chain_id=orig_pose_stack.chain_id,
        chain_id64=orig_pose_stack.chain_id64,
        pdb_info=new_pdb_info,
        constraint_set=orig_pose_stack.constraint_set,
        device=device,
        split_block_mapping=orig_pose_stack.split_block_mapping,
    )
