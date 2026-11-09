import copy

import torch

from tmol.numeric import coord_dihedrals
from tmol.pose._pose_stack import PoseStack
from tmol.io.details._build_missing_leaf_atoms import _build_coords_from_icoors

# A hydrogen farther than this from its input position was placed by optH;
# refolding the unchanged atoms moves them by orders of magnitude less.
_MOVED_TOLERANCE = 1e-3


def _optimize_hydrogens(
    pose_stack: PoseStack,
    sfxn,
    seed: int | None = None,
    minimize: bool = False,
    fixed_blocks=None,
) -> PoseStack:
    """Place hydrogens where optH puts them, differentiably in the input atoms.

    Packing is not differentiable, so optH runs in two steps:

    1. Pack proton chis without gradients, optionally minimize them from
       there, and measure the dihedral each hydrogen that moved has in the
       frame it is built from.
    2. Rebuild those hydrogens from the input coordinates with the measured
       dihedrals in place of their ideal ones.

    Gradients reach the moved hydrogens through the atoms they are built
    from; every other atom keeps its input coordinate. Heavy atoms, block
    types and the atom layout are unchanged: ASN/GLN/HIS are not flipped.
    fixed_blocks, ``[n_poses, max_n_blocks]``, marks blocks left as they are.
    """
    from tmol.pack._build_missing_sidechains import _build_missing_sidechains

    with torch.no_grad():
        packed = _build_missing_sidechains(
            pose_stack,
            sfxn,
            None,
            torch.zeros_like(pose_stack.block_type_ind64, dtype=torch.bool),
            no_optH=False,
            has_missing_atoms=False,
            seed=seed,
            fixed_blocks=fixed_blocks,
        )
        if minimize:
            packed = _minimize_proton_chis(packed, sfxn, fixed_blocks)

    pbt = pose_stack.packed_block_types
    pose, block, atom = _block_atoms(pose_stack)
    ind = pose_stack.block_coord_offset64[pose, block] + atom
    types = pose_stack.block_type_ind64[pose, block]
    moved = (pbt.atom_is_hydrogen[types, atom] != 0) & (
        torch.linalg.vector_norm(
            packed.coords[pose, ind] - pose_stack.coords[pose, ind], dim=-1
        )
        > _MOVED_TOLERANCE
    )
    pose, block, atom, ind, types = (x[moved] for x in (pose, block, atom, ind, types))

    # step 1, continued: each moved hydrogen's primary frame, and its dihedral there
    frame = torch.stack(
        [
            _resolve_uaids(
                pose_stack,
                pose,
                block,
                pbt.build_missing_leaf_atom_icoor_ann.anc_uaids[types, atom, i],
            )
            for i in range(3)
        ],
        dim=1,
    )
    framed = (frame >= 0).all(dim=1)
    frame_coords = packed.coords[pose[:, None], frame.clamp_min(0)].double()
    measured = coord_dihedrals(
        frame_coords[:, 2],
        frame_coords[:, 1],
        frame_coords[:, 0],
        packed.coords[pose, ind].double(),
    )
    dihedrals = torch.full_like(pose_stack.coords[..., 0], torch.nan)
    dihedrals[pose[framed], ind[framed]] = measured[framed].to(dihedrals.dtype)

    # step 2: a hydrogen's frame is heavy atoms, all at their input coordinates
    targets = torch.zeros(
        (*pose_stack.block_type_ind.shape, pbt.max_n_atoms),
        dtype=torch.bool,
        device=pose_stack.device,
    )
    targets[pose, block, atom] = True
    missing = torch.zeros_like(pose_stack.coords[..., 0], dtype=torch.bool)
    missing[pose, ind] = True
    result = copy.copy(pose_stack)
    result.coords = _build_coords_from_icoors(
        pbt,
        pose_stack.coords,
        targets,
        missing,
        pose_stack.block_coord_offset,
        pose_stack.block_type_ind,
        pose_stack.inter_residue_connections,
        dihedrals=dihedrals,
    )
    return result


def _minimize_proton_chis(pose_stack: PoseStack, sfxn, fixed_blocks=None) -> PoseStack:
    """Minimize the proton chis of all but fixed_blocks in torsion space, all
    other DOFs held fixed."""
    from tmol.kinematics._fold_forest import FoldForest
    from tmol.kinematics._move_map import MoveMap
    from tmol.optimization._minimizers import run_kin_min

    types = pose_stack.block_type_ind64
    selected = (types >= 0)[..., None] & _proton_chi_torsions(
        pose_stack.packed_block_types
    )[types.clamp_min(0)]
    if fixed_blocks is not None:
        selected &= ~fixed_blocks[..., None]
    if not bool(torch.any(selected)):
        return pose_stack
    move_map = MoveMap.from_pose_stack(pose_stack)
    move_map.set_move_named_torsion_for_blocks(*torch.nonzero(selected, as_tuple=True))
    # the minimizer differentiates the score, even if the caller turned grad off
    with torch.enable_grad():
        minimized = run_kin_min(
            pose_stack, sfxn, FoldForest.reasonable_fold_forest(pose_stack), move_map
        )
    minimized.coords = minimized.coords.detach()
    return minimized


def _proton_chi_torsions(pbt):
    """[n_types, max_n_torsions]: which of each block type's torsions are proton chis."""
    if not hasattr(pbt, "opth_proton_chi_torsions"):
        table = torch.zeros((pbt.n_types, pbt.max_n_torsions), dtype=torch.bool)
        for i, bt in enumerate(pbt.active_block_types):
            index = {torsion.name: j for j, torsion in enumerate(bt.torsions)}
            for sample in bt.chi_samples:
                if sample.is_proton and sample.chi_dihedral in index:
                    table[i, index[sample.chi_dihedral]] = True
        pbt.opth_proton_chi_torsions = table.to(pbt.device)
    return pbt.opth_proton_chi_torsions


def _blocks_with_given_proton_hydrogens(pbt, block_types64, given):
    """[n_poses, max_n_blocks]: blocks with an input hydrogen on a proton chi.

    given, ``[n_poses, max_n_blocks, max_n_atoms]``, marks the atoms the input
    supplied. optH leaves such blocks alone so it does not move those hydrogens.
    """
    table = _proton_chi_hydrogens(pbt)[block_types64.clamp_min(0)]
    return ((block_types64 >= 0)[..., None] & table & given).any(dim=-1)


def _proton_chi_hydrogens(pbt):
    """[n_types, max_n_atoms]: the hydrogens a proton chi of each block type turns."""
    if not hasattr(pbt, "opth_proton_chi_hydrogens"):
        is_h = pbt.atom_is_hydrogen.cpu() != 0
        table = torch.zeros((pbt.n_types, pbt.max_n_atoms), dtype=torch.bool)
        for i, bt in enumerate(pbt.active_block_types):
            for sample in bt.chi_samples:
                uaids = bt.torsion_to_uaids.get(sample.chi_dihedral)
                if not sample.is_proton or uaids is None or uaids[2][0] < 0:
                    continue
                # the chi turns everything bonded to its third atom past the axis
                axis = (int(uaids[1][0]), int(uaids[2][0]))
                for a, b in bt.bond_indices:
                    a, b = int(a), int(b)
                    if a == axis[1] and b != axis[0] and is_h[i, b]:
                        table[i, b] = True
        pbt.opth_proton_chi_hydrogens = table.to(pbt.device)
    return pbt.opth_proton_chi_hydrogens


def _block_atoms(pose_stack):
    """The (pose, block, block atom) of each real atom of a pose."""
    pbt = pose_stack.packed_block_types
    types = pose_stack.block_type_ind64
    n_atoms = torch.where(types >= 0, pbt.n_atoms[types.clamp_min(0)], 0)
    real = torch.arange(pbt.max_n_atoms, device=pose_stack.device) < n_atoms[..., None]
    return torch.nonzero(real, as_tuple=True)


def _resolve_uaids(pose_stack, pose, block, uaids):
    """Pose atom indices of unresolved atom ids ``[n, 3]``; -1 if unresolvable."""
    pbt = pose_stack.packed_block_types
    local, connection, separation = uaids.long().unbind(-1)
    offset = pose_stack.block_coord_offset64
    partner, port = pose_stack.inter_residue_connections64[
        pose, block, connection.clamp_min(0)
    ].unbind(-1)
    partner_type = pose_stack.block_type_ind64[pose, partner.clamp_min(0)]
    partner_atom = pbt.atom_downstream_of_conn[
        partner_type, port.clamp_min(0), separation.clamp_min(0)
    ].long()
    across = torch.where(
        (connection >= 0) & (partner >= 0) & (port >= 0) & (partner_atom >= 0),
        offset[pose, partner.clamp_min(0)] + partner_atom,
        -1,
    )
    return torch.where(local >= 0, offset[pose, block] + local, across)
