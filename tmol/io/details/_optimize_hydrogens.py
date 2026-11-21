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
    flip_NHQ: bool = False,
):
    """Place hydrogens where optH puts them, differentiably in the input atoms.

    Packing is not differentiable, so optH runs in two steps:

    1. Pack proton chis (and, with flip_NHQ, ASN/GLN/HIS flips) without
       gradients, optionally minimize the proton chis from there, and measure
       how each moved atom sits relative to the atoms it is placed from.
    2. Build those atoms again from the input coordinates, outward from the
       atoms that did not move, with the measured dihedrals in place of the
       ideal ones. Flipped side chains get the ideal geometry the packer
       built them with.

    Gradients reach every moved atom through the input atoms it is placed
    from; every other atom keeps its input coordinate. Without flip_NHQ, heavy
    atoms, block types and the atom layout are unchanged. With it, a histidine
    may change tautomer, so block types are the packed pose's.

    Returns:
        The pose and, when a block type changed, the index in it of each
        input pose atom ``[n_poses, max_n_pose_atoms]`` (-1 where the atom no
        longer exists); otherwise None, as the layout is the input's.
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
            flip_NHQ=flip_NHQ,
        )
        if minimize:
            packed = _minimize_proton_chis(packed, sfxn)

    pbt = packed.packed_block_types
    # only a flip can retype a block; skip the device synchronization otherwise
    retyped = flip_NHQ and bool(
        torch.any(packed.block_type_ind64 != pose_stack.block_type_ind64)
    )
    pose, block, atom = _block_atoms(packed)
    ind = packed.block_coord_offset64[pose, block] + atom
    types = packed.block_type_ind64[pose, block]
    if retyped:
        source = _source_atoms(pose_stack, packed, pose, block, atom)
        has_source = source >= 0
        coords = torch.full_like(packed.coords, torch.nan).index_put(
            (pose[has_source], ind[has_source]),
            pose_stack.coords[pose[has_source], source[has_source]],
        )
    else:
        coords = pose_stack.coords
        has_source = torch.ones_like(ind, dtype=torch.bool)
    moved = ~has_source | (
        torch.linalg.vector_norm(
            packed.coords[pose, ind] - coords.detach()[pose, ind], dim=-1
        )
        > _MOVED_TOLERANCE
    )
    if not flip_NHQ:
        # only hydrogens can have moved; refolding jitters nothing past this
        moved &= pbt.atom_is_hydrogen[types, atom] != 0
    pose, block, atom, ind, types = (x[moved] for x in (pose, block, atom, ind, types))

    # step 1, continued: each moved atom's primary frame, and its dihedral there
    frame = torch.stack(
        [
            _resolve_uaids(
                packed,
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
    dihedrals = torch.full_like(packed.coords[..., 0], torch.nan)
    dihedrals[pose[framed], ind[framed]] = measured[framed].to(dihedrals.dtype)

    # step 2: build each moved atom once its frame is in place. A hydrogen's
    #    frame is heavy atoms, so without flips one pass builds them all; a
    #    flipped ring atom is built from other flipped atoms
    placed = ~torch.isnan(coords.detach()).any(dim=-1)
    placed[pose, ind] = False
    unbuilt = torch.ones_like(ind, dtype=torch.bool)
    while True:
        ready = unbuilt
        if flip_NHQ:
            ready = unbuilt & framed & placed[pose[:, None], frame.clamp_min(0)].all(1)
            if not bool(torch.any(ready)):
                # no frame reaches them: the stored backup frames will
                ready = unbuilt
        targets = torch.zeros(
            (*packed.block_type_ind.shape, pbt.max_n_atoms),
            dtype=torch.bool,
            device=packed.device,
        )
        targets[pose[ready], block[ready], atom[ready]] = True
        coords = _build_coords_from_icoors(
            pbt,
            coords,
            targets,
            ~placed,
            packed.block_coord_offset,
            packed.block_type_ind,
            packed.inter_residue_connections,
            dihedrals=dihedrals,
        )
        placed[pose[ready], ind[ready]] = True
        unbuilt &= ~ready
        if not flip_NHQ or not bool(torch.any(unbuilt)):
            break

    result = copy.copy(packed if retyped else pose_stack)
    result.pdb_info = pose_stack.pdb_info
    result.constraint_set = pose_stack.constraint_set
    result.coords = coords
    if not retyped:
        return result, None
    new_for_old = torch.full(
        pose_stack.coords.shape[:2], -1, dtype=torch.int64, device=packed.device
    )
    p, b, a = _block_atoms(packed)
    new_for_old[p[has_source], source[has_source]] = (
        packed.block_coord_offset64[p, b] + a
    )[has_source]
    return result, new_for_old


# Stop after two consecutive steps that each improve the score by less than
# 0.05 (the defaults are five steps and 0.01). On 50 PDB structures this cut
# minimization time from 58 to 32 ms per structure on GPU, and raised the
# median total score by 0.03 (at most 0.14) relative to the defaults.
_PROTON_CHI_MIN_OPTIONS = {"atol": 0.05, "patience": 2}


def _minimize_proton_chis(pose_stack: PoseStack, sfxn) -> PoseStack:
    """Minimize the proton chis in torsion space, all other DOFs held fixed."""
    from tmol.kinematics._fold_forest import FoldForest
    from tmol.kinematics._move_map import MoveMap
    from tmol.optimization._minimizers import run_kin_min

    types = pose_stack.block_type_ind64
    selected = (types >= 0)[..., None] & _proton_chi_torsions(
        pose_stack.packed_block_types
    )[types.clamp_min(0)]
    if not bool(torch.any(selected)):
        return pose_stack
    move_map = MoveMap.from_pose_stack(pose_stack)
    move_map.set_move_named_torsion_for_blocks(*torch.nonzero(selected, as_tuple=True))
    # the minimizer differentiates the score, even if the caller turned grad off
    with torch.enable_grad():
        minimized = run_kin_min(
            pose_stack,
            sfxn,
            FoldForest.reasonable_fold_forest(pose_stack),
            move_map,
            optimizer_kwargs=_PROTON_CHI_MIN_OPTIONS,
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


def _source_atoms(pose_stack, packed, pose, block, atom):
    """The input pose atom each packed atom started as, or -1 if none did."""
    old_types = pose_stack.block_type_ind64
    new_types = packed.block_type_ind64
    source = pose_stack.block_coord_offset64[pose, block] + atom
    retyped = (old_types != new_types)[pose, block]
    # optH changed a histidine's tautomer; its atoms are matched by name
    pbt = packed.packed_block_types
    for i in torch.nonzero(retyped).flatten().tolist():
        p, b, a = int(pose[i]), int(block[i]), int(atom[i])
        name = pbt.active_block_types[int(new_types[p, b])].atoms[a].name
        old = pbt.active_block_types[int(old_types[p, b])].atom_to_idx.get(name)
        source[i] = -1 if old is None else source[i] - a + old
    return source


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
