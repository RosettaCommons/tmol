"""Kinematic operations on a PoseStack: DOF setting and torsion manipulation.

Functions here sit above both tmol.pose (which holds the PoseStack data
structure) and tmol.kinematics (which holds the KinForest and inverse-kin
machinery).  They are placed in tmol.kinematics rather than tmol.pose so that
tmol.pose itself carries no kinematics dependency.
"""

from __future__ import annotations

import attr
import torch

from tmol.kinematics.compiled._compiled_ops import _inverse_kin
from tmol.kinematics._datatypes import BondDOFTypes
from tmol.kinematics._fold_forest import FoldForest
from tmol.kinematics._script_modules import PoseStackKinematicsModule
from tmol.pose._pose_stack import PoseStack
from tmol.pose._util import _measure_torsions, _torsion_requests
from tmol.pose import residue_label
from tmol_core.kinematics.torsion_updates import (
    controlling_node,
    is_bonded_node,
    require_bonded_node,
    torsion_deltas,
    torsion_targets,
)


def _controlling_kfo_node(pose_stack, kmd, kfo_for_atom, pose, block, tor_ind):
    pbt = pose_stack.packed_block_types
    return controlling_node(
        pose_stack.block_type_ind64,
        kmd.block_in_and_first_out,
        pbt.gen_seg_scan_path_segs.uaid_for_torsion_by_inconn,
        pose_stack.block_coord_offset64,
        pose_stack.inter_residue_connections64,
        pbt.atom_downstream_of_conn,
        pose_stack.max_n_pose_atoms,
        kfo_for_atom,
        pose,
        block,
        tor_ind,
    )


def torsion_labels(pose_stack, kin_module, pose):
    """Labels and kinforest nodes for the named torsions of one pose driven by a bond phi_c."""
    kmd = kin_module.kmd
    ids = kmd.forest.id[1:].to(torch.int64)
    kfo_for_atom = torch.full(
        (pose_stack.n_poses * pose_stack.max_n_pose_atoms,), -1, dtype=torch.int64
    )
    kfo_for_atom[ids.cpu()] = torch.arange(1, ids.shape[0] + 1)

    labels, nodes = [], []
    for _, block, tor_ind, name in _torsion_requests(pose_stack, pose, None, None):
        node = _controlling_kfo_node(
            pose_stack, kmd, kfo_for_atom, pose, block, tor_ind
        )
        if not is_bonded_node(node, kmd.forest.doftype):
            continue
        labels.append(f"{residue_label(pose_stack, pose, block)} {name}")
        nodes.append(node)
    return labels, torch.tensor(nodes, dtype=torch.int64)


def _apply_torsion_deltas(pose_stack, requests, deltas, fold_forest=None):
    """Apply a set of torsion deltas to a PoseStack using inverse kinematics."""
    if fold_forest is None:
        fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
    kin_module = PoseStackKinematicsModule(pose_stack, fold_forest)
    kmd = kin_module.kmd

    kincoords = torch.zeros(
        (kmd.forest.id.shape[0], 3), dtype=torch.float32, device=pose_stack.device
    )
    kincoords[1:] = pose_stack.coords.view(-1, 3)[kmd.forest.id[1:].to(torch.int64)]
    dofs = _inverse_kin(
        kincoords,
        kmd.forest.parent,
        kmd.forest.frame_x,
        kmd.forest.frame_y,
        kmd.forest.frame_z,
        kmd.forest.doftype,
    )

    kfo_for_atom = torch.full(
        (pose_stack.n_poses * pose_stack.max_n_pose_atoms,),
        -1,
        dtype=torch.int64,
        device=pose_stack.device,
    )
    ids = kmd.forest.id[1:].to(torch.int64)
    kfo_for_atom[ids] = torch.arange(
        1, ids.shape[0] + 1, dtype=torch.int64, device=pose_stack.device
    )

    for (pose, block, tor_ind, name), delta in zip(requests, deltas):
        node = _controlling_kfo_node(
            pose_stack, kmd, kfo_for_atom, pose, block, tor_ind
        )
        require_bonded_node(node, kmd.forest.doftype, (pose, block, tor_ind, name))
        dofs[node, BondDOFTypes.phi_c] += delta

    kin_coords = kin_module(dofs)
    flat = pose_stack.coords.reshape(-1, 3).clone()
    flat[ids] = kin_coords[1:].to(flat.dtype)
    return attr.evolve(pose_stack, coords=flat.view(pose_stack.coords.shape))


def set_named_torsions(
    pose_stack: PoseStack,
    poses,
    blocks,
    names,
    values,
    degrees: bool = True,
    fold_forest=None,
) -> PoseStack:
    """Set named torsions, returning a new PoseStack.

    ``poses``, ``blocks``, ``names`` and ``values`` are either all scalars or
    equal-length sequences; a batch is applied in one kinematic pass. Movement
    follows the fold forest, so which side of each bond stays fixed depends on
    it; ``reasonable_fold_forest`` roots each chain at its first residue.
    """
    requests = _torsion_requests(pose_stack, poses, blocks, names)
    targets = torsion_targets(requests, values, degrees)
    current = _measure_torsions(pose_stack, requests, degrees=False)
    deltas = torsion_deltas(requests, targets, current)
    return _apply_torsion_deltas(pose_stack, requests, deltas, fold_forest=fold_forest)
