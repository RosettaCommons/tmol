"""Named torsion updates through shared selection and native kinematics."""

import jax
import numpy as np

from tmol_core.kinematics import BondDOFTypes
from tmol_core.kinematics.torsion_updates import (
    controlling_node,
    is_bonded_node,
    require_bonded_node,
    torsion_deltas,
    torsion_targets,
)
from tmol_jax.pose.query import _measure_torsions, _torsion_requests, residue_label
from .pose import PoseStackKinematics, _prepare_palette


def _node_resolver(pose_stack, kin):
    host, stencils = _prepare_palette(pose_stack.packed_block_types)
    ids = kin._host_data[0][1:]
    kfo_for_atom = np.full(
        pose_stack.n_poses * pose_stack.max_n_pose_atoms, -1, np.int32
    )
    kfo_for_atom[ids] = np.arange(1, len(ids) + 1)
    metadata = (
        np.asarray(pose_stack.block_type_ind),
        kin._host_data[12],
        stencils["uaid_for_torsion_by_inconn"],
        np.asarray(pose_stack.block_coord_offset),
        np.asarray(pose_stack.inter_residue_connections),
        host.atom_downstream_of_conn,
        pose_stack.max_n_pose_atoms,
        kfo_for_atom,
    )
    return lambda pose, block, torsion: controlling_node(
        *metadata, pose, block, torsion
    )


def torsion_labels(pose_stack, kin_module, pose):
    """Author residue labels and forest nodes for bonded named torsions.

    ``kin_module`` is a prepared :class:`PoseStackKinematics` for this pose.
    Node indices are returned on CPU, matching the host-query API.
    """
    resolve = _node_resolver(pose_stack, kin_module)
    kinds = kin_module.forest._indices.doftype
    labels, nodes = [], []
    for _, block, torsion, name in _torsion_requests(pose_stack, pose, None, None):
        node = resolve(pose, block, torsion)
        if is_bonded_node(node, kinds):
            labels.append(f"{residue_label(pose_stack, pose, block)} {name}")
            nodes.append(node)
    return labels, jax.device_put(np.asarray(nodes, np.int32), jax.devices("cpu")[0])


def _apply_torsion_deltas(pose_stack, requests, deltas, fold_forest=None):
    kin = PoseStackKinematics(pose_stack, fold_forest)
    mapping = kin.bind()
    dofs = mapping.initial_dofs
    resolve = _node_resolver(pose_stack, kin)
    for request, delta in zip(requests, deltas):
        node = resolve(*request[:3])
        require_bonded_node(node, kin.forest._indices.doftype, request)
        # Repeated requests may alias one controlling node: add sequentially.
        dofs = dofs.at[node, BondDOFTypes.phi_c].add(delta)
    return pose_stack.replace(coords=mapping(dofs))


def set_named_torsions(
    pose_stack, poses, blocks, names, values, degrees=True, fold_forest=None
):
    """Set scalar or batched named torsions, preserving chemistry and input.

    Selection and validation use the common mature policy. Native inverse and
    forward kinematics run on the pose device and preserve coordinate dtype.
    This eager preparation API is not differentiable through initialization;
    use a bound ``PoseStackKinematics`` map for differentiable DOF execution.
    """
    requests = _torsion_requests(pose_stack, poses, blocks, names)
    targets = torsion_targets(requests, values, degrees)
    current = _measure_torsions(pose_stack, requests, degrees=False)
    deltas = torsion_deltas(requests, targets, current)
    return _apply_torsion_deltas(pose_stack, requests, deltas, fold_forest)
