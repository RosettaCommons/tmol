"""A torsion adjacent to a ring must rotate an intact chemical component."""

import pytest
import torch

from tmol import pose_stack_from_atom_array
from tmol.dock import ligand_dofs as ld
from tmol.kinematics import PoseStackKinematicsModule
from tmol.ligand import (
    nonstandard_residue_info_from_mol2,
    prepare_ligand_from_mol2,
)
from tmol.tests.data import data_path
from tmol.pose import repeat_pose_stack


@pytest.mark.parametrize("n_poses", [1, 4])
def test_ring_adjacent_torsions_preserve_geometry(torch_device, n_poses):
    # Ligand-only 1nav geometry reproduces the jump-stub/ring interaction;
    # regenerating its conformer changes the chosen root and hides the failure.
    path = data_path("protein_ligand_test/dock_ring_ether.mol2")
    info = nonstandard_residue_info_from_mol2(path, res_name="LG1")
    database, _ = prepare_ligand_from_mol2(path, res_name="LG1", mode="keep")
    pose = pose_stack_from_atom_array(
        info.atom_array, torch_device, param_db=database, no_optH=True
    )
    pose.coords = pose.coords.double()
    pose = repeat_pose_stack(pose, n_poses)
    docking = ld.build_ligand_docking_dofs(pose, 0)
    kin = PoseStackKinematicsModule(pose, docking.fold_forest)
    raw = ld._inverse_dofs(pose, kin)
    block_type = pose.block_type(0, 0)
    nodes = ld._ligand_kinforest_nodes(pose, 0, block_type.n_atoms, kin.kmd.forest.id)
    torsion_atoms = torch.tensor(
        [ld._local_torsion_atoms(block_type, i) for i in range(block_type.n_torsions)],
        device=torch_device,
    )
    atom_nodes = nodes[:, torsion_atoms]
    original = kin(raw)
    original_chis = ld._batched_coord_dihedrals(original, atom_nodes)
    bonds = [(int(a), int(b)) for a, b in block_type.bond_indices]
    neighbors = [set() for _ in range(block_type.n_atoms)]
    for a, b in bonds:
        neighbors[a].add(b)
        neighbors[b].add(a)
    angles = [
        (a, b, c)
        for b, adjacent in enumerate(neighbors)
        for a in adjacent
        for c in adjacent
        if a < c
    ]

    def geometry(coords):
        xyz = coords[nodes]
        lengths = torch.stack(
            [torch.linalg.vector_norm(xyz[:, a] - xyz[:, b], dim=-1) for a, b in bonds]
        )
        cosines = torch.stack(
            [
                torch.nn.functional.cosine_similarity(
                    xyz[:, a] - xyz[:, b], xyz[:, c] - xyz[:, b], dim=-1
                )
                for a, b, c in angles
            ]
        )
        return lengths, cosines

    original_geometry = geometry(original)
    for index, mapping in enumerate(docking.torsions):
        for delta in (0.5, -1.2):
            changed = raw.clone()
            changed[nodes[:, mapping.raw_atom], int(mapping.dof_type)] += delta
            coords = kin(changed)
            for actual, expected in zip(geometry(coords), original_geometry):
                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-8)
            response = ld._wrap_angle(
                ld._batched_coord_dihedrals(coords, atom_nodes) - original_chis
            )
            expected = torch.zeros_like(response)
            expected[:, index] = mapping.sign * delta
            torch.testing.assert_close(response, expected, rtol=0, atol=1e-8)

    # Distinct simultaneous torsions in each pose also preserve the ring and
    # the logical-chi contract when the affected chemical components overlap.
    changed = raw.clone()
    deltas = (
        torch.linspace(-1.2, 0.8, n_poses * len(docking.torsions), device=torch_device)
        .double()
        .reshape(n_poses, -1)
    )
    expected = torch.empty_like(deltas)
    for index, mapping in enumerate(docking.torsions):
        changed[nodes[:, mapping.raw_atom], int(mapping.dof_type)] += deltas[:, index]
        expected[:, index] = mapping.sign * deltas[:, index]
    coords = kin(changed)
    for actual, original_value in zip(geometry(coords), original_geometry):
        torch.testing.assert_close(actual, original_value, rtol=0, atol=1e-8)
    response = ld._wrap_angle(
        ld._batched_coord_dihedrals(coords, atom_nodes) - original_chis
    )
    torch.testing.assert_close(response, expected, rtol=0, atol=1e-8)
