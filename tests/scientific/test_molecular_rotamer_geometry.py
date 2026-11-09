"""Prepared residue-local geometry reconstructs actual ideal molecular atoms."""

import numpy as np
import pytest

from .kinematics import KinematicExecution
from .molecules import DOMAIN_CASES


def _check_geometry(backend, tree, ideal):
    parent = np.concatenate((np.array([0], np.int32), tree.parent + 1))
    kinds = np.concatenate((np.array([0], np.int32), tree.doftype))
    reference = np.concatenate((np.zeros((1, 9), np.float32), tree.dofs_ideal))
    mapping = KinematicExecution(backend, parent, kinds, reference)
    coords = mapping.evaluate(reference[:, :6])[1:]
    np.testing.assert_allclose(coords, ideal[tree.id], rtol=2e-5, atol=2e-5)
    np.testing.assert_array_equal(tree.id[tree.kinforest_idx], np.arange(len(ideal)))


@pytest.mark.parametrize(
    "name",
    ("ligand", "connected_noncanonical", "d_peptide", "modified_dna", "metal_cluster"),
)
def test_rotamer_geometry_preserves_molecular_atom_identity(backend, name):
    backend.molecule(DOMAIN_CASES[name], terms=("ref",))
    pbt = backend.pose.packed_block_types
    if backend.framework == "jax":
        from tmol_jax.pack import prepare_rotamer_kinematics

        packed = prepare_rotamer_kinematics(pbt)
    else:
        from tmol.pack.rotamer._single_residue_kinforest import (
            _construct_single_residue_kinforest,
            _coalesce_single_residue_kinforests,
        )

        for block in pbt.active_block_types:
            _construct_single_residue_kinforest(block)
        _coalesce_single_residue_kinforests(pbt)
        packed = pbt.rotamer_kinforest
    for index, block in enumerate(pbt.active_block_types):
        tree = block.rotamer_kinforest
        count = block.n_atoms
        ideal = block.ideal_coords[block.at_to_icoor_ind]
        _check_geometry(backend, tree, ideal)
        np.testing.assert_array_equal(
            backend.numpy(packed.dofs_ideal[index, :count]), tree.dofs_ideal
        )


def test_connected_rotamer_group_preserves_atoms_and_linkage(backend):
    backend.molecule(DOMAIN_CASES["connected_noncanonical"], terms=("ref",))
    blocks = [backend.pose.block_type(0, index) for index in (0, 1)]
    links = [
        (0, blocks[0].connection_to_cidx["up"], 1, blocks[1].connection_to_cidx["down"])
    ]
    if backend.framework == "jax":
        from tmol_jax.pack import prepare_block_group_kinematics as prepare
    else:
        from tmol.pack.rotamer._single_residue_kinforest import (
            _construct_block_group_kinforest as prepare,
        )
    tree, offsets = prepare(blocks, links)
    np.testing.assert_array_equal(
        offsets, [0, blocks[0].n_atoms, sum(b.n_atoms for b in blocks)]
    )
    ideal = np.concatenate([b.ideal_coords[b.at_to_icoor_ind] for b in blocks])
    _check_geometry(backend, tree, ideal)
    # The one inter-block edge belongs to the physical tree, not a second jump.
    assert np.count_nonzero(tree.doftype == 1) == 1
