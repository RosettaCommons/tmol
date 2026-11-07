"""Public molecular queries preserve custom identities and author insertion codes."""

import importlib

import attrs
import numpy as np
import pytest

from .molecules import BACKBONE_CASES


def test_public_pose_queries_keep_noncanonical_ligand_and_author_identity(backend):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    api = importlib.import_module("tmol_jax" if b.framework == "jax" else "tmol")
    pose_api = importlib.import_module(api.__name__ + ".pose")
    pose = b.pose
    info = attrs.evolve(
        pose.pdb_info,
        chain_labels=np.full(pose.block_type_ind.shape, "A"),
        residue_labels=np.array([[52, 52, 53, 100], [11, 12, 13, -1]]),
        residue_insertion_codes=np.array([["", "B", "", ""], ["", "", "", ""]]),
    )
    pose = attrs.evolve(pose, pdb_info=info)
    original = b.numpy(pose.coords).copy()
    assert api.get_sequence(pose, 0) == "AXVX"
    ligand = api.get_ligand_mask(pose)
    assert ligand.device == b.device
    np.testing.assert_array_equal(b.numpy(ligand), [[0, 0, 0, 1], [0, 0, 0, 0]])
    assert api.block_for_residue_label(pose, 0, "A", "52B") == 1
    assert api.block_for_residue_label(pose, 0, "A", "52") == 0
    numbers = np.array([[52, 100], [53, 99]], np.int32)
    if b.framework == "torch":
        numbers = b.module.as_tensor(numbers, device=b.device)
    found = api.blocks_for_residue_labels(pose, 0, "A", numbers)
    assert found.device == b.device
    np.testing.assert_array_equal(b.numpy(found), [[0, 3], [2, -1]])
    atom = api.atom_for_residue_label(pose, 0, "A", "52B", "CA")
    np.testing.assert_array_equal(
        b.numpy(atom), [0, 1, pose.block_type(0, 1).atom_to_idx["CA"]]
    )
    ligand_name = pose.block_type(0, 3).atoms[2].name
    atoms = api.constraint_atoms_for_labels(
        pose, [(0, "A", "52B", "CA"), (0, "A", "100", ligand_name)]
    )
    assert atoms.device == b.device
    np.testing.assert_array_equal(b.numpy(atoms)[0], b.numpy(atom))
    np.testing.assert_array_equal(b.numpy(atoms)[1], [0, 3, 2])
    assert pose_api.residue_label(pose, 0, 1) == "A 52 HYP"
    offset = int(b.numpy(pose.block_coord_offset)[0, 3])
    assert pose_api.atom_label(pose, 0, offset + 2) == f"A 100 LG1 {ligand_name}"
    with pytest.raises(ValueError, match="no block found"):
        api.block_for_residue_label(pose, 0, "A", "52A")
    with pytest.raises(ValueError, match="has no atom"):
        api.atom_for_residue_label(pose, 0, "A", "100", "MISSING")
    np.testing.assert_array_equal(b.numpy(pose.coords), original)
