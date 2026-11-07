"""Native torsion updates preserve molecular identities, geometry and snapshots."""

from importlib import import_module

import numpy as np
import pytest

from .molecules import BACKBONE_CASES, DOMAIN_CASES


@pytest.mark.parametrize(
    "case,block,name",
    [
        (BACKBONE_CASES["mixed"], 2, "chi1"),
        (BACKBONE_CASES["d_peptide"], 2, "chi1"),
        (DOMAIN_CASES["connected_noncanonical"], 1, "theta1"),
        (DOMAIN_CASES["dna"], 1, "beta"),
        (DOMAIN_CASES["rna"], 1, "beta"),
    ],
    ids=["mixed", "d-peptide", "beta-peptide", "dna", "rna"],
)
def test_named_torsion_target_and_bond_geometry(backend, case, block, name):
    b = backend.molecule(case, terms=())
    api = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    kin = import_module(api.__name__ + ".kinematics")
    query = import_module(api.__name__ + ".pose")
    original = b.numpy(b.pose.coords).copy()
    moved = kin.set_named_torsions(b.pose, 0, block, name, 61.5)
    angle = query.get_named_torsions(moved, 0, block, name)
    assert (angle - 61.5 + 180) % 360 - 180 == pytest.approx(0, abs=1e-3)
    xyz = b.numpy(moved.coords)
    assert np.linalg.norm(xyz[0] - original[0]) > 1e-3
    assert moved.coords.dtype == b.pose.coords.dtype
    assert moved.coords.device == b.pose.coords.device
    assert moved.packed_block_types is b.pose.packed_block_types
    np.testing.assert_array_equal(b.numpy(b.pose.coords), original)
    np.testing.assert_array_equal(
        b.numpy(moved.inter_residue_connections),
        b.numpy(b.pose.inter_residue_connections),
    )
    # Bond lengths are independent invariants of a torsion-only operation.
    for pose, row in enumerate(case.sequences):
        for residue in range(len(row)):
            rt = b.pose.block_type(pose, residue)
            offset = int(b.numpy(b.pose.block_coord_offset)[pose, residue])
            edges = (
                np.asarray(
                    [[rt.atom_to_idx[a], rt.atom_to_idx[c]] for a, c, *_ in rt.bonds],
                    int,
                )
                + offset
            )
            before = original[pose, edges[:, 0]] - original[pose, edges[:, 1]]
            after = xyz[pose, edges[:, 0]] - xyz[pose, edges[:, 1]]
            np.testing.assert_allclose(
                np.linalg.norm(after, axis=1),
                np.linalg.norm(before, axis=1),
                rtol=2e-5,
                atol=2e-5,
            )
    if b.pose.n_poses > 1:
        np.testing.assert_allclose(xyz[1:], original[1:], atol=2e-5, rtol=2e-5)
        ligand_offset = int(b.numpy(b.pose.block_coord_offset)[0, 3])
        np.testing.assert_allclose(
            xyz[0, ligand_offset:], original[0, ligand_offset:], atol=2e-5, rtol=2e-5
        )


def test_named_torsion_units_duplicate_requests_and_errors(backend):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    api = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    kin = import_module(api.__name__ + ".kinematics")
    query = import_module(api.__name__ + ".pose")
    current = query.get_named_torsions(b.pose, 0, 2, "chi1", degrees=False)
    moved = kin.set_named_torsions(
        b.pose, 0, 2, ["chi1", "chi1"], [current + 0.1, current + 0.2], degrees=False
    )
    angle = query.get_named_torsions(moved, 0, 2, "chi1", degrees=False)
    assert (angle - current - 0.3 + np.pi) % (2 * np.pi) - np.pi == pytest.approx(
        0, abs=2e-5
    )
    with pytest.raises(ValueError, match="no torsion"):
        kin.set_named_torsions(b.pose, 0, 2, "not_a_torsion", 0)
    with pytest.raises(ValueError, match="undefined"):
        kin.set_named_torsions(b.pose, 0, 0, "phi", 0)
    with pytest.raises(ValueError, match="values must"):
        kin.set_named_torsions(b.pose, 0, 0, "phi", [0, 1])
