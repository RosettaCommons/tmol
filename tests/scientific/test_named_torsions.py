"""Named physical torsions retain molecular selection and periodic geometry."""

import importlib
import numpy as np
import pytest

from .molecules import BACKBONE_CASES


def signed_angle(a, b, c, d):
    # Independent plane-normal oracle in double precision.
    axis = c - b
    axis /= np.linalg.norm(axis)
    left, right = np.cross(b - a, axis), np.cross(axis, d - c)
    return np.arctan2(np.dot(np.cross(left, right), axis), np.dot(left, right))


@pytest.mark.parametrize("case", ("canonical", "d_peptide", "hydroxyproline", "mixed"))
def test_named_polymer_torsions_and_selection(backend, case):
    backend.molecule(BACKBONE_CASES[case], terms=())
    api = importlib.import_module(
        "tmol.pose" if backend.framework == "torch" else "tmol_jax.pose"
    )
    pose = backend.pose
    xyz = backend.coords().astype(np.float64)
    offsets = backend.numpy(pose.block_coord_offset)

    def atom(block, name):
        bt = pose.block_type(0, block)
        return xyz[0, int(offsets[0, block]) + bt.atom_to_idx[name]]

    expected = (
        signed_angle(atom(0, "C"), atom(1, "N"), atom(1, "CA"), atom(1, "C")),
        signed_angle(atom(1, "N"), atom(1, "CA"), atom(1, "C"), atom(2, "N")),
    )
    actual = tuple(
        api.get_named_torsions(pose, 0, 1, name, degrees=False)
        for name in ("phi", "psi")
    )
    np.testing.assert_allclose(np.sin(actual), np.sin(expected), atol=2e-6)
    np.testing.assert_allclose(np.cos(actual), np.cos(expected), atol=2e-6)
    values = api.get_named_torsions(pose, 0, 1)
    assert list(values) == importlib.import_module(
        "tmol" if backend.framework == "torch" else "tmol_jax"
    ).get_torsion_names(pose, 0, 1)
    np.testing.assert_allclose(
        [values["phi"], values["psi"]], np.degrees(actual), atol=2e-5
    )
    paired = api.get_named_torsions(pose, [0, 0], [1, 1], ["phi", "psi"], degrees=False)
    assert set(paired[0][1]) == {"phi", "psi"}
    assert all(not values for i, values in enumerate(paired[0]) if i != 1)
    assert np.isnan(api.get_named_torsions(pose, 0, 0, "phi"))
    with pytest.raises(ValueError, match="same length"):
        api.get_named_torsions(pose, [0], [0, 1], "phi")
