"""Both frontends display the same real molecular data through common rendering."""

from importlib import import_module

import numpy as np
import pytest

from tmol_core.io.visualize import _biotite_to_pdb_string
from .molecules import BACKBONE_CASES


def test_notebook_helpers_export_real_pose_and_keep_atom_selection(backend):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    front = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    pose = b.pose.split(0)
    atoms = front.atom_array_from_pose_stack(pose)
    pdb = _biotite_to_pdb_string(atoms)
    before = b.numpy(pose.coords).copy()
    mask = np.arange(len(atoms)) % 3 == 0
    viewer = front.view(pose, highlighted=mask, show_hover=False)
    assert hasattr(viewer, "addStyle")
    assert pdb.replace("\n", r"\n") in viewer._make_html()
    with pytest.raises(ValueError, match="length"):
        front.view(pose, highlighted=mask[:-1])
    switch = front.switchable_view(
        {"pose": pose, "atoms": atoms}, notes={"pose": "<label>"}
    )
    overlay = front.overlay_view({"pose": pose, "atoms": atoms})
    gallery = front.selection_gallery(atoms, {"selected": mask})
    assert "&lt;label&gt;" in switch.data
    assert "$3Dmol" in overlay.data
    assert '"serials": ' + str((np.flatnonzero(mask) + 1).tolist()) in gallery.data
    np.testing.assert_array_equal(b.numpy(pose.coords), before)
