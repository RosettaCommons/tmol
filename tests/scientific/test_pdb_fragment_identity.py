"""PDB output retains the original identities of repeated fragmented ligands."""

import attrs
import numpy as np
import pytest
from biotite.structure import get_residue_count
from biotite.structure.io.pdb import PDBFile

from .fragment_molecules import fragment_fixture
from .test_fragment_input import _build, _frontend


@pytest.mark.parametrize("merge_fragments", (True, False))
def test_pdb_fragment_output_keeps_each_original_ligand(
    backend, tmp_path, merge_fragments
):
    whole, annotated, parameters = fragment_fixture(repeated=True, mixed=False)
    pose, _ = _build(backend, annotated, parameters)
    translated = attrs.evolve(pose, coords=pose.coords + 3)
    pose = _frontend(backend).PoseStackBuilder.from_poses(
        [pose, translated], backend.device
    )
    path = tmp_path / "ligands.pdb"
    options = {} if merge_fragments else {"merge_fragments": False}
    if backend.framework == "jax":
        from tmol_jax.io import write_pose_stack_pdb

        write_pose_stack_pdb(pose, path, pose_index=None, **options)
    else:
        from tmol.io._write_pose_stack_pdb import _write_pose_stack_pdb

        _write_pose_stack_pdb(pose, str(path), **options)
    models = PDBFile.read(path).get_structure()
    assert models.stack_depth() == 2
    for model, exported in enumerate(models):
        assert get_residue_count(exported) == (2 if merge_fragments else 4)
        assert set(exported.chain_id) == set(whole.chain_id)
        if merge_fragments:
            assert set(exported.res_name) == {"LG1"}
            for chain in set(whole.chain_id):
                actual = exported[exported.chain_id == chain]
                expected = whole[whole.chain_id == chain]
                by_name = {atom.atom_name: atom.coord for atom in actual}
                assert by_name.keys() == set(expected.atom_name)
                assert len(by_name) == len(actual)
                for atom in expected:
                    # PDB fixed-width coordinates retain three decimal places.
                    np.testing.assert_allclose(
                        by_name[atom.atom_name],
                        atom.coord + 3 * model,
                        rtol=0,
                        atol=5.1e-4,
                    )
                np.testing.assert_array_equal(actual.res_id, expected.res_id)
                np.testing.assert_array_equal(actual.ins_code, expected.ins_code)
