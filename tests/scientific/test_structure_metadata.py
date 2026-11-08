"""Source metadata follows physical residue/atom identity through public input."""

import numpy as np
import pytest

from .test_atom_array_input import CASES, bonded_array
from .molecules import BACKBONE_CASES


@pytest.mark.parametrize("name", ("noncanonical", "heterochiral_disulfide", "metal"))
@pytest.mark.parametrize("no_optH", (True, False))
def test_molecular_input_preserves_metadata(backend, name, no_optH):
    case = BACKBONE_CASES["hydroxyproline"] if name == "noncanonical" else CASES[name]
    b = backend.molecule(case, terms=("ref",))
    atoms = bonded_array(b, case)
    atoms.ins_code[:] = "B"
    atoms.res_id[:] += 40
    atoms.set_annotation("occupancy", np.linspace(0, 1, len(atoms), dtype=np.float32))
    atoms.set_annotation("b_factor", np.arange(len(atoms), dtype=np.float32) + 0.5)
    atoms.set_annotation("is_polymer", atoms.res_id != 41)
    atoms.set_annotation("chain_type", np.full(len(atoms), "fixture", object))
    if b.framework == "jax":
        import tmol_jax as frontend
    else:
        import tmol as frontend
    pose = frontend.pose_stack_from_atom_array(
        atoms,
        b.device,
        param_db=b.parameter_database,
        no_optH=no_optH,
        packer_seed=73,
        ligand_hydrogens="input",
        missing_density_distance_threshold=0.0,
    )
    info = pose.pdb_info
    offsets = b.numpy(pose.block_coord_offset)
    for r in range(pose.max_n_blocks):
        block = pose.block_type(0, r)
        label = int(info.residue_labels[0, r])
        assert label == 41 + r
        assert info.residue_insertion_codes[0, r] == "B"
        assert info.residue_annotations[0, r]["is_polymer"] == (label != 41)
        assert info.residue_annotations[0, r]["chain_type"] == "fixture"
        for a, atom in enumerate(block.atoms):
            source = np.flatnonzero(
                (atoms.res_id == label) & (atoms.atom_name == atom.name)
            )
            if len(source) and atoms.element[source[0]] not in ("H", "D"):
                dest = offsets[0, r] + a
                assert info.atom_occupancy[0, dest] == atoms.occupancy[source[0]]
                assert info.atom_b_factor[0, dest] == atoms.b_factor[source[0]]
    selected = info.take([0, 0])
    np.testing.assert_array_equal(
        selected.residue_annotations[1], info.residue_annotations[0]
    )
    selected.atom_occupancy[0, 0] = -2
    assert info.atom_occupancy[0, 0] >= 0
    assert selected.atom_occupancy[1, 0] >= 0

    if backend.framework != "jax" or name != "noncanonical" or not no_optH:
        return
    from .test_canonical_molecular_input import canonical_protein_input

    new_input, build = canonical_protein_input(backend)
    from io import StringIO
    import biotite.structure.io.pdb as pdb
    from tmol_jax.io.write_pose_stack_pdb import write_pose_stack_pdb

    canonical = new_input()
    canonical["res_num"][0] = [-2, 0, 17]
    canonical["res_ins_codes"][0] = ["A", "", "B"]
    canonical["atom_occupancy"][:] = 0.25
    canonical["atom_b_factor"][:] = 42.5
    pose = build(canonical)
    output = StringIO()
    write_pose_stack_pdb(pose, output)
    restored = pdb.PDBFile.read(StringIO(output.getvalue())).get_structure(
        model=1, extra_fields=["occupancy", "b_factor"]
    )
    np.testing.assert_array_equal(np.unique(restored.res_id), [-2, 0, 17])
    np.testing.assert_array_equal(restored.ins_code[restored.res_id == -2], "A")
    np.testing.assert_array_equal(restored.ins_code[restored.res_id == 17], "B")
    np.testing.assert_array_equal(restored.occupancy, 0.25)
    np.testing.assert_array_equal(restored.b_factor, 42.5)
    duplicate = pose[[0, 0]]
    duplicate.pdb_info.atom_occupancy[0, 0] = 0.75
    assert pose.pdb_info.atom_occupancy[0, 0] == 0.25
    assert duplicate.pdb_info.atom_occupancy[1, 0] == 0.25


@pytest.mark.parametrize(
    "filename",
    ("metal_fixtures/zn_tetrahedral_3ks3.cif.zst", "cif/155c__1__1.A__1.B.cif"),
)
def test_canonical_compaction_preserves_metal_origins_and_residue_annotations(
    backend, filename
):
    from pathlib import Path
    from tmol_core.io import atom_array_from_file

    atoms = atom_array_from_file(
        Path(__file__).parents[2] / "tmol/tests/data" / filename
    )
    atoms = atoms[~np.isin(atoms.res_name, ("HOH", "ACE"))]
    b = backend.molecule(CASES["metal"], terms=("ref",))
    if b.framework == "jax":
        from tmol_jax.io.canonical_ordering import (
            default_canonical_ordering,
            default_packed_block_types,
        )
        from tmol_jax.io.pose_stack_from_biotite import canonical_form_from_biotite
        from tmol_jax.io.pose_stack_construction import pose_stack_from_canonical_form

        co = default_canonical_ordering()
        cf = canonical_form_from_biotite(
            atoms, canonical_ordering=co, prepare_input=True
        )
        cf["canonical_ordering"] = co
        values = cf

        def build():
            return pose_stack_from_canonical_form(
                cf, default_packed_block_types(), compact_padding=True
            )

    else:
        from tmol.io._pose_stack_from_biotite import (
            canonical_form_from_biotite,
            _canonical_ordering_for_biotite,
            _packed_block_types_for_biotite_with_metals,
        )
        from tmol.io._pose_stack_construction import _pose_stack_from_canonical_form

        co = _canonical_ordering_for_biotite()
        cf = canonical_form_from_biotite(atoms, b.device, co=co)
        values = {
            "res_num": cf.res_labels,
            "res_types": cf.res_types,
            "covalent_bonds": cf.covalent_bonds,
            "residue_annotations": cf.residue_annotations,
            "metal_origins": cf.metal_origins,
        }

        def build():
            return _pose_stack_from_canonical_form(
                co, _packed_block_types_for_biotite_with_metals(b.device), *cf
            )

    expected = {
        name: np.delete(values[name], 5, axis=1)
        for name in ("residue_annotations", "metal_origins")
        if values.get(name) is not None
    }
    assert expected
    values["res_types"][0, 5] = -1
    bonds = values["covalent_bonds"]
    filtered = bonds[(bonds[:, [1, 3]] != 5).all(1)]
    if b.framework == "jax":
        cf["covalent_bonds"] = filtered
        # Names are redundant canonical inputs and must agree with the excision.
        cf["res_name"][0, 5] = ""
    else:
        cf.covalent_bonds = filtered
    info = build().pdb_info
    for name, annotations in expected.items():
        np.testing.assert_array_equal(
            getattr(info, name)[:, : annotations.shape[1]], annotations
        )
