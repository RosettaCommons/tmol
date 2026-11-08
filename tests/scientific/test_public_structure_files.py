"""Public file helpers preserve custom molecular identity and declared bonds."""

import numpy as np
import pytest

from .fragment_molecules import fragment_fixture
from .test_fragment_input import _build, _frontend


@pytest.mark.parametrize("format", ("pdb", "cif"))
def test_public_structure_files_keep_custom_ligand_atoms_and_physics(
    backend, format, tmp_path
):
    frontend = _frontend(backend)
    whole, _, parameters = fragment_fixture(mixed=True)
    # PDB readers normalize blank chain labels; give every source chain an
    # explicit representable identity before requiring an exact round trip.
    whole.chain_id[whole.chain_id == ""] = "A"
    pose, context = _build(backend, whole, parameters)
    path = tmp_path / f"mixed.{format}"
    assert frontend.file_from_pose_stack(pose, path) == path
    atoms = frontend.atom_array_from_file(path)
    text = frontend.file_from_atom_array(atoms, format=format)
    assert isinstance(text, str) and len(text) > 100
    exported = frontend.atom_array_from_pose_stack(pose)
    for field in ("atom_name", "res_name", "res_id", "chain_id", "element"):
        np.testing.assert_array_equal(getattr(atoms, field), getattr(exported, field))
    np.testing.assert_allclose(atoms.coord, exported.coord, rtol=0, atol=5.1e-4)
    rebuilt = frontend.pose_stack_from_file(
        path, backend.device, context=context, no_optH=True
    )
    np.testing.assert_array_equal(
        backend.numpy(rebuilt.block_type_ind), backend.numpy(pose.block_type_ind)
    )
    np.testing.assert_array_equal(
        backend.numpy(rebuilt.inter_residue_connections),
        backend.numpy(pose.inter_residue_connections),
    )
    backend.pose, backend.parameter_database = rebuilt, context.parameter_database
    channels, _ = backend.use_beta2016()
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    gen = channels.index("gen_torsions")
    assert np.linalg.norm(values[gen]) > 1e-3
    assert np.linalg.norm(gradients[gen]) > 1e-3

    if backend.framework != "jax" or format != "pdb":
        return
    from tmol_jax.io import canonical_form_from_pdb
    from .test_canonical_molecular_input import CANONICAL_PROTEIN_PDB

    lines = CANONICAL_PROTEIN_PDB.read_text().splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("ATOM"))
    lines[first] = lines[first][:12] + " ZZ " + lines[first][16:]
    with pytest.raises(ValueError, match="atom ZZ"):
        canonical_form_from_pdb("\n".join(lines))

    from tmol_jax import pose_stack_from_pdb
    from .test_canonical_molecular_input import peptide_pdb

    protein_path = tmp_path / "tripeptide.pdb"
    protein_path.write_text(peptide_pdb())
    from tmol_jax.io.pdb_parsing import parse_pdb

    def atom(serial, x):
        return (
            f"ATOM  {serial:5d}  CA  ALA A   1    "
            f"{x:8.3f}{2.0:8.3f}{3.0:8.3f}  1.00 10.00           C  "
        )

    text = "\n".join(
        [
            "MODEL        1",
            atom(1, 1.0),
            "ENDMDL",
            "MODEL        2",
            atom(1, 9.0),
            "ENDMDL",
            "END",
        ]
    )
    records = parse_pdb(text)
    np.testing.assert_array_equal(records["coords"], [[1.0, 2.0, 3.0]])
    with backend.module.default_device(backend.device):
        from tmol_jax.io import write_pose_stack_pdb

        pose = pose_stack_from_pdb(protein_path)
        out = tmp_path / "written.pdb"
        write_pose_stack_pdb(pose, out)
        again = pose_stack_from_pdb(out)
        assert again.coords.shape == pose.coords.shape
        np.testing.assert_allclose(again.coords, pose.coords, atol=1e-3)
        lines = protein_path.read_text().splitlines()
        shifted = [
            (
                line[:30] + f"{float(line[30:38]) + 6:8.3f}" + line[38:]
                if line.startswith("ATOM") and int(line[22:26]) == 3
                else line
            )
            for line in lines
        ]
        path = tmp_path / "gap.pdb"
        path.write_text("\n".join(shifted) + "\n")
        pose = pose_stack_from_pdb(path)
        names = [pose.block_type(0, i).name for i in range(3)]
        assert not any("cterm" in name for name in names[:2]), names
        connections = np.asarray(pose.inter_residue_connections)[0]
        assert set(connections[1, :, 0]) == {0, -1}
        assert set(connections[2, :, 0]) == {-1}
        import attrs
        from biotite.structure.io.pdb import PDBFile
        from tmol_jax.io import CanonicalOrdering, default_packed_block_types
        from tmol_jax.io.pose_stack_from_biotite import pose_stack_from_biotite

        path = protein_path
        expected = pose_stack_from_pdb(path)
        pbt = default_packed_block_types()
        # A legitimate chemical database may order atom records differently without
        # changing identities, connectivity or geometry. Canonical slots then differ.
        reordered = attrs.evolve(
            pbt.chem_db,
            residues=tuple(
                attrs.evolve(residue, atoms=tuple(reversed(residue.atoms)))
                for residue in pbt.chem_db.residues
            ),
        )
        ordering = CanonicalOrdering.from_chemdb(reordered)
        atoms = PDBFile.read(path).get_structure(model=1)
        for source, loader in (
            (path, pose_stack_from_pdb),
            (atoms, pose_stack_from_biotite),
        ):
            actual = loader(source, canonical_ordering=ordering, packed_block_types=pbt)
            assert [r.name for r in actual.packed_block_types.active_block_types] == [
                r.name for r in expected.packed_block_types.active_block_types
            ]
            np.testing.assert_array_equal(actual.coords, expected.coords)
            np.testing.assert_array_equal(
                actual.inter_residue_connections, expected.inter_residue_connections
            )
        np.testing.assert_array_equal(pose_stack_from_pdb(path).coords, expected.coords)
