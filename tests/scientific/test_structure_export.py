"""Molecular export retains physical chemistry and source observations."""

from pathlib import Path
import numpy as np
import pytest

from tmol_core.io import atom_array_from_file
from .test_fragment_input import _frontend, _build
from .fragment_molecules import fragment_fixture


@pytest.mark.parametrize(
    "path",
    (
        "metal_fixtures/zn_tetrahedral_3ks3.cif.zst",
        "atomworks_regressions/plp_enzyme_7mkv.cif",
        "atomworks_regressions/isopeptide_2rm9.cif.zst",
        "atomworks_regressions/repeated_glycans_6mub.cif.zst",
    ),
)
def test_export_reconstructs_real_molecular_topology_and_physics(backend, path):
    frontend = _frontend(backend)
    atoms = atom_array_from_file(Path(__file__).parents[2] / "tmol/tests/data" / path)
    atoms = atoms[atoms.res_name != "HOH"]
    pose, context = frontend.pose_stack_from_atom_array(
        atoms,
        backend.device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )
    exported = frontend.atom_array_from_pose_stack(pose, context.canonical_ordering)
    rebuilt = frontend.pose_stack_from_atom_array(
        exported,
        backend.device,
        context=context,
        no_optH=True,
    )
    np.testing.assert_array_equal(
        backend.numpy(rebuilt.block_type_ind), backend.numpy(pose.block_type_ind)
    )
    np.testing.assert_array_equal(
        backend.numpy(rebuilt.inter_residue_connections),
        backend.numpy(pose.inter_residue_connections),
    )
    np.testing.assert_allclose(
        backend.numpy(rebuilt.coords), backend.numpy(pose.coords), atol=2e-5, rtol=2e-5
    )
    backend.parameter_database = context.parameter_database
    results = []
    for current in (pose, rebuilt):
        backend.pose = current
        backend.use_beta2016()
        results.append(backend.channel_values_and_gradients(backend.coords()))
    for actual, expected in zip(results[1], results[0]):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)


def test_fragment_export_restores_distinct_original_molecules_and_bonds(backend):
    import biotite.structure as struc

    whole, annotated, parameters = fragment_fixture(repeated=True, mixed=False)
    pose, context = _build(backend, annotated, parameters)
    frontend = _frontend(backend)
    merged = frontend.atom_array_from_pose_stack(pose, context.canonical_ordering)
    separate = frontend.atom_array_from_pose_stack(
        pose, context.canonical_ordering, merge_fragments=False
    )
    assert struc.get_residue_count(merged) == 2
    assert struc.get_residue_count(separate) == 4
    assert set(merged.res_name) == {"LG1"}
    assert set(merged.chain_id) == set(whole.chain_id)
    np.testing.assert_array_equal(merged.bonds.as_array(), separate.bonds.as_array())
    for chain in set(whole.chain_id):
        expected = whole[whole.chain_id == chain]
        actual = merged[merged.chain_id == chain]
        by_name = {name: xyz for name, xyz in zip(actual.atom_name, actual.coord)}
        assert by_name.keys() == set(expected.atom_name)
        for atom in expected:
            np.testing.assert_allclose(
                by_name[atom.atom_name], atom.coord, atol=2e-5, rtol=2e-5
            )


def test_export_multimodel_coordinates_and_metadata_contract(backend):
    import attrs

    whole, _, parameters = fragment_fixture(mixed=False)
    pose, context = _build(backend, whole, parameters)
    frontend = _frontend(backend)
    if backend.framework == "jax":
        batch = pose[[0, 0]]
        coords = backend.module.device_put(backend.numpy(batch.coords), backend.device)
        coords = coords.at[1].add(3)
    else:
        batch = frontend.PoseStackBuilder.from_poses([pose, pose], backend.device)
        coords = batch.coords.clone()
        coords[1] += 3
    batch = attrs.evolve(batch, coords=coords)
    exported = frontend.atom_array_from_pose_stack(batch, context.canonical_ordering)
    np.testing.assert_allclose(
        exported.coord[1] - exported.coord[0], 3, atol=5e-6, rtol=0
    )
    # One metadata record cannot represent different per-model observations.
    for field in ("atom_b_factor", "residue_annotations", "metal_origins"):
        saved = getattr(batch.pdb_info, field)
        if field == "atom_b_factor":
            changed = saved.copy()
            changed[1, 0] += 1
        elif field == "residue_annotations":
            changed = np.zeros(batch.block_type_ind.shape, dtype=[("hetero", bool)])
            changed[1, 0]["hetero"] = True
        else:
            changed = np.full(batch.block_type_ind.shape, None, object)
            changed[1, 0] = (8, "", "LIG", "ZN")
        setattr(batch.pdb_info, field, changed)
        with pytest.raises(ValueError, match="different metadata"):
            frontend.atom_array_from_pose_stack(batch, context.canonical_ordering)
        setattr(batch.pdb_info, field, saved)


def test_export_preserves_coordination_and_explicit_virtual_atom_option(backend):
    import biotite.structure as struc

    frontend = _frontend(backend)
    path = (
        Path(__file__).parents[2]
        / "tmol/tests/data/metal_fixtures/zn_tetrahedral_3ks3.cif.zst"
    )
    atoms = atom_array_from_file(path)
    # This protein-only input omits the file's two water coordination partners.
    atoms = atoms[atoms.res_name != "HOH"]
    pose, context = frontend.pose_stack_from_atom_array(
        atoms,
        backend.device,
        prepare_ligands=True,
        no_optH=True,
        return_context=True,
    )
    ordinary = frontend.atom_array_from_pose_stack(pose, context.canonical_ordering)
    virtual = frontend.atom_array_from_pose_stack(
        pose, context.canonical_ordering, include_virtual_atoms=True
    )
    assert ordinary.atom_name[ordinary.res_name == "ZN"].tolist() == ["ZN"]
    assert virtual.atom_name[virtual.res_name == "ZN"].tolist() == [
        "ZN",
        "V1",
        "V2",
        "V3",
        "V4",
    ]
    for output in (ordinary, virtual):
        bonds = output.bonds.as_array()
        metal = np.flatnonzero((output.res_name == "ZN") & (output.atom_name == "ZN"))[
            0
        ]
        coordination = bonds[
            (bonds[:, :2] == metal).any(axis=1)
            & (bonds[:, 2] == struc.BondType.COORDINATION)
        ]
        donors = {
            (int(output.res_id[i]), str(output.res_name[i]), str(output.atom_name[i]))
            for bond in coordination
            for i in bond[:2]
            if i != metal
        }
        assert donors == {(94, "HIS", "NE2"), (96, "HIS", "NE2"), (119, "HIS", "ND1")}

    if backend.framework != "jax":
        return
    from io import StringIO
    from .molecules import DOMAIN_CASES
    from tmol_jax.io.write_pose_stack_pdb import (
        atom_records_from_pose_stack,
        write_pose_stack_pdb,
    )

    jax = backend.module
    for case_name, metal in (("metal_mixed", "Zn"), ("metal_cluster", "Fe")):
        backend = backend.molecule(DOMAIN_CASES[case_name], terms=("ref",))
        with jax.enable_x64():
            coords = np.asarray(backend.pose.coords, dtype=np.float64).copy()
            coords[0, 0, 0] = 1.123456789123
            backend.pose = backend.pose.replace(
                coords=jax.device_put(coords, backend.device)
            )
            records = atom_records_from_pose_stack(backend.pose)
            assert records[0]["x"] == coords[0, 0, 0]
        metal_rows = [
            row for row in records if row["element"].strip().lower() == metal.lower()
        ]
        assert metal_rows
        stream = StringIO()
        write_pose_stack_pdb(backend.pose, stream)
        text = stream.getvalue()
        lines = [
            line for line in text.splitlines() if line.startswith(("ATOM  ", "HETATM"))
        ]
        assert len(lines) == len(records)
        for line, record in zip(lines, records):
            assert line[12:16].strip() == record["atomn"]
            assert line[76:78].strip().lower() == record["element"].strip().lower()
