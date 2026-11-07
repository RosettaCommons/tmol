"""Explicit molecular bond graphs survive public AtomArray identity mapping."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES, BACKBONE_CASES
from .disulfide_molecules import DISULFIDE_CASES
from .metal_molecules import METAL_CASES

CASES = {
    "noncanonical": DOMAIN_CASES["connected_noncanonical"],
    "cyclic": BACKBONE_CASES["cyclic"],
    "heterochiral_disulfide": DISULFIDE_CASES["ld"],
    "metal": METAL_CASES["zinc_site"],
}


def bonded_array(b, case):
    import biotite.structure as struc

    row = case.blocks(b.parameter_database)[0]
    atoms = struc.AtomArray(sum(len(names) for _, names, _ in row))
    starts = np.cumsum([0] + [len(names) for _, names, _ in row])
    elements = {a.name: a.element for a in b.parameter_database.chemical.atom_types}
    chains = np.repeat(np.arange(len(case.chain_lengths[0])), case.chain_lengths[0])
    for r, (_, names, xyz) in enumerate(row):
        residue = b.pose.block_type(0, r)
        sl = slice(starts[r], starts[r + 1])
        atoms.coord[sl] = xyz
        atoms.atom_name[sl] = names
        atoms.res_name[sl] = residue.io_equiv_class
        atoms.res_id[sl] = r + 1
        atoms.chain_id[sl] = chr(ord("A") + chains[r])
        atoms.element[sl] = [elements[a.atom_type] for a in residue.atoms]
        atoms.hetero[sl] = True
    bonds = []
    for r, port, other, other_port in case.connections()[0]:
        first, second = b.pose.block_type(0, r), b.pose.block_type(0, other)
        a = first.connections[first.connection_to_cidx[port]].atom
        c = second.connections[second.connection_to_cidx[other_port]].atom
        bonds.append(
            (
                starts[r] + first.atom_to_idx[a],
                starts[other] + second.atom_to_idx[c],
                struc.BondType.SINGLE,
            )
        )
    atoms.bonds = struc.BondList(len(atoms), np.asarray(bonds, np.int32))
    return atoms


@pytest.mark.parametrize("name", tuple(CASES))
def test_atom_array_preserves_explicit_bonds_and_atom_identity(backend, name):
    import biotite.structure as struc

    b = backend.molecule(CASES[name], terms=("ref",))
    atoms = bonded_array(b, CASES[name])
    original_bonds = atoms.bonds.as_array().copy()
    moved = atoms.copy()
    moved.coord += [0, 0, 1]
    ensemble = struc.stack([atoms, moved])
    if b.framework == "jax":
        from tmol_jax.io.canonical_ordering import (
            CanonicalOrdering,
            canonical_form_from_pdb,
        )
        from tmol_jax.io.pose_stack_from_biotite import canonical_form_from_biotite

        co = CanonicalOrdering.from_chemdb(b.parameter_database.chemical)
        result = canonical_form_from_biotite(
            ensemble, canonical_ordering=co, missing_density_distance_threshold=0
        )
        field = result.__getitem__
    else:
        from tmol.io._canonical_ordering import CanonicalOrdering
        from tmol.io._pose_stack_from_biotite import canonical_form_from_biotite

        co = CanonicalOrdering.from_chemdb(b.parameter_database.chemical)
        result = canonical_form_from_biotite(
            ensemble, b.device, co=co, missing_density_distance_threshold=0
        )

        def field(key):
            return getattr(result, key)

    expected = []
    if name == "heterochiral_disulfide":
        channel = "disulfides"
        expected = [(p, 0, 1) for p in range(2)]
    elif name == "metal":
        channel = "metal_coordination"
        slot = co.restypes_atom_index_mapping["ASP"]["OD1"]
        expected = [(p, 0, -1, 1, slot) for p in range(2)]
    else:
        channel = "covalent_bonds"
        for p in range(2):
            for r, port, other, other_port in CASES[name].connections()[0]:
                first, second = b.pose.block_type(0, r), b.pose.block_type(0, other)
                a = co.restypes_atom_index_mapping[first.io_equiv_class][
                    first.connections[first.connection_to_cidx[port]].atom
                ]
                c = co.restypes_atom_index_mapping[second.io_equiv_class][
                    second.connections[second.connection_to_cidx[other_port]].atom
                ]
                expected.append(
                    (p, r, a, other, c) if r < other else (p, other, c, r, a)
                )
    np.testing.assert_array_equal(b.numpy(field(channel)), sorted(expected))
    coords = b.numpy(field("coords"))
    for i in range(len(atoms)):
        slot = co.restypes_atom_index_mapping[atoms.res_name[i]][atoms.atom_name[i]]
        np.testing.assert_array_equal(
            coords[:, atoms.res_id[i] - 1, slot], ensemble.coord[:, i]
        )
    np.testing.assert_array_equal(atoms.bonds.as_array(), original_bonds)
    np.testing.assert_array_equal(ensemble.bonds.as_array(), original_bonds)
    if b.framework == "jax":
        from biotite.structure.io.pdb import PDBFile

        pdb = PDBFile()
        pdb.set_structure(atoms)
        assert any(line.startswith("CONECT") for line in pdb.lines)
        parsed = canonical_form_from_pdb(
            "\n".join(pdb.lines), co, missing_density_distance_threshold=0
        )
        np.testing.assert_array_equal(
            parsed[channel], [row for row in sorted(expected) if row[0] == 0]
        )


@pytest.mark.parametrize(
    "name", ("noncanonical", "cyclic", "heterochiral_disulfide", "metal")
)
def test_public_atom_array_context_keeps_chemistry_and_input(backend, name):
    from importlib import import_module

    case = BACKBONE_CASES["hydroxyproline"] if name == "noncanonical" else CASES[name]
    b = backend.molecule(case, terms=())
    atoms = bonded_array(b, case)
    original = atoms.coord.copy()
    bonds = atoms.bonds.as_array().copy()
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    io = import_module(frontend.__name__ + ".io")
    context = io.build_context_from_atom_array(
        atoms, b.device, param_db=b.parameter_database
    )
    assert context.parameter_database is b.parameter_database
    kwargs = dict(
        context=context,
        no_optH=True,
        ligand_hydrogens="input",
        missing_density_distance_threshold=0.0,
    )
    pose = frontend.pose_stack_from_atom_array(atoms, b.device, **kwargs)
    again = frontend.pose_stack_from_atom_array(atoms, b.device, **kwargs)
    np.testing.assert_array_equal(b.numpy(pose.coords), b.numpy(again.coords))
    np.testing.assert_array_equal(atoms.coord, original)
    np.testing.assert_array_equal(atoms.bonds.as_array(), bonds)
    for i, (number, atom) in enumerate(zip(atoms.res_id, atoms.atom_name)):
        # Terminal protonation may replace H with named terminal hydrogens;
        # virtual metal sites are reconstructed from the donor geometry.
        virtual = context.canonical_ordering.restypes_virtual_atoms.get(
            atoms.res_name[i], ()
        )
        if atoms.element[i] in ("H", "D") or atom in virtual:
            continue
        residue = int(number) - 1
        block = pose.block_type(0, residue)
        offset = int(b.numpy(pose.block_coord_offset)[0, residue])
        np.testing.assert_allclose(
            b.numpy(pose.coords)[0, offset + block.atom_to_idx[atom]],
            original[i],
            atol=2e-5,
            rtol=2e-5,
        )
    b.pose = pose
    channels, _ = b.use_beta2016()
    coords = b.coords()
    values, gradient = b.channel_values_and_gradients(coords)
    assert (
        len(channels) == 25
        and np.isfinite(values).all()
        and np.isfinite(gradient).all()
    )
    direction = np.random.default_rng(711).normal(size=coords.shape)
    direction /= np.linalg.norm(direction)
    expected = b.float64_directional_derivative(coords, direction)
    np.testing.assert_allclose(
        np.einsum("cpai,pai->cp", gradient, direction), expected, rtol=2e-3, atol=2e-3
    )


@pytest.mark.parametrize(
    "name,missing,no_optH",
    [
        (name, False, False)
        for name in ("noncanonical", "heterochiral_disulfide", "metal")
    ]
    + [
        (name, True, disabled)
        for name in ("noncanonical", "d_peptide")
        for disabled in (False, True)
    ],
)
def test_public_input_completion_keeps_backbone_and_scores_all_terms(
    backend, name, missing, no_optH
):
    case = (
        BACKBONE_CASES["hydroxyproline"]
        if name == "noncanonical"
        else DOMAIN_CASES[name] if name == "d_peptide" else CASES[name]
    )
    b = backend.molecule(case, terms=())
    atoms = bonded_array(b, case)
    if missing:
        atoms = atoms[
            ((atoms.res_id != 2) | np.isin(atoms.atom_name, ["N", "CA", "C", "O"]))
            & ~np.isin(atoms.element, ["H", "D"])
        ]
    if b.framework == "jax":
        import tmol_jax as frontend
    else:
        import tmol as frontend
    pose, context = frontend.pose_stack_from_atom_array(
        atoms,
        b.device,
        param_db=b.parameter_database,
        no_optH=no_optH,
        return_context=True,
        ligand_hydrogens="input",
        packer_seed=73,
        missing_density_distance_threshold=0.0,
    )
    assert context.parameter_database is b.parameter_database
    for r in range(pose.max_n_blocks):
        block = pose.block_type(0, r)
        start = int(b.numpy(pose.block_coord_offset)[0, r])
        for atom in ("N", "CA", "C"):
            index = np.flatnonzero((atoms.res_id == r + 1) & (atoms.atom_name == atom))
            if len(index):
                np.testing.assert_allclose(
                    b.numpy(pose.coords)[0, start + block.atom_to_idx[atom]],
                    atoms.coord[index[0]],
                    atol=3e-5,
                    rtol=3e-5,
                )
    b.pose = pose
    channels, _ = b.use_beta2016()
    values, gradient = b.channel_values_and_gradients(b.coords())
    assert (
        len(channels) == 25
        and np.isfinite(values).all()
        and np.isfinite(gradient).all()
    )
    direction = np.random.default_rng(73).normal(size=b.coords().shape)
    direction /= np.linalg.norm(direction)
    np.testing.assert_allclose(
        np.einsum("cpai,pai->cp", gradient, direction),
        b.float64_directional_derivative(b.coords(), direction),
        rtol=2e-3,
        atol=2e-3,
    )


@pytest.mark.parametrize("format", ("mol2", "cif"))
def test_public_file_input_preserves_ligand_context(backend, format):
    from importlib import import_module
    from pathlib import Path
    from tmol_core.io import file_from_atom_array, atom_array_from_file

    b = backend.molecule(DOMAIN_CASES["ligand"], terms=())
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    ligand = import_module(frontend.__name__ + ".ligand")
    source = (
        Path(__file__).resolve().parents[2]
        / "tmol/tests/data/ligand_test/ligand_ground_truth/mol2/comt_2.mol2"
    )
    name = str(atom_array_from_file(source).res_name[0])
    info = ligand.nonstandard_residue_info_from_mol2(source, res_name=name)
    db, _ = ligand.prepare_ligand_from_mol2(source, res_name=name, mode="keep")
    text = (
        source
        if format == "mol2"
        else file_from_atom_array(info.atom_array, format=format)
    )
    pose, context = frontend.pose_stack_from_file(
        text,
        b.device,
        format=format,
        param_db=db,
        no_optH=True,
        ligand_hydrogens="input",
        return_context=True,
    )
    assert context.parameter_database is db
    block = pose.block_type(0, 0)
    for i, name in enumerate(info.atom_names):
        if info.elements[i] not in ("H", "D"):
            np.testing.assert_allclose(
                b.numpy(pose.coords)[0, block.atom_to_idx[name]],
                info.coords[i],
                atol=2e-5,
                rtol=2e-5,
            )
    b.pose, b.parameter_database = pose, db
    channels, _ = b.use_beta2016()
    values, gradients = b.channel_values_and_gradients(b.coords())
    assert (
        len(channels) == 25
        and np.isfinite(values).all()
        and np.isfinite(gradients).all()
    )
    assert np.linalg.norm(gradients) > 1e-3


@pytest.mark.parametrize("ccd_h_name", (False, True))
def test_public_input_preserves_supplied_neutral_terminal_hydrogen(backend, ccd_h_name):
    from importlib import import_module
    from pathlib import Path
    from tmol_core.io import atom_array_from_file

    b = backend.molecule(BACKBONE_CASES["canonical"], terms=())
    atoms = atom_array_from_file(
        Path(__file__).resolve().parents[2]
        / "tmol/tests/data/sweep_regressions/stated_terminus_5lh4.pdb"
    )
    atoms = atoms[atoms.res_id == 132]
    if ccd_h_name:
        atoms.atom_name[atoms.atom_name == "H1"] = "H"
    original = atoms.coord.copy()
    frontend = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    pose = frontend.pose_stack_from_atom_array(
        atoms,
        b.device,
        param_db=b.parameter_database,
        no_optH=True,
        ligand_hydrogens="input",
    )
    block = pose.block_type(0, 0)
    assert "nterm_neutral" in block.name and "H3" not in block.atom_to_idx
    for i, name in enumerate(atoms.atom_name):
        if not np.isfinite(original[i]).all():
            continue  # Reader placeholders must be rebuilt, not copied.
        mapped = "H1" if ccd_h_name and name == "H" else name
        np.testing.assert_allclose(
            b.numpy(pose.coords)[0, block.atom_to_idx[mapped]],
            original[i],
            atol=1e-5,
            rtol=1e-5,
        )
    np.testing.assert_array_equal(atoms.coord, original)
    b.pose = pose
    b.use_beta2016()
    values, gradients = b.channel_values_and_gradients(b.coords())
    assert np.isfinite(values).all() and np.isfinite(gradients).all()


@pytest.mark.parametrize("invalid", ("unknown_heavy", "duplicate"))
def test_public_input_rejects_lost_heavy_or_ambiguous_atom_identity(backend, invalid):
    b = backend.molecule(BACKBONE_CASES["canonical"], terms=())
    atoms = bonded_array(b, BACKBONE_CASES["canonical"])
    atoms = atoms[atoms.res_id == 1]
    if invalid == "unknown_heavy":
        atoms.atom_name[atoms.atom_name == "CB"] = "XYZ"
        expected = "Heavy atoms.*XYZ"
    else:
        atoms = atoms + atoms[atoms.atom_name == "CA"]
        expected = "Multiple input atoms map to canonical"
    if b.framework == "jax":
        from tmol_jax.io.pose_stack_from_biotite import canonical_form_from_biotite

        def build():
            return canonical_form_from_biotite(atoms, prepare_input=True)

    else:
        from tmol.io._pose_stack_from_biotite import canonical_form_from_biotite

        def build():
            return canonical_form_from_biotite(atoms, b.device)

    with pytest.raises(ValueError, match=expected):
        build()
