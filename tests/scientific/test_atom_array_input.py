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
