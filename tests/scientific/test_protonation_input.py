"""Hydrogen evidence and required donors select actual L/D chemical forms."""

import numpy as np
import pytest


def test_native_protonation_evidence_and_required_donors(backend):
    from tmol_core.chemical import DEPROTONATED_VAR_IND, _special_case_variant_index
    from tmol_core.io.canonical_ordering import CanonicalOrdering
    from tmol_core.io.protonation import prepare_protonation_plan
    from .molecules import MolecularCase

    b = backend.molecule(MolecularCase((), (("GLY",),), ((1,),)), terms=("ref",))
    db = b.parameter_database.chemical
    co = CanonicalOrdering.from_chemdb(db)
    # Heavy-only coordination, hydrogen evidence, and a preset disulfide.
    cases = (
        ("HIS", "NE2", (), "HIS_D", 0),
        ("HIS", "ND1", (), "HIS", 0),
        ("CYS", "SG", (), "CYS_DEP", 0),
        ("DCYS", "SG", (), "DCYS_DEP", 0),
        ("TYR", "OH", (), "TYR_DEP", 0),
        ("DTYR", "OH", (), "DTYR_DEP", 0),
        ("SER", "OG", (), "SER", 0),
        ("CYS", None, ("H", "HA", "HB2", "HB3"), "CYS_DEP", 0),
        ("CYS", None, ("H", "HA", "HB2", "HB3", "HG"), "CYS", 0),
        ("CYD", None, (), "CYD", 1),
    )
    raw = {r.name: r for r in db.residues}
    classes = [raw[name].io_equiv_class for name, *_ in cases]
    types = np.array(
        [[co.restype_io_equiv_classes.index(c) for c in classes] + [-1]], np.int32
    )
    variants = np.array([[row[-1] for row in cases] + [7]], np.int32)
    expected = [[_special_case_variant_index(raw[row[3]]) for row in cases] + [7]]
    elements = {a.name: a.element for a in db.atom_types}
    present = np.zeros((*types.shape, co.max_n_canonical_atoms), bool)
    donors = []
    for r, ((name, donor, hydrogens, _, _), cls) in enumerate(zip(cases, classes)):
        index = co.restypes_atom_index_mapping[cls]
        for atom in raw[name].atoms:
            if elements[atom.atom_type] != "H":
                present[0, r, index[atom.name]] = True
        for atom in hydrogens:
            present[0, r, index[atom]] = True
        if donor:
            donors.append((0, r, index[donor]))
    donors = np.asarray(donors, np.int64)
    bonds = np.empty((0, 5), np.int64)
    plan = prepare_protonation_plan(co, db, types)
    if b.framework == "jax":
        from tmol_jax.io._native import canonical_protonation
        from tmol_jax.io._selection import _wide

        args = b.module.device_put(
            (types, variants, present, plan.arrays, _wide(bonds), _wide(donors)),
            b.device,
        )
        result, unresolved = b.module.jit(canonical_protonation)(*args)
        mapped = b.module.jit(b.module.vmap(canonical_protonation))(
            *b.module.tree.map(lambda x: b.module.numpy.stack((x, x)), args)
        )
        for original, batched in zip((result, unresolved), mapped):
            np.testing.assert_array_equal(b.numpy(batched[0]), b.numpy(original))
        np.testing.assert_array_equal(b.numpy(args[1]), variants)
    else:
        from tmol.io.details.compiled._compiled import _ops

        inputs = [
            b.module.as_tensor(a, device=b.device)
            for a in (types, variants, present, *plan.arrays, bonds, donors)
        ]
        result, unresolved = _ops.protonation(inputs, DEPROTONATED_VAR_IND)
        np.testing.assert_array_equal(b.numpy(inputs[1]), variants)
    np.testing.assert_array_equal(b.numpy(result), expected)
    assert not np.any(b.numpy(unresolved))


@pytest.mark.parametrize("name", ("CYS_DEP", "DTYR_DEP"))
def test_explicit_protonation_retains_heavy_atoms_without_hydrogen_evidence(
    backend, name
):
    from tmol_core.chemical import _special_case_variant_index
    from .molecules import MolecularCase
    from .test_canonical_molecular_input import canonical_input, rebuild_canonical

    case = MolecularCase((), ((name,),), ((1,),))
    b = backend.molecule(case, terms=("cart_lengths", "fa_elec"))
    expected = b.coords().copy()
    residue = b.pose.block_type(0, 0)
    elements = {a.name: a.element for a in b.parameter_database.chemical.atom_types}
    heavy = np.array([elements[a.atom_type] != "H" for a in residue.atoms])
    co, cf = canonical_input(case, b.parameter_database)
    index = co.restypes_atom_index_mapping[residue.io_equiv_class]
    for atom in residue.atoms:
        if elements[atom.atom_type] == "H":
            cf["coords"][0, 0, index[atom.name]] = np.nan
    cf["protonation_variants"] = np.full_like(
        cf["res_types"], _special_case_variant_index(residue)
    )
    rebuild_canonical(b, co, cf)
    assert b.pose.block_type(0, 0).name == name
    np.testing.assert_array_equal(b.coords()[0, heavy], expected[0, heavy])
    energy, gradient = b.channel_values_and_gradients(b.coords())
    assert np.isfinite(energy).all() and np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0


def test_supplied_thiol_hydrogen_precedes_provisional_metal_request(backend):
    from .molecules import MolecularCase
    from .test_canonical_molecular_input import canonical_input, rebuild_canonical

    case = MolecularCase((), (("ZN_tetrahedral", "CYS"),), ((1, 1),))
    b = backend.molecule(case, terms=("metal_coordination",))
    co, cf = canonical_input(case, b.parameter_database)
    cf["metal_sites"] = np.empty((0, 3), np.int32)
    cf["metal_coordination"] = np.array(
        [[0, 0, -1, 2, co.restypes_atom_index_mapping["CYS"]["SG"]]], np.int32
    )
    rebuild_canonical(b, co, cf, unpatched_metals=True)
    assert b.pose.block_type(0, 1).base_name == "CYS"
    assert "HG" in b.pose.block_type(0, 1).atom_to_idx
    assert not np.any(b.numpy(b.pose.inter_residue_connections)[..., 0] >= 0)
    with pytest.raises(ValueError, match="metal coordination could not be resolved"):
        rebuild_canonical(
            b, co, dict(cf, require_all_metal_donors=True), unpatched_metals=True
        )
