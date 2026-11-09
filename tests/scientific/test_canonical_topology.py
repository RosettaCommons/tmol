"""Shared canonical topology preserves molecular boundaries and stated termini."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES


def _variant_names(pose):
    return [pose.block_type(0, index).name for index in range(pose.max_n_blocks)]


def topology_context(b, case):
    from tmol_core.chemical.restypes import refined_from_raw
    from tmol_core.io.canonical import prepare_canonical_selection
    from tmol_core.io.canonical_ordering import CanonicalOrdering

    b.molecule(DOMAIN_CASES[case], terms=("ref",))
    chemistry = b.parameter_database.chemical
    co = CanonicalOrdering.from_chemdb(chemistry)
    types = tuple(refined_from_raw(r) for r in chemistry.residues)
    ann = prepare_canonical_selection(co, types, max(r.n_atoms for r in types))

    def array(value):
        return (
            b.module.device_put(np.asarray(value), b.device)
            if b.framework == "jax"
            else b.module.as_tensor(value, device=b.device)
        )

    def run(operation, values):
        values = tuple(map(array, values))
        if b.framework == "jax":
            from tmol_jax.io._native import (
                canonical_chain_endings,
                canonical_atom_termini,
            )

            function = (
                canonical_chain_endings
                if operation == "chain"
                else lambda *a: canonical_atom_termini(*a[:6], a[6:])
            )
            result = b.module.jit(function)(*values)
            batched = b.module.jit(b.module.vmap(function))(
                *(b.module.numpy.stack((v, v)) for v in values)
            )
            for actual, mapped in zip(
                b.module.tree.leaves(result), b.module.tree.leaves(batched)
            ):
                np.testing.assert_array_equal(b.numpy(actual), b.numpy(mapped[0]))
            return b.module.tree.map(b.numpy, result)
        from tmol.io.details.compiled._compiled import (
            _canonical_chain_endings,
            _canonical_atom_termini,
        )

        result = (
            _canonical_chain_endings(list(values))
            if operation == "chain"
            else _canonical_atom_termini(list(values))
        )
        return (
            tuple(map(b.numpy, result))
            if isinstance(result, tuple)
            else b.numpy(result)
        )

    return co, types, ann, run


@pytest.mark.parametrize("case", ("mixed_batch", "d_peptide", "modified_dna"))
def test_chain_breaks_cycles_and_explicit_ports(backend, case):
    co, types, ann, run = topology_context(backend, case)
    original = backend.pose.block_type(0, 0)
    polymer = co.restype_io_equiv_classes.index(original.io_equiv_class)
    assert ann.co_equiv_class_is_polymeric[polymer]
    # A disconnected metal between polymers cannot inherit a polymer bond even
    # if the source file gives every residue the same chain label.
    metal = co.restype_io_equiv_classes.index("ZN")
    assert not ann.co_equiv_class_is_polymeric[metal]
    kinds = np.array(
        [[polymer, polymer, metal, polymer], [polymer, polymer, -1, -1]], np.int32
    )
    chains = np.array([[0, 0, 0, 0], [0, 0, -1, -1]], np.int32)
    real = kinds >= 0
    broken = np.zeros((*kinds.shape, 2), bool)
    explicit = np.zeros_like(broken)
    cycles = np.array([[1, 1, 0]], np.int32)
    values = [
        chains,
        kinds,
        broken,
        real,
        ann.co_equiv_class_is_polymeric,
        ann.co_polymer_connection_atoms.astype(np.int32),
        cycles,
        explicit,
    ]
    termini, first, last, polymeric = run("chain", values)
    np.testing.assert_array_equal(termini, [[0, 2, 1, 3], [1, 1, 1, 1]])
    assert first[1, 0] and last[1, 1] and not polymeric[0, 2]
    np.testing.assert_array_equal(polymeric[~real], False)
    # An explicit external bond consumes its port and prevents an inferred link.
    explicit[0, 0, 1] = True
    termini, first, last, _ = run("chain", values)
    assert last[0, 0] and first[0, 1]
    np.testing.assert_array_equal(termini[0], [0, 3, 1, 3])

    if backend.framework != "jax":
        return
    if case == "mixed_batch":
        from .test_canonical_molecular_input import canonical_protein_input

        # These public JAX defaults intentionally terminate a break/interior gap.
        # Torch canonical compaction has a different adjacency policy.
        new_input, build = canonical_protein_input(backend)
        canonical = new_input()
        canonical["res_not_connected"] = np.zeros((1, 3, 2), dtype=bool)
        canonical["res_not_connected"][0, 0, 1] = True
        pose = build(canonical)
        names = _variant_names(pose)
        assert "nterm" in names[0] and "cterm" in names[0]
        assert "nterm" in names[1] and "cterm" not in names[1]
        assert np.all(np.asarray(pose.inter_residue_connections)[0, 0] == -1)
        assert 0 not in np.asarray(pose.inter_residue_connections)[0, 1, :, 0]
        assert np.isfinite(np.asarray(pose.coords)).all()
        canonical = new_input()
        canonical["res_name"][0, 1] = ""
        canonical["res_types"][0, 1] = -1
        canonical["chain_id"][0, 1] = -1
        canonical["coords"][0, 1] = np.nan
        canonical["atom_is_present"][0, 1] = False
        pose = build(canonical)
        assert pose.max_n_blocks == 2
        assert all("nterm" in name and "cterm" in name for name in _variant_names(pose))
        assert np.all(np.asarray(pose.inter_residue_connections) == -1)
        assert np.isfinite(np.asarray(pose.coords)).all()
        return
    names = ("DA", "DT") if case == "modified_dna" else ("DALA", "DPHE")
    import biotite.structure as struc
    from tmol_jax.io.canonical_ordering import default_canonical_ordering
    from tmol_jax.io.pose_stack_from_biotite import canonical_form_from_biotite

    from tmol_jax.chemical.patched_chemdb import default_patched_chemdb

    raw = {r.name: r for r in default_patched_chemdb().residues}
    classes = tuple(raw[name].io_equiv_class for name in names)
    co = default_canonical_ordering()
    kinds = [co.restype_io_equiv_classes.index(c) for c in classes]
    ports = co.polymer_conn_inds
    atoms = struc.AtomArray(4)
    atoms.chain_id[:] = "A"
    atoms.res_id[:] = [1, 1, 2, 2]
    atoms.res_name[:] = np.repeat(classes, 2)
    for r, cls in enumerate(kinds):
        names = co.restypes_ordered_atom_names[classes[r]]
        atoms.atom_name[2 * r : 2 * r + 2] = [
            names[ports.down_atom_for_co_restype[cls]],
            names[ports.up_atom_for_co_restype[cls]],
        ]
    atoms.coord[:] = [[30, 0, 0], [0, 0, 0], [1.6, 0, 0], [40, 0, 0]]
    close = canonical_form_from_biotite(atoms)
    assert not close["res_gap"].any()
    distant = atoms.copy()
    distant.coord[2, 0] = 8
    broken = canonical_form_from_biotite(distant)
    np.testing.assert_array_equal(broken["res_gap"], [[[False, True], [True, False]]])
    assert not canonical_form_from_biotite(
        distant, missing_density_distance_threshold=0
    )["res_gap"].any()
    ensemble = canonical_form_from_biotite(struc.stack([distant, atoms]))
    assert not ensemble["res_gap"].any()
    distant.coord[2] = np.nan
    assert not canonical_form_from_biotite(distant)["res_gap"].any()
    atoms.bonds = struc.BondList(4, np.array([[0, 1, struc.BondType.SINGLE]]))
    assert canonical_form_from_biotite(atoms)["covalent_bonds"].shape == (0, 5)


@pytest.mark.parametrize("case", ("d_peptide", "modified_dna"))
def test_supplied_terminal_atoms_select_terminal_chemistry(backend, case):
    from tmol_core.io.canonical import _assign_var_inds_for_bt

    co, types, ann, run = topology_context(backend, case)
    equiv = backend.pose.block_type(0, 0).io_equiv_class
    options = [
        r
        for r in types
        if r.io_equiv_class == equiv and _assign_var_inds_for_bt(co, r)[:2] == (3, 0)
    ]
    assert options, f"{equiv} must exercise actual terminal chemistry"
    residue = options[0]
    present = np.zeros((1, 1, co.max_n_canonical_atoms), bool)
    mapping = co.restypes_atom_index_mapping[equiv]
    for atom in residue.atoms:
        present[0, 0, mapping[atom.name]] = True
    kinds = np.array([[co.restype_io_equiv_classes.index(equiv)]], np.int32)
    variant = np.zeros((1, 1), np.int32)
    original = np.ones((1, 1), np.int32)
    broken = np.ones((1, 1, 2), bool)
    values = [
        present,
        kinds,
        variant,
        original,
        broken,
        np.ones((1, 1), bool),
        ann.var_combo_candidate_bt_index.astype(np.int32),
        ann.var_combo_is_real_candidate,
        ann.bt_canonical_atom_is_absent,
    ]
    np.testing.assert_array_equal(run("termini", values), [[3]])
    # Without a gap the explicit chain policy remains authoritative.
    broken[:] = False
    np.testing.assert_array_equal(run("termini", values), original)
