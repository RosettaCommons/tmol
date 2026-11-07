"""Shared canonical topology preserves molecular boundaries and stated termini."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES


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
