"""Canonical selection retains actual molecular identities and padding."""

import numpy as np
import pytest

from .molecules import DOMAIN_CASES
from .test_leaf_reconstruction import LEAF_CASES


@pytest.mark.parametrize("name", (*LEAF_CASES, "metal_cluster", "rna"))
def test_canonical_candidates_preserve_molecular_identity(backend, name):
    from tmol_core.io.canonical import (
        prepare_canonical_selection,
        _assign_var_inds_for_bt,
    )
    from tmol_core.io.canonical_ordering import CanonicalOrdering

    b = backend.molecule(DOMAIN_CASES[name], terms=("ref",))
    pbt = b.pose.packed_block_types
    co = CanonicalOrdering.from_chemdb(pbt.chem_db)
    prepared = prepare_canonical_selection(co, pbt.active_block_types, pbt.max_n_atoms)
    types = b.numpy(b.pose.block_type_ind).astype(np.int32)
    # Add a padding row and an explicitly invalid class for the native status
    # contract. The public loader must reject the latter, never coerce its type.
    shape = (types.shape[0], types.shape[1] + 2)
    real = np.zeros(shape, bool)
    io = np.full(shape, -1, np.int32)
    term, variant = np.zeros(shape, np.int32), np.zeros(shape, np.int32)
    present = np.zeros((*shape, co.max_n_canonical_atoms), bool)
    expected = np.full(shape, -1, np.int32)
    for pose, residue in np.argwhere(types >= 0):
        t = types[pose, residue]
        bt = pbt.active_block_types[t]
        real[pose, residue] = True
        io[pose, residue] = co.restype_io_equiv_classes.index(bt.io_equiv_class)
        term[pose, residue], variant[pose, residue], _ = _assign_var_inds_for_bt(co, bt)
        names = co.restypes_atom_index_mapping[bt.io_equiv_class]
        for atom in bt.atoms:
            present[pose, residue, names[atom.name]] = True
        expected[pose, residue] = t
    real[:, -1] = True
    io[:, -1] = len(co.restype_io_equiv_classes)
    metadata = (
        prepared.var_combo_candidate_bt_index.astype(np.int32),
        prepared.var_combo_is_real_candidate,
        prepared.bt_canonical_atom_is_absent,
        prepared.bt_non_term_patch_added_canonical_atom_is_present,
        prepared.bt_is_non_default_terminus,
    )

    def array(value):
        value = np.asarray(value)
        return (
            b.module.device_put(value, b.device)
            if b.framework == "jax"
            else b.module.as_tensor(value, device=b.device)
        )

    inputs = tuple(map(array, (present, real, io, term, variant, *metadata)))
    if b.framework == "jax":
        from tmol_jax.io._native import canonical_candidates

        def run(*args):
            return canonical_candidates(*args[:5], args[5:])

        result = b.module.jit(run)(*inputs)
        mapped = b.module.jit(b.module.vmap(run))(
            *(b.module.numpy.stack((v, v)) for v in inputs)
        )
        for original, batched in zip(result, mapped):
            np.testing.assert_array_equal(b.numpy(original), b.numpy(batched[0]))
        selected, score, candidate = map(b.numpy, result)
        score = score.copy().view(np.int64).reshape(shape)
    else:
        from tmol.io.details.compiled._compiled import _select_canonical_candidates

        selected, score, candidate = map(
            b.numpy, _select_canonical_candidates(list(inputs))
        )
    np.testing.assert_array_equal(selected, expected)
    np.testing.assert_array_equal(score[~real], 0)
    np.testing.assert_array_equal(score[:, -1], 1 << 30)
    np.testing.assert_array_equal(candidate[:, -2:], 0)
    assert np.all(score[:, :-2][types >= 0] < 2 * (co.max_n_canonical_atoms + 1))
