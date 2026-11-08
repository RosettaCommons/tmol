"""Molecular Cartesian objectives decrease without changing frozen atoms."""

import numpy as np
import pytest

from .molecules import CASES, DOMAIN_CASES
from .dunbrack_molecules import DUNBRACK_CASES

CART = (
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
)
NONBONDED = ("fa_ljatr", "fa_ljrep", "fa_lk", "fa_elec")
CASES_AND_TERMS = {
    "mixed_genbonded": (CASES["mixed_batch"], ("gen_torsions",)),
    "ligand": (CASES["ligand"], CART + NONBONDED + ("gen_torsions",)),
    "beta_amino_acids": (
        CASES["connected_noncanonical"],
        CART + NONBONDED + ("gen_torsions",),
    ),
    "d_peptide": (
        DUNBRACK_CASES["d_peptide"],
        CART + NONBONDED + ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
    ),
}


@pytest.mark.parametrize("name", CASES_AND_TERMS)
def test_molecular_minimization_preserves_fixed_atoms_and_input(backend, name):
    case, terms = CASES_AND_TERMS[name]
    backend.molecule(case, terms=terms)
    _check_frozen_atom_minimization(backend, native_case=name == "mixed_genbonded")


def _check_frozen_atom_minimization(backend, native_case=False):
    initial = backend.coords().copy()
    mask = np.ones(initial.shape[:2], dtype=bool)
    if native_case:
        mask = backend.numpy(
            backend.pose.atom_is_real
            if backend.framework == "jax"
            else backend.pose.real_atoms
        ).copy()
    mask[:, : 3 if native_case else 1] = False
    before = backend.scores(initial).sum()
    if native_case and backend.framework == "jax":
        from tmol_jax import run_cart_min

        with backend.module.default_device(backend.device):
            minimized_pose, result = run_cart_min(
                backend.pose,
                backend.score_function,
                move_mask=mask,
                max_iter=15,
                return_result=True,
            )
        minimized = backend.numpy(minimized_pose.coords)
        assert float(result["fun"]) < result["initial_fun"] - 1e-3
    else:
        minimized = backend.minimize(mask, iterations=15 if native_case else 5)
    after = backend.scores(minimized).sum()
    assert np.isfinite(minimized).all() and np.isfinite(after)
    assert after < before - 1e-3
    assert np.linalg.norm((minimized - initial)[mask]) > (1e-3 if native_case else 1e-4)
    np.testing.assert_array_equal(minimized[~mask], initial[~mask])
    np.testing.assert_array_equal(backend.coords(), initial)


@pytest.mark.parametrize(
    "name",
    (
        "ligand",
        "connected_noncanonical",
        "d_peptide",
        "modified_dna",
        "metal_mixed",
        "mixed_batch",
    ),
)
def test_complete_preset_minimization_preserves_fixed_atoms(backend, name):
    backend.molecule(DOMAIN_CASES[name], terms=())
    backend.use_beta2016()
    _check_frozen_atom_minimization(backend)
