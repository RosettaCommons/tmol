"""Array execution boundaries of the shared ligand refinement objective."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.ligand import refinement
from tmol_jax.ligand._conformer_generation import _OPERATIONS


def test_distance_floor_tie_keeps_mature_derivative():
    with jax.enable_x64():

        def objective(x):
            return refinement.planar_loss(
                x.reshape((2, 3)),
                jnp.array([0]),
                jnp.array([1]),
                jnp.array([0.0]),
                _OPERATIONS,
            )

        x = jnp.array([0.0, 0.0, 0.0, 1e-6, 0.0, 0.0])
        gradient = jax.jit(jax.grad(objective))(x)
        np.testing.assert_allclose(gradient, [-2e-6, 0, 0, 2e-6, 0, 0], atol=1e-18)
        # Below the floor the distance is constant; repeated atom gathers must
        # not introduce a spurious force on a coincident pair.
        np.testing.assert_array_equal(jax.grad(objective)(x * 0.5), np.zeros(6))


def test_planar_objective_batches_coordinates_without_mixing_gradients():
    with jax.enable_x64():
        first, second = jnp.array([0, 1, 0]), jnp.array([1, 2, 2])
        target = jnp.array([1.0, 1.0, np.sqrt(2)])

        def objective(x):
            return refinement.planar_loss(x, first, second, target, _OPERATIONS)

        x = jnp.array([[0.0, 0.0], [1.1, 0.0], [1.0, 1.2]])
        batch = jnp.stack([x, x + 7.0, 1.3 * x])
        evaluate = jax.jit(jax.vmap(jax.value_and_grad(objective)))
        values, gradients = evaluate(batch)
        np.testing.assert_allclose(values[:2], values[0], atol=1e-14)
        np.testing.assert_allclose(
            gradients[:2], jnp.stack([gradients[0]] * 2), atol=1e-14
        )
        np.testing.assert_allclose(gradients.sum(axis=1), 0, atol=1e-14)
        single_value, single_gradient = jax.value_and_grad(objective)(batch[2])
        np.testing.assert_allclose(values[2], single_value, atol=1e-14)
        np.testing.assert_allclose(gradients[2], single_gradient, atol=1e-14)


def test_seeded_ligand_context_is_independent_of_scoring_device():
    from tmol_core.ligand._conformer_generation import _seeded_conformer
    from tmol_jax.ligand import (
        nonstandard_residue_info_from_smiles_via_mol2,
        prepare_ligand_from_smiles,
    )

    results = []
    try:
        for device in (jax.devices("cpu")[0], jax.devices()[0]):
            # Exercise fresh numerical preparation in both contexts; a seeded
            # conformer cache must not conceal dependence on the first device.
            _seeded_conformer.cache_clear()
            with jax.default_device(device):
                options = dict(res_name="LGP", protonate=False, seed=39)
                smiles = "CC(=O)Oc1ccccc1C(=O)O"
                info = nonstandard_residue_info_from_smiles_via_mol2(smiles, **options)
                database, _ = prepare_ligand_from_smiles(smiles, **options)
                residue = next(r for r in database.chemical.residues if r.name == "LGP")
                charges = tuple(
                    p
                    for p in database.scoring.elec.atom_charge_parameters
                    if p.res == "LGP"
                )
            results.append((np.array(info.coords), residue, charges))
        np.testing.assert_array_equal(results[0][0], results[1][0])
        assert results[0][1] == results[1][1]
        assert results[0][2] == results[1][2]
    finally:
        _seeded_conformer.cache_clear()
