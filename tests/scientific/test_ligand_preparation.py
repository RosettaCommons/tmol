"""Public generated and supplied ligands retain chemistry through scoring."""

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize(
    "source", ("l_alanine", "d_alanine", "aspirin", "carboxylate", "mol2")
)
def test_public_ligand_preparation_scoring_and_minimization(
    backend, source, conformer_execution=None
):
    frontend = "tmol_jax" if backend.framework == "jax" else "tmol"
    ligand = import_module(frontend + ".ligand")
    database_type = import_module(frontend + ".database").ParameterDatabase

    # A subtype supplied by a caller must survive injection with its custom
    # scoring tables and original database untouched.
    class CustomDatabase(database_type):
        pass

    default = database_type.get_default()
    base = CustomDatabase(chemical=default.chemical, scoring=default.scoring)
    if source == "mol2":
        path = Path(__file__).resolve().parents[2] / (
            "tmol/tests/data/ligand_test/ligand_ground_truth/mol2/comt_2.mol2"
        )
        info = ligand.nonstandard_residue_info_from_mol2(path, res_name="LG1")
        database, ordering = ligand.prepare_ligand_from_mol2(
            path, res_name="LG1", param_db=base, mode="keep"
        )
    else:
        smiles = {
            "l_alanine": "[NH3+][C@H](C)C(=O)[O-]",
            "d_alanine": "[NH3+][C@@H](C)C(=O)[O-]",
            "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
            "carboxylate": "CC(=O)O",
        }[source]
        options = dict(res_name="LG1", protonate=source == "carboxylate", seed=47)
        if conformer_execution is not None:
            options["conformer_execution"] = conformer_execution
        info = ligand.nonstandard_residue_info_from_smiles_via_mol2(smiles, **options)
        database, ordering = ligand.prepare_ligand_from_smiles(
            smiles, param_db=base, **options
        )
    if source in ("l_alanine", "d_alanine"):
        from rdkit import Chem
        from tmol_core.ligand._rdkit_mol import _ligand_atom_array_to_rdkit_mol

        molecule = _ligand_atom_array_to_rdkit_mol(info, keep_hydrogens=True)
        Chem.AssignAtomChiralTagsFromStructure(molecule, replaceExistingTags=True)
        assert Chem.MolToSmiles(Chem.RemoveHs(molecule)) == Chem.MolToSmiles(
            Chem.MolFromSmiles(smiles)
        )
    assert type(database) is CustomDatabase
    assert database.scoring.hbond is base.scoring.hbond
    assert all(res.name != "LG1" for res in base.chemical.residues)
    assert "LG1" in ordering.restype_io_equiv_classes
    charges = {
        p.atom: p.charge
        for p in database.scoring.elec.atom_charge_parameters
        if p.res == "LG1"
    }
    heavy_names = {
        name for name, element in zip(info.atom_names, info.elements) if element != "H"
    }
    assert {name: charges[name] for name in heavy_names} == {
        name: info.partial_charges[name] for name in heavy_names
    }
    if source == "mol2":
        assert charges == info.partial_charges
    else:
        # The generated MOL2 uses generic H names; mature preparation names
        # hydrogens by their parent while retaining their indexed MMFF charges.
        np.testing.assert_allclose(
            sum(charges.values()), -1.0 if source == "carboxylate" else 0.0, atol=1e-7
        )
    assert max(map(abs, charges.values())) > 0.05
    residue = next(res for res in database.chemical.residues if res.name == "LG1")
    names = tuple(atom.name for atom in residue.atoms)
    if source == "mol2":
        indices = {name: i for i, name in enumerate(info.atom_names)}
        xyz = np.asarray(info.coords, np.float32)[[indices[name] for name in names]]
    else:
        from tmol_core.chemical.restypes import refined_from_raw

        xyz = refined_from_raw(residue).ideal_coords.astype(np.float32)
    assert set(charges) == set(names)

    case = SimpleNamespace(
        sequences=(("LG1",),),
        chain_lengths=((1,),),
        blocks=lambda db: [[("LG1", names, xyz)]],
        extra_connections=lambda: [[]],
        connections=lambda: [[]],
    )
    backend.molecule(case, terms=(), parameter_database=database)
    channels, weights = backend.use_beta2016()
    assert len(channels) == 25
    coordinates = backend.coords()
    np.testing.assert_array_equal(coordinates[0, : len(xyz)], xyz)
    # Move away from the optimized geometry so derivatives and bonded energies
    # are exercised even for a supplied conformer already near equilibrium.
    displaced = coordinates + np.random.default_rng(120).normal(
        0, 0.015, coordinates.shape
    ).astype(np.float32)
    values, gradients = backend.channel_values_and_gradients(displaced)
    assert np.isfinite(values).all() and np.isfinite(gradients).all()
    assert np.linalg.norm(gradients[channels.index("cart_lengths")]) > 0.1
    direction = np.random.default_rng(19).normal(size=displaced.shape)
    direction /= np.linalg.norm(direction)
    finite_difference = backend.float64_directional_derivative(displaced, direction)
    np.testing.assert_allclose(
        np.einsum("cpai,pai->cp", gradients, direction),
        finite_difference,
        rtol=2e-3,
        atol=2e-3,
    )
    before = backend.scores(coordinates)
    minimized = backend.minimize(np.ones(coordinates.shape[:2], bool), iterations=3)
    after = backend.scores(minimized)
    assert np.all(after <= before + 1e-4 * np.maximum(1, np.abs(before)))


@pytest.mark.parametrize("fourth_dimension", (None, 5.0))
def test_refinement_gradient_and_rigid_motion(backend, fourth_dimension):
    from contextlib import nullcontext
    from tmol_core.ligand import refinement

    frontend = "tmol_jax" if backend.framework == "jax" else "tmol"
    runtime = import_module(frontend + ".ligand._conformer_generation")
    module = backend.module
    context = module.enable_x64() if backend.framework == "jax" else nullcontext()
    with context:

        def index(value):
            if backend.framework == "jax":
                return module.device_put(np.asarray(value, np.int32), backend.device)
            return module.as_tensor(value, dtype=module.int64, device=backend.device)

        def floating(value):
            if backend.framework == "jax":
                return module.device_put(np.asarray(value, np.float64), backend.device)
            return module.as_tensor(value, dtype=module.float64, device=backend.device)

        initial_lower = np.zeros((5, 5))
        initial_upper = np.full((5, 5), 10.0)
        initial_lower[0, 1] = initial_upper[0, 1] = 1.5
        lower = np.full((5, 5), 1.3)
        upper = np.full((5, 5), 2.0)
        parameters = refinement.prepare_refinement(
            initial_lower,
            initial_upper,
            lower,
            upper,
            [(0, 1, 2, 3)],
            [(0, 1, 2, 4, 1)],
            index_array=index,
            float_array=floating,
        )
        coordinates = np.random.default_rng(291).normal(
            size=(5, 3 if fourth_dimension is None else 4)
        )

        def objective(x):
            return refinement.refinement_loss(
                x, parameters, runtime._OPERATIONS, fourth_dimension=fourth_dimension
            )

        x = floating(coordinates)
        if backend.framework == "jax":
            evaluate = module.jit(module.value_and_grad(objective))
            value, gradient = evaluate(x)
        else:
            x.requires_grad_()
            value = objective(x)
            gradient = module.autograd.grad(value, x)[0]
        value, gradient = backend.numpy(value), backend.numpy(gradient)
        assert value > 1 and np.isfinite(gradient).all()
        direction = np.random.default_rng(29).normal(size=coordinates.shape)
        step = 1e-5
        difference = (
            backend.numpy(objective(floating(coordinates + step * direction)))
            - backend.numpy(objective(floating(coordinates - step * direction)))
        ) / (2 * step)
        np.testing.assert_allclose(
            np.sum(gradient * direction), difference, rtol=2e-7, atol=2e-7
        )
        np.testing.assert_allclose(gradient[:, :3].sum(0), 0, atol=1e-10)
        np.testing.assert_allclose(
            np.cross(coordinates[:, :3], gradient[:, :3]).sum(0), 0, atol=1e-10
        )
        shifted = coordinates.copy()
        shifted[:, :3] += [9, -7, 2]
        np.testing.assert_allclose(
            backend.numpy(objective(floating(shifted))), value, rtol=2e-13
        )
