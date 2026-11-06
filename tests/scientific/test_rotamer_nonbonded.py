"""Sparse conformer values conserve pose energy and arbitrary weighted forces."""

from contextlib import nullcontext

import numpy as np
import pytest

from .molecules import DOMAIN_CASES
from .rotamer_execution import RotamerExecution


@pytest.mark.parametrize(
    "case_name",
    (
        "ligand",
        "connected_noncanonical",
        "d_peptide",
        "modified_dna",
        "metal_mixed",
        "mixed_batch",
    ),
)
@pytest.mark.parametrize("electrostatic", (False, True))
def test_sparse_nonbonded_values_and_coordinate_derivatives(
    backend, case_name, electrostatic
):
    terms = ("fa_elec",) if electrostatic else ("fa_ljatr", "fa_ljrep", "fa_lk")
    backend.molecule(DOMAIN_CASES[case_name], terms=terms)
    precision = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with precision:
        execution = RotamerExecution(backend, electrostatic=electrostatic)
        coords = backend.numpy(execution.rotamers.coords).astype(np.float64)
        values, indices = map(backend.numpy, execution.scores(coords))
        assert np.isfinite(values).all() and np.any(np.abs(values) > 1e-3)
        whole = backend.scores(
            backend.coords().astype(np.float64), sum_terms=False, apply_weights=False
        )
        aggregated = np.zeros_like(whole)
        for entry, pose in enumerate(indices[0]):
            aggregated[:, pose] += values[:, entry]
        np.testing.assert_allclose(aggregated, whole, rtol=2e-12, atol=2e-10)
        weights = np.cos(np.arange(values.size).reshape(values.shape) * 0.21) + 0.4
        scored, gradient, scored_indices = execution.value_and_grad(coords, weights)
        np.testing.assert_array_equal(scored_indices, indices)
        np.testing.assert_allclose(scored, values, rtol=0, atol=0)
        direction = np.random.default_rng(103).normal(size=coords.shape)
        direction /= np.linalg.norm(direction)
        analytic = np.sum(gradient * direction)
        for step in (1e-5, 2e-6):
            plus, plus_indices = map(
                backend.numpy, execution.scores(coords + step * direction)
            )
            minus, minus_indices = map(
                backend.numpy, execution.scores(coords - step * direction)
            )
            np.testing.assert_array_equal(plus_indices, indices)
            np.testing.assert_array_equal(minus_indices, indices)
            numerical = np.sum((plus - minus) * weights) / (2 * step)
            np.testing.assert_allclose(analytic, numerical, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(gradient.sum(axis=0), 0, atol=2e-10)
        np.testing.assert_array_equal(backend.numpy(execution.rotamers.coords), coords)


def test_sparse_custom_parameters_preserve_existing_scorer(backend):
    import attrs

    backend.molecule(DOMAIN_CASES["ligand"], terms=("ref",))
    precision = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with precision:
        original = RotamerExecution(backend)
        coords = backend.numpy(original.rotamers.coords).astype(np.float64)
        first, indices = map(backend.numpy, original.scores(coords))
        weights = np.broadcast_to(
            np.asarray([1.0, 1.0, 0.0])[:, None], first.shape
        ).copy()
        _, gradient, _ = original.value_and_grad(coords, weights)
        database = backend.parameter_database
        # Fourfold well depth doubles its float32 square-root table exactly;
        # this keeps the analytic scaling check free of parameter quantization.
        parameters = tuple(
            attrs.evolve(p, lj_wdepth=4 * p.lj_wdepth)
            for p in database.scoring.ljlk.atom_type_parameters
        )
        ljlk = attrs.evolve(database.scoring.ljlk, atom_type_parameters=parameters)
        modified = attrs.evolve(
            database, scoring=attrs.evolve(database.scoring, ljlk=ljlk)
        )
        backend.molecule(
            DOMAIN_CASES["ligand"], terms=("ref",), parameter_database=modified
        )
        changed = RotamerExecution(backend)
        actual, actual_gradient, actual_indices = changed.value_and_grad(
            coords, weights
        )
        np.testing.assert_array_equal(actual_indices, indices)
        np.testing.assert_allclose(actual[:2], 4 * first[:2], rtol=2e-12, atol=2e-11)
        np.testing.assert_allclose(actual[2], first[2], rtol=2e-12, atol=2e-11)
        np.testing.assert_allclose(
            actual_gradient, 4 * gradient, rtol=2e-12, atol=2e-11
        )
        np.testing.assert_array_equal(backend.numpy(original.scores(coords)[0]), first)


def test_sparse_dunbrack_preserves_library_free_conformer_identity(backend):
    backend.molecule(DOMAIN_CASES["ligand"], terms=("ref",))
    e = RotamerExecution(backend, family="dunbrack")
    coords = backend.numpy(e.rotamers.coords)
    values, indices = map(backend.numpy, e.scores(coords))
    np.testing.assert_array_equal(values, 0)
    np.testing.assert_array_equal(indices[0], backend.numpy(e.rotamers.pose_for_rot))
    np.testing.assert_array_equal(indices[1], np.arange(e.rotamers.n_rotamers_total))
    np.testing.assert_array_equal(indices[2], indices[1])
    _, gradient, _ = e.value_and_grad(coords, np.ones_like(values))
    np.testing.assert_array_equal(gradient, np.zeros_like(coords))


@pytest.mark.parametrize("case_name", ("l_peptide", "mixed_batch"))
@pytest.mark.parametrize("family", ("ljlk", "elec", "dunbrack"))
def test_sampled_ensemble_selection_conserves_energy_and_forces(
    backend, case_name, family
):
    from .dunbrack_molecules import DUNBRACK_CASES

    terms = {
        "elec": ("fa_elec",),
        "ljlk": ("fa_ljatr", "fa_ljrep", "fa_lk"),
        "dunbrack": ("dunbrack_rot", "dunbrack_rotdev", "dunbrack_semirot"),
    }[family]
    backend.molecule(DUNBRACK_CASES[case_name], terms=terms)
    precision = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with precision:
        e = RotamerExecution(backend, family=family, sampled=True)
        r = e.rotamers
        xyz = backend.numpy(r.coords).astype(np.float64)
        counts = backend.numpy(r.n_rots_for_block)
        starts = backend.numpy(r.rot_offset_for_block)
        assert counts.max() > 2
        values, indices = map(backend.numpy, e.scores(xyz))
        selected = np.zeros(r.n_rotamers_total, bool)
        reconstructed = backend.coords().astype(np.float64)
        offsets = backend.numpy(r.coord_offset_for_rot)
        types = backend.numpy(r.block_type_ind_for_rot)
        for pose, block in np.argwhere(counts > 0):
            rot = starts[pose, block] + (pose + block + 1) % counts[pose, block]
            selected[rot] = True
            n = e.packed_block_types.active_block_types[types[rot]].n_atoms
            dest = backend.numpy(backend.pose.block_coord_offset)[pose, block]
            reconstructed[pose, dest : dest + n] = xyz[offsets[rot] : offsets[rot] + n]
        chosen = selected[indices[1]] & selected[indices[2]]
        aggregated = np.zeros((values.shape[0], len(counts)))
        for entry in np.flatnonzero(chosen):
            aggregated[:, indices[0, entry]] += values[:, entry]
        np.testing.assert_allclose(
            aggregated,
            backend.scores(reconstructed, sum_terms=False, apply_weights=False),
            rtol=2e-12,
            atol=2e-9,
        )
        weights = np.cos(np.arange(values.size).reshape(values.shape) * 0.21) + 0.4
        _, gradient, _ = e.value_and_grad(xyz, weights)
        direction = np.random.default_rng(203).normal(size=xyz.shape)
        direction /= np.linalg.norm(direction)
        for step in (1e-5, 2e-6):
            plus = backend.numpy(e.scores(xyz + step * direction)[0])
            minus = backend.numpy(e.scores(xyz - step * direction)[0])
            np.testing.assert_allclose(
                (gradient * direction).sum(),
                ((plus - minus) * weights).sum() / (2 * step),
                rtol=3e-6,
                atol=3e-6,
            )
