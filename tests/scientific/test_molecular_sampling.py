"""Public sampling preserves molecular choice identity and task restrictions."""

import numpy as np
import pytest
import attrs

from .dunbrack_molecules import DUNBRACK_CASES
from .molecules import CASES, DOMAIN_CASES, MolecularCase


def sampling_task(backend):
    if backend.framework == "jax":
        from tmol_jax.pack import PackerPalette, PackerTask, SetPackerTask
        from tmol_jax.pack import DunbrackChiSampler

        sampler = DunbrackChiSampler.from_database(backend.parameter_database)
    else:
        from tmol import PackerPalette, PackerTask
        from tmol.pack import (
            SetPackerTask,
            create_dunbrack_sampler_from_database,
        )

        sampler = create_dunbrack_sampler_from_database(
            backend.parameter_database, backend.device
        )
    task = PackerTask(backend.pose, PackerPalette())
    task.add_conformer_sampler(sampler)
    task.restrict_to_repacking()
    return task, sampler, SetPackerTask.from_packer_task


def block_mask(backend, value):
    if backend.framework == "jax":
        return backend.module.device_put(np.asarray(value, dtype=bool), backend.device)
    return backend.module.tensor(
        value, dtype=backend.module.bool, device=backend.device
    )


@pytest.mark.parametrize("case_name", DUNBRACK_CASES)
def test_public_sampling_counts_atoms_and_monotone_masks(backend, case_name):
    backend.molecule(DUNBRACK_CASES[case_name], terms=("ref",))
    task, sampler, concrete = sampling_task(backend)
    fixed = concrete(task)
    result = sampler.sample_chi_for_poses(backend.pose, fixed)
    counts, choices, atoms, chi = map(backend.numpy, result)
    assert all(value.device == backend.device for value in result)
    assert chi.dtype == backend.coords().dtype
    assert len(choices) == counts.sum() > 0
    np.testing.assert_array_equal(choices, np.repeat(np.arange(len(counts)), counts))
    assert atoms.shape == chi.shape and np.isfinite(chi).all()
    considered = backend.numpy(fixed.cons_bt_block_type)
    allowed = backend.numpy(fixed.is_cons_bt_allowed)
    blocks = backend.pose.packed_block_types.active_block_types
    for row, type_index in enumerate(considered):
        block = blocks[type_index]
        if not allowed[row] or not sampler.defines_rotamers_for_rt(block):
            assert counts[row] == 0
        if counts[row]:
            selected = atoms[choices == row]
            assert np.all(
                (selected == -1) | ((selected >= 0) & (selected < block.n_atoms))
            )
            assert np.any(selected >= 0)
    # The concrete task is a snapshot; disabling all sites produces no samples
    # without changing identities or the previous task's allowed rows.
    task.disable_packing_by_block_mask(
        block_mask(backend, np.ones(backend.numpy(task.is_real_block).shape, bool))
    )
    empty = sampler.sample_chi_for_poses(backend.pose, concrete(task))
    assert backend.numpy(empty[0]).shape == counts.shape
    assert not backend.numpy(empty[0]).any() and len(empty[1]) == 0
    np.testing.assert_array_equal(backend.numpy(fixed.is_cons_bt_allowed), allowed)


def test_public_sampler_mirrors_disconnected_l_d_libraries_and_retains_ligand(backend):
    case = MolecularCase(
        CASES["ligand"].parameter_files,
        (("ARG", "DARG", "LG1"),),
        ((1, 1, 1),),
    )
    backend.molecule(case, terms=("ref",))
    task, sampler, concrete = sampling_task(backend)
    fixed = concrete(task)
    counts, choices, _, chi = map(
        backend.numpy, sampler.sample_chi_for_poses(backend.pose, fixed)
    )
    types = backend.numpy(fixed.cons_bt_block_type)
    blocks = backend.pose.packed_block_types.active_block_types
    rows = {
        blocks[types[i]].name: i
        for i in np.flatnonzero(backend.numpy(fixed.is_cons_bt_allowed))
    }
    assert counts[rows["ARG"]] == counts[rows["DARG"]] > 1
    assert counts[rows["LG1"]] == 0
    np.testing.assert_allclose(
        chi[choices == rows["ARG"]], -chi[choices == rows["DARG"]], atol=2e-6, rtol=0
    )
    before = backend.coords().copy()
    # A required library cannot be reduced to fit a too-small task budget.
    task.set_chi_sample_budget(1, 1)
    with pytest.raises((ValueError, RuntimeError), match="budget|limit|rotamer"):
        sampler.sample_chi_for_poses(backend.pose, concrete(task))
    np.testing.assert_array_equal(backend.coords(), before)


@pytest.mark.parametrize("gapped", (False, True))
def test_custom_libraryless_chi_preserves_parameter_context_and_unsampled_atoms(
    backend, gapped
):
    from tmol_core.chemical import ChiSamples

    backend.molecule(MolecularCase((), (("ILE",),), ((1,),)), terms=("ref",))
    original_db = backend.parameter_database
    ile = next(rt for rt in original_db.chemical.residues if rt.name == "ILE")
    case = MolecularCase((), (("XIL",), ("XIL",)), ((1,), (1,)))
    for angles in ((-60.0, 60.0, 180.0), (-45.0, 45.0)):
        custom = attrs.evolve(
            ile,
            name="XIL",
            base_name="XIL",
            name3="XIL",
            io_equiv_class="XIL",
            dunbrack_reference=None,
            chi_samples=(ChiSamples("chi2" if gapped else "chi1", angles, (), False),),
        )
        database = attrs.evolve(
            original_db,
            chemical=attrs.evolve(
                original_db.chemical, residues=original_db.chemical.residues + (custom,)
            ),
        )
        backend.molecule(case, terms=("ref",), parameter_database=database)
        task, sampler, concrete = sampling_task(backend)
        fixed = concrete(task)
        counts, choices, atoms, chi = map(
            backend.numpy, sampler.sample_chi_for_poses(backend.pose, fixed)
        )
        np.testing.assert_array_equal(counts, [len(angles)] * 2)
        assert atoms.shape == chi.shape
        for row in range(2):
            sampled = chi[choices == row]
            np.testing.assert_allclose(
                sampled[:, int(gapped)], np.deg2rad(angles), atol=2e-7, rtol=0
            )
            if gapped:
                # An unspecified torsion is left to coordinate reconstruction;
                # it has no defining atom and is not a frozen sampled torsion.
                np.testing.assert_array_equal(atoms[choices == row, 0], -1)

        # Changing a custom database must change reconstructed coordinates,
        # preserving each requested torsion and the untouched upstream chi.
        from .test_conformer_coordinates import coordinate_dihedral

        if backend.framework == "jax":
            from tmol_jax.pack.rotamers import build_rotamers
        else:
            from tmol.pack import build_rotamers
        _, rotamers = build_rotamers(backend.pose, fixed, database.chemical)
        xyz = backend.numpy(rotamers.coords)
        offsets = backend.numpy(rotamers.coord_offset_for_rot)
        block = backend.pose.packed_block_types.active_block_types[0]
        for i, row in enumerate(choices):
            indices = np.asarray(block.torsion_to_uaids[f"chi{2 if gapped else 1}"])[
                :, 0
            ]
            measured = coordinate_dihedral(xyz[offsets[i] + indices])
            error = measured - chi[i, int(gapped)]
            assert abs(np.arctan2(np.sin(error), np.cos(error))) < 2e-5
            if gapped:
                indices = np.asarray(block.torsion_to_uaids["chi1"])[:, 0]
                measured = coordinate_dihedral(xyz[offsets[i] + indices])
                expected = coordinate_dihedral(backend.coords()[row, indices])
                error = measured - expected
                assert abs(np.arctan2(np.sin(error), np.cos(error))) < 2e-5


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_task_budget_freezes_distal_chi_at_the_input_conformation(backend, dtype):
    from contextlib import nullcontext

    context = (
        backend.module.enable_x64() if backend.framework == "jax" else nullcontext()
    )
    with context:
        _check_budget_frozen_chi(backend, dtype)


def _check_budget_frozen_chi(backend, dtype):
    from tmol_core.chemical import ChiSamples
    from tmol_core.geometry import numpy_coord_dihedrals

    backend.molecule(MolecularCase((), (("ILE",),), ((1,),)), terms=("ref",))
    database = backend.parameter_database
    ile = next(rt for rt in database.chemical.residues if rt.name == "ILE")
    custom = attrs.evolve(
        ile,
        name="XIL",
        base_name="XIL",
        name3="XIL",
        io_equiv_class="XIL",
        dunbrack_reference=None,
        chi_samples=(
            ChiSamples("chi1", (-60.0, 60.0, 180.0), (), False),
            ChiSamples("chi2", (-45.0, 45.0), (), False),
        ),
    )
    database = attrs.evolve(
        database,
        chemical=attrs.evolve(
            database.chemical, residues=database.chemical.residues + (custom,)
        ),
    )
    backend.molecule(
        MolecularCase((), (("XIL",), ("XIL",)), ((1,), (1,))),
        terms=("ref",),
        parameter_database=database,
    )
    coordinates = backend.coords().astype(dtype)
    if backend.framework == "jax":
        backend.pose = backend.pose.replace(
            coords=backend.module.device_put(coordinates, backend.device)
        )
    else:
        backend.pose.coords = backend.module.as_tensor(
            coordinates, device=backend.device
        )
    task, sampler, concrete = sampling_task(backend)
    task.set_chi_sample_budget(3, 3)
    counts, choices, atoms, chi = map(
        backend.numpy, sampler.sample_chi_for_poses(backend.pose, concrete(task))
    )
    np.testing.assert_array_equal(counts, [3, 3])
    block = backend.pose.packed_block_types.active_block_types[0]
    indices = [block.atom_to_idx[name] for name in ("CA", "CB", "CG1", "CD1")]
    for row in range(2):
        points = backend.pose.coords[row, indices]
        if dtype == np.float64:
            # Independent plane-normal geometry avoids both framework helpers,
            # whose output dtype is itself part of the precision contract.
            a, b, c, d = backend.numpy(points)
            axis = c - b
            normal1, normal2 = np.cross(b - a, axis), np.cross(axis, d - c)
            measured = np.arctan2(
                np.dot(np.cross(normal1, normal2), axis / np.linalg.norm(axis)),
                np.dot(normal1, normal2),
            )
        else:
            measured = numpy_coord_dihedrals(*backend.numpy(points))
        assert chi.dtype == dtype
        np.testing.assert_allclose(
            chi[choices == row, 1],
            measured,
            atol=2e-14 if dtype == np.float64 else 3e-7,
            rtol=0,
        )
        np.testing.assert_array_equal(atoms[choices == row, 1], indices[2])
    # A task's tighter budget must not mutate reusable sampler chemistry.
    task.set_chi_sample_budget(1000, 1000)
    restored = sampler.sample_chi_for_poses(backend.pose, concrete(task))
    np.testing.assert_array_equal(backend.numpy(restored[0]), [6, 6])


@pytest.mark.parametrize("case_name", ("ligand", "modified_dna"))
def test_public_amino_acid_sampler_declares_uncovered_chemistry(backend, case_name):
    backend.molecule(DOMAIN_CASES[case_name], terms=("ref",))
    task, sampler, concrete = sampling_task(backend)
    fixed = concrete(task)
    before = backend.coords().copy()
    result = sampler.sample_chi_for_poses(backend.pose, fixed)
    counts, choices, atoms, chi = map(backend.numpy, result)
    assert all(value.device == backend.device for value in result)
    assert counts.shape == backend.numpy(fixed.cons_bt_block_type).shape
    assert not counts.any() and len(choices) == 0
    assert atoms.shape == chi.shape and len(chi) == 0
    np.testing.assert_array_equal(backend.coords(), before)
