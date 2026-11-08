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


def check_sampler_support(backend, sampler, task, counts):
    """Advertised support must agree with real samples at allowed task rows."""
    support = sampler.defines_rotamers_for_bts(
        backend.pose.packed_block_types, task.cons_bt_block_type
    )
    assert support.device == backend.device
    support = backend.numpy(support)
    allowed = backend.numpy(task.is_cons_bt_allowed)
    assert support.shape == counts.shape and support.dtype == np.bool_
    np.testing.assert_array_equal(support[allowed], counts[allowed] > 0)


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
    check_sampler_support(backend, sampler, fixed, counts)
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

    if backend.framework == "jax" and case_name == "l_peptide":
        _check_sampler_snapshot(backend)

    if backend.framework == "jax" and case_name == "l_peptide":
        _check_native_sampling(backend, sampler.tables)

    if backend.name == "jax-cpu" and case_name == "l_peptide":
        _check_task_coordinate_reuse_and_device_errors(backend)


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

    if backend.framework == "jax":
        _check_ineligible_sampler(backend, "ligand" if case_name == "ligand" else "dna")


def _check_ineligible_sampler(backend, case):
    jax = backend.module
    from .dunbrack_molecules import DUNBRACK_CASES
    from .molecules import DOMAIN_CASES

    from tmol_jax.pack import PackerTask, SetPackerTask, DunbrackChiSampler
    from tests.scientific.test_canonical_molecular_input import (
        canonical_input,
        rebuild_canonical,
    )

    b = backend.molecule(DOMAIN_CASES[case], terms=())
    co, cf = canonical_input(DOMAIN_CASES[case], b.parameter_database)
    rebuild_canonical(b, co, cf)
    # The public palette includes amino acids; eligibility comes from the actual
    # task choices, not from whether any palette type has a Dunbrack library.
    assert any(
        rt.base_name == "LEU" for rt in b.pose.packed_block_types.active_block_types
    )
    sampler = DunbrackChiSampler.from_database(b.parameter_database)
    task = PackerTask(b.pose)
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    concrete = SetPackerTask.from_packer_task(task)
    assert np.asarray(concrete.is_cons_bt_allowed).any()
    width = sampler._host_metadata(b.pose.packed_block_types).max_n_chi
    with jax.enable_x64():
        for dtype in (np.float32, np.float64):
            pose = b.pose.replace(coords=b.pose.coords.astype(dtype))
            counts, choices, atoms, chi = sampler.sample_chi_for_poses(pose, concrete)
            assert counts.shape == concrete.cons_bt_pose.shape
            np.testing.assert_array_equal(counts, 0)
            assert choices.shape == (0,) and atoms.shape == chi.shape == (0, width)
            assert counts.dtype == choices.dtype == atoms.dtype == np.int32
            assert chi.dtype == dtype
            assert all(x.device == b.device for x in (counts, choices, atoms, chi))
    assert sampler._tables is None and sampler._native_storage is None

    # An empty request must not permanently disable a reusable sampler.
    b.molecule(
        DUNBRACK_CASES["l_peptide"],
        terms=(),
        parameter_database=b.parameter_database,
    )
    task = PackerTask(b.pose)
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    counts, _, _, chi = sampler.sample_chi_for_poses(b.pose, task)
    assert np.asarray(counts).sum() > 0 and np.isfinite(np.asarray(chi)).all()


def _check_sampler_snapshot(backend):
    jax = backend.module
    from .dunbrack_molecules import DUNBRACK_CASES

    import copy
    import attrs
    from tmol_jax.pack import PackerTask, SetPackerTask, DunbrackChiSampler

    b = backend
    b.molecule(DUNBRACK_CASES["l_peptide"], terms=("ref",))
    prepared = DunbrackChiSampler.from_database(b.parameter_database)
    tables = copy.deepcopy(prepared.tables)
    means = tables.sampling_db["rotameric_mean_tables"].copy()
    tables.sampling_db["rotameric_mean_tables"] = means
    sampler = DunbrackChiSampler(tables)

    def sample(current):
        task = PackerTask(b.pose)
        task.restrict_to_repacking()
        task.add_conformer_sampler(current)
        return tuple(
            np.asarray(x)
            for x in current.sample_chi_for_poses(
                b.pose, SetPackerTask.from_packer_task(task)
            )
        )

    first = sample(sampler)
    means += 0.25
    retained = sample(sampler)
    changed = sample(DunbrackChiSampler(tables))
    for before, after in zip(first, retained):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(first[:3], changed[:3]):
        np.testing.assert_array_equal(before, after)
    assert np.max(np.abs(changed[3] - first[3])) > 1e-5

    db = b.parameter_database
    library = db.scoring.dunbrack
    name = next(
        row.dun_table_name for row in library.dun_lookup if row.residue_name == "LEU"
    )
    records = []
    for record in library.rotameric_libraries:
        if record.table_name == name:
            source_means = record.rotameric_data.rotamer_means.copy()
            record = attrs.evolve(
                record,
                rotameric_data=attrs.evolve(
                    record.rotameric_data, rotamer_means=source_means
                ),
            )
        records.append(record)
    custom = attrs.evolve(
        db,
        scoring=attrs.evolve(
            db.scoring,
            dunbrack=attrs.evolve(library, rotameric_libraries=tuple(records)),
        ),
    )
    expected = sample(DunbrackChiSampler.from_database(custom))
    delayed = DunbrackChiSampler.from_database(custom)
    # Mutate the caller's source before this sampler first reads fitted tables.
    # Delayed preparation must retain the same eager snapshot contract.
    source_means += 10
    retained = sample(delayed)
    changed = sample(DunbrackChiSampler.from_database(custom))
    for before, after in zip(expected, retained):
        np.testing.assert_array_equal(before, after)
    for before, after in zip(expected[:3], changed[:3]):
        np.testing.assert_array_equal(before, after)
    assert np.max(np.abs(changed[3] - expected[3])) > 1e-5

    mismatched = DunbrackChiSampler.from_database(
        attrs.evolve(db, chemical=attrs.evolve(db.chemical))
    )
    with pytest.raises(ValueError, match="chemistry context"):
        sample(mismatched)


def _check_native_sampling(backend, tables):
    jax, jnp, sampling_device = backend.module, backend.module.numpy, backend.device
    from tmol_jax.pack._native import sample_chi, sample_chi_cpu

    with jax.enable_x64(False):
        with pytest.raises(ValueError, match="preserve output dtype"):
            sample_chi_cpu(tables, np.zeros((4, 3), np.float64))
    with pytest.raises(TypeError, match="float32 or float64"):
        sample_chi_cpu(tables, np.zeros((4, 3), np.float16))

    for dtype in (np.float32, np.float64):
        coordinates = np.zeros((4, 3), dtype)
        angles = np.asarray([-np.pi / 3, np.pi / 3, np.pi], dtype)
        request = (
            np.array([2], np.int32),
            np.array([0], np.int32),
            np.full((2, 4), -1, np.int32),
            np.array([[0, -1]], np.int32),
            np.zeros((1, 1), np.int32),
            angles.reshape(1, 1, 3),
            np.array([[3]], np.int32),
            np.array([0.98], dtype),
            np.array([1], np.int32),
        )
        with jax.enable_x64():
            coordinates = jax.device_put(coordinates, sampling_device)
            counts, offsets, identities, chi = sample_chi(tables, coordinates, *request)
            np.testing.assert_array_equal(counts, [3])
            np.testing.assert_array_equal(offsets, [0])
            np.testing.assert_array_equal(identities, [0, 0, 0])
            np.testing.assert_array_equal(chi[:, 0], angles)
            assert chi.dtype == dtype
            assert all(
                value.device == sampling_device
                for value in (counts, offsets, identities, chi)
            )
            with pytest.raises(ValueError, match="budget|limit"):
                sample_chi(tables, coordinates, *request, max_samples_per_restype=2)
            # Dynamic output sizes cannot be silently traced as a fixed-size kernel.
            with pytest.raises(TypeError, match="eager"):
                jax.jit(lambda x: sample_chi(tables, x, *request))(
                    jnp.asarray(coordinates)
                )

    indices = [tables.all_table_indices[name] for name in ("ARG", "DARG")]
    request = (
        np.array([2], np.int32),
        np.array([0], np.int32),
        np.full((2, 4), -1, np.int32),
        np.array([[0, index] for index in indices], np.int32),
        np.zeros((2, 4), np.int32),
        np.zeros((2, 4, 1), np.float32),
        np.ones((2, 4), np.int32),
        np.full(2, 0.95, np.float32),
        np.full(2, 4, np.int32),
    )
    # Metadata remain int64 without changing the user's floating precision mode.
    with jax.enable_x64(False):
        coordinates = jax.device_put(np.zeros((1, 3), np.float32), sampling_device)
        count, offsets, identities, chi = sample_chi(tables, coordinates, *request)
        count, offsets, identities, chi = map(
            np.asarray, (count, offsets, identities, chi)
        )
        assert count[0] == count[1] and count[0] > 1
        np.testing.assert_array_equal(offsets, [0, count[0]])
        np.testing.assert_array_equal(identities, np.repeat([0, 1], count))
        np.testing.assert_allclose(chi[: count[0]], -chi[count[0] :], rtol=0, atol=2e-6)
        assert chi.dtype == np.float32 and not jax.config.x64_enabled

    request = (
        np.empty(0, np.int32),
        np.empty(0, np.int32),
        np.empty((0, 4), np.int32),
        np.empty((0, 2), np.int32),
        np.empty((0, 0), np.int32),
        np.empty((0, 0, 0), np.float32),
        np.empty((0, 0), np.int32),
        np.empty(0, np.float32),
        np.empty(0, np.int32),
    )
    result = sample_chi(
        tables, jax.device_put(np.empty((0, 3), np.float32), sampling_device), *request
    )
    assert tuple(x.shape for x in result) == ((0,), (0,), (0,), (0, 0))
    assert all(x.device == sampling_device for x in result)


def _check_task_coordinate_reuse_and_device_errors(backend):
    jax = backend.module
    from tmol_jax.pack import PackerTask

    backend.molecule(DUNBRACK_CASES["l_peptide"], terms=("ref",))
    task, sampler, concrete = sampling_task(backend)
    fixed = concrete(task)
    pose = backend.pose
    coords = pose.coords.at[0, 2, 0].add(0.2)
    updated = pose.replace(coords=coords)
    fresh = PackerTask(updated)
    fresh.add_conformer_sampler(sampler)
    fresh.restrict_to_repacking()
    reused = sampler.sample_chi_for_poses(updated, fixed)
    expected = sampler.sample_chi_for_poses(updated, fresh)
    for actual, wanted in zip(reused, expected):
        np.testing.assert_array_equal(actual, wanted)
    changed = updated.replace(block_type_ind=updated.block_type_ind.at[0, 1].set(0))
    with pytest.raises(ValueError, match="layout"):
        sampler.sample_chi_for_poses(changed, fixed)

    def sample(xyz):
        return sampler.sample_chi_for_poses(pose.replace(coords=xyz), fixed)

    with pytest.raises(TypeError, match="eager"):
        jax.jit(sample)(coords)
    with pytest.raises(TypeError, match="eager"):
        jax.vmap(sample)(coords[None])

    from tmol_jax import pack_rotamers

    b = backend.molecule(DUNBRACK_CASES["l_peptide"], terms=("ref",))
    task = PackerTask(b.pose)
    for device, error, message in (
        ("mps", ValueError, "native CPU and CUDA"),
        ("cpu:-1", ValueError, "index is out of range"),
        (object(), TypeError, "JAX device or a device string"),
    ):
        with pytest.raises(error, match=message):
            pack_rotamers(b.pose, b.score_function, task, seed=17, anneal_device=device)
