"""JAX assembly precision, topology reuse and explicit transformation boundary."""

import jax
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.dunbrack_molecules import DUNBRACK_CASES
from tests.scientific.molecules import DOMAIN_CASES
from tests.scientific.test_molecular_sampling import sampling_task
from tmol_jax.pack.rotamers import build_rotamers


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_conformer_assembly_reuses_topology_but_not_coordinates(platform):
    backend = Execution(f"jax-{platform}")
    with jax.enable_x64():
        backend.molecule(DUNBRACK_CASES["l_peptide"], terms=("ref",))
        pose = backend.pose.replace(coords=backend.pose.coords.astype(np.float64))
        backend.pose = pose
        task, _, concrete = sampling_task(backend)
        task = concrete(task)
        _, first = build_rotamers(pose, task)
        shift = jax.device_put(np.asarray([0.2, -0.3, 0.4]), backend.device)
        moved = pose.replace(coords=pose.coords + shift)
        _, second = build_rotamers(moved, task)
        assert first.coords.dtype == second.coords.dtype == np.float64
        np.testing.assert_allclose(
            second.coords, first.coords + shift, atol=2e-12, rtol=0
        )
        np.testing.assert_array_equal(
            first.block_type_ind_for_rot, second.block_type_ind_for_rot
        )
        np.testing.assert_array_equal(pose.coords, backend.coords())
        with pytest.raises(TypeError, match="eager coordinates"):
            jax.jit(lambda x: build_rotamers(pose.replace(coords=x), task)[1].coords)(
                pose.coords
            )
        # Prepared results themselves remain ordinary immutable array pytrees.
        np.testing.assert_array_equal(
            jax.jit(lambda r: r.coords)(second), second.coords
        )
        # Fresh tasks may share static chemistry, never their coordinate result.
        fresh, _, concrete = sampling_task(backend)
        fresh = concrete(fresh)
        _, third = build_rotamers(moved, fresh)
        np.testing.assert_allclose(third.coords, second.coords, atol=2e-12, rtol=0)
        lower = moved.replace(coords=moved.coords.astype(np.float32))
        _, single = build_rotamers(lower, fresh)
        assert single.coords.dtype == np.float32
        np.testing.assert_allclose(single.coords, third.coords, atol=3e-5, rtol=3e-5)
        _, restored = build_rotamers(moved, fresh)
        assert restored.coords.dtype == np.float64
        np.testing.assert_allclose(restored.coords, third.coords, atol=2e-12, rtol=0)


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("family", ("genbonded", "na_torsion", "disulfide", "metal"))
def test_conformer_assembly_without_fallback_preserves_empty_selection(
    platform, family
):
    backend = Execution(f"jax-{platform}")
    backend.molecule(DOMAIN_CASES["ligand"], terms=("ref",))
    from tmol_jax.pack import PackerPalette, PackerTask

    class NoFallbackPalette(PackerPalette):
        def default_conformer_samplers(self):
            return []

    _, sampler, concrete = sampling_task(backend)
    task = PackerTask(backend.pose, NoFallbackPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    task.disable_sampler_by_block_mask(
        sampler, jax.device_put(np.ones(task.is_real_block.shape, bool), backend.device)
    )
    _, result = build_rotamers(backend.pose, concrete(task))
    assert result.coords.shape == (0, 3) and result.n_rotamers_total == 0
    np.testing.assert_array_equal(result.n_rots_for_block, 0)
    np.testing.assert_array_equal(result.first_rot_block_type, -1)
    from tmol_jax.pack.bonded_scoring import (
        prepare_rotamer_genbonded,
        sparse_genbonded_scores,
    )

    from tmol_jax.pack.na_scoring import (
        prepare_rotamer_na_torsion,
        sparse_na_torsion_scores,
    )

    from tmol_jax.pack.connection_scoring import (
        prepare_rotamer_disulfide,
        prepare_rotamer_metal,
        sparse_connection_scores,
    )

    prepare, score, channels = {
        "genbonded": (prepare_rotamer_genbonded, sparse_genbonded_scores, 1),
        "na_torsion": (prepare_rotamer_na_torsion, sparse_na_torsion_scores, 2),
        "disulfide": (prepare_rotamer_disulfide, sparse_connection_scores, 1),
        "metal": (prepare_rotamer_metal, sparse_connection_scores, 1),
    }[family]
    state = prepare(backend.pose, result, backend.parameter_database)
    values, indices = score(result.coords, state)
    assert values.shape == (channels, 0) and indices.shape == (3, 0)
    assert result.device == values.device == indices.device == backend.device
    arrays = [a for a in jax.tree.leaves(state) if isinstance(a, jax.Array)]
    assert all(a.device == backend.device for a in arrays)
    gradient = jax.grad(lambda x: score(x, state)[0].sum())(result.coords)
    assert gradient.shape == (0, 3)


def test_conformer_assembly_rejects_unknown_sampler_science():
    from tmol_jax.pack import PackerTask, IncludeCurrentSampler

    class UnsupportedSampler(IncludeCurrentSampler):
        def create_samples_for_poses(self, pose, task):
            counts, choices, payload = super().create_samples_for_poses(pose, task)
            return counts, choices, {**payload, "unimplemented_internal_dofs": True}

    backend = Execution("jax-cpu")
    backend.molecule(DOMAIN_CASES["ligand"], terms=("ref",))
    task = PackerTask(backend.pose)
    task.add_conformer_sampler(UnsupportedSampler())
    with pytest.raises(NotImplementedError, match="unsupported conformer payload"):
        build_rotamers(backend.pose, task)


def test_reused_assembly_consumes_new_chi_payloads():
    from tests.scientific.test_conformer_coordinates import coordinate_dihedral
    from tmol_jax.pack import PackerTask, SetPackerTask, DunbrackChiSampler

    class ShiftedSampler(DunbrackChiSampler):
        shift = 0.0

        def create_samples_for_poses(self, pose, task):
            counts, choices, payload = super().create_samples_for_poses(pose, task)
            return (
                counts,
                choices,
                {
                    **payload,
                    "chi_for_rotamers": payload["chi_for_rotamers"] + self.shift,
                },
            )

    b = Execution("jax-cpu")
    b.molecule(DUNBRACK_CASES["l_peptide"], terms=("ref",))
    sampler = ShiftedSampler.from_database(b.parameter_database)
    task = PackerTask(b.pose)
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    task = SetPackerTask.from_packer_task(task)
    _, first = build_rotamers(b.pose, task)
    sampler.shift = 0.3
    _, second = build_rotamers(b.pose, task)
    np.testing.assert_array_equal(
        first.block_type_ind_for_rot, second.block_type_ind_for_rot
    )
    block_types = b.pose.packed_block_types.active_block_types
    checked = 0
    for index, type_index in enumerate(np.asarray(first.block_type_ind_for_rot)):
        block = block_types[type_index]
        if block.name != "LEU":
            continue
        atoms = np.asarray(block.torsion_to_uaids["chi1"])[:, 0]
        offset = int(first.coord_offset_for_rot[index])
        before, after = (
            coordinate_dihedral(np.asarray(r.coords)[offset + atoms])
            for r in (first, second)
        )
        difference = np.arctan2(np.sin(after - before), np.cos(after - before))
        np.testing.assert_allclose(difference, 0.3, atol=2e-5, rtol=0)
        checked += 1
    assert checked > 1


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("case", ("ligand", "dna"))
def test_ineligible_sampling_preserves_empty_result_without_fitting(platform, case):
    from tmol_jax.pack import PackerTask, SetPackerTask, DunbrackChiSampler
    from tests.scientific.test_canonical_molecular_input import (
        canonical_input,
        rebuild_canonical,
    )

    b = Execution("jax-" + platform).molecule(DOMAIN_CASES[case], terms=())
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


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_sampler_parameter_snapshot_and_chemistry_context(platform):
    import copy
    import attrs
    from tmol_jax.pack import PackerTask, SetPackerTask, DunbrackChiSampler

    b = Execution("jax-" + platform)
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
