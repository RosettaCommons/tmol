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


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
def test_conformer_assembly_without_fallback_preserves_empty_selection(platform):
    backend = Execution(f"jax-{platform}")
    backend.molecule(DOMAIN_CASES["ligand"], terms=("ref",))
    task, sampler, concrete = sampling_task(backend)
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

    state = prepare_rotamer_genbonded(backend.pose, result, backend.parameter_database)
    values, indices = sparse_genbonded_scores(result.coords, state)
    assert values.shape == (1, 0) and indices.shape == (3, 0)
    gradient = jax.grad(lambda x: sparse_genbonded_scores(x, state)[0].sum())(
        result.coords
    )
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


def test_sampler_parameter_snapshot_and_chemistry_context():
    import copy
    import attrs
    from tmol_jax.pack import PackerTask, SetPackerTask, DunbrackChiSampler

    b = Execution("jax-cpu")
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
    mismatched = DunbrackChiSampler.from_database(
        attrs.evolve(db, chemical=attrs.evolve(db.chemical))
    )
    with pytest.raises(ValueError, match="chemistry context"):
        sample(mismatched)
