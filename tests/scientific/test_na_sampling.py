"""Public NA sampling preserves chemical chi and reconstructed atom identity."""

import numpy as np
import pytest

from .na_molecules import NA_CASES
from .test_conformer_coordinates import coordinate_dihedral
from .test_molecular_sampling import block_mask
from .test_current_conformers import ensure_fallback_sampler


def na_sampling_task(backend, *, level=0, sample_syn=True):
    if backend.framework == "jax":
        from tmol_jax.pack import (
            PackerTask,
            PackerPalette,
            SetPackerTask,
            NaChiRotamerSampler,
        )

        sampler = NaChiRotamerSampler.from_database(
            backend.parameter_database, chi_sample_level=level, sample_syn=sample_syn
        )
    else:
        from tmol import PackerTask, PackerPalette
        from tmol.pack import (
            SetPackerTask,
            NaChiRotamerSampler,
        )

        sampler = NaChiRotamerSampler.from_database(
            backend.parameter_database,
            backend.device,
            chi_sample_level=level,
            sample_syn=sample_syn,
        )
    task = PackerTask(backend.pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(sampler)
    return task, sampler, SetPackerTask.from_packer_task


@pytest.mark.parametrize("name", NA_CASES)
@pytest.mark.parametrize("level", (0, 1))
def test_na_samples_preserve_chi_and_molecular_identity(backend, name, level):
    backend.molecule(NA_CASES[name], terms=("na_torsion", "na_torsion_well"))
    before = backend.coords().copy()
    task, sampler, concrete = na_sampling_task(backend, level=level)
    fixed = concrete(task)
    counts, owners, atoms, chi = map(
        backend.numpy, sampler.sample_chi_for_poses(backend.pose, fixed)
    )
    assert counts.sum() == len(owners) > 0
    np.testing.assert_array_equal(owners, np.repeat(np.arange(len(counts)), counts))
    assert atoms.shape == chi.shape and np.isfinite(chi).all()
    assert chi.dtype == np.float32
    types = backend.numpy(fixed.cons_bt_block_type)
    blocks = backend.pose.packed_block_types.active_block_types
    for row, type_index in enumerate(types):
        if not sampler.defines_rotamers_for_rt(blocks[type_index]):
            assert counts[row] == 0
    if name == "rna":
        assert chi.shape[1] > 1 and np.ptp(chi[:, 1]) > 0.1
    if level == 1:
        base_task, base_sampler, base_concrete = na_sampling_task(backend)
        base_counts = backend.numpy(
            base_sampler.sample_chi_for_poses(backend.pose, base_concrete(base_task))[0]
        )
        np.testing.assert_array_equal(counts, 3 * base_counts)

    if backend.framework == "jax":
        from tmol_jax.pack import build_rotamers
    else:
        from tmol.pack import build_rotamers
    ensure_fallback_sampler(backend, task)
    _, built = build_rotamers(
        backend.pose, concrete(task), backend.parameter_database.chemical
    )
    xyz = backend.numpy(built.coords)
    btypes = backend.numpy(built.block_type_ind_for_rot)
    offsets = backend.numpy(built.coord_offset_for_rot)
    poses = backend.numpy(built.pose_for_rot)
    positions = backend.numpy(built.block_ind_for_rot)
    for pose, block in np.argwhere(backend.numpy(task.is_real_block)):
        assert np.any((poses == pose) & (positions == block))
    choice_poses = backend.numpy(fixed.cons_bt_pose)
    choice_blocks = backend.numpy(fixed.cons_bt_block)
    rank = np.zeros(len(counts), np.int32)
    for row, (type_index, offset) in enumerate(zip(btypes, offsets)):
        block = blocks[type_index]
        if not sampler.defines_rotamers_for_rt(block):
            continue
        choice = np.flatnonzero(
            (types == type_index)
            & (choice_poses == poses[row])
            & (choice_blocks == positions[row])
        )[0]
        source = np.flatnonzero(owners == choice)[rank[choice]]
        rank[choice] += 1
        # The physical dihedral is an independent coordinate reconstruction
        # invariant, not an equality between two copies of the chi formula.
        for column, atom in enumerate(atoms[source]):
            if atom < 0:
                continue
            torsion = next(
                np.asarray(uaids)[:, 0]
                for name, uaids in block.torsion_to_uaids.items()
                if name.startswith("chi") and uaids[2][0] == atom
            )
            measured = coordinate_dihedral(xyz[offset + torsion])
            delta = measured - chi[source, column]
            assert abs(np.arctan2(np.sin(delta), np.cos(delta))) < 2e-5
    np.testing.assert_array_equal(backend.coords(), before)
    task.disable_packing_by_block_mask(
        block_mask(backend, np.ones(backend.numpy(task.is_real_block).shape, bool))
    )
    empty = sampler.sample_chi_for_poses(backend.pose, concrete(task))
    assert backend.numpy(empty[0]).sum() == 0 and empty[1].shape == (0,)


def test_na_sampling_budget_cannot_drop_library_states(backend):
    backend.molecule(NA_CASES["rna"], terms=("ref",))
    task, sampler, concrete = na_sampling_task(backend, level=1)
    task.set_chi_sample_budget(1, 1)
    with pytest.raises((ValueError, RuntimeError), match="budget|limit|states"):
        sampler.sample_chi_for_poses(backend.pose, concrete(task))


def test_na_custom_database_widths_change_sample_spread(backend):
    import attrs
    from .execution import Execution

    case = NA_CASES["rna"]
    backend.molecule(case, terms=("ref",))
    database = backend.parameter_database
    source = database.scoring.na_torsion
    changed = attrs.evolve(
        database,
        scoring=attrs.evolve(
            database.scoring,
            na_torsion=attrs.evolve(
                source,
                global_parameters={
                    key: attrs.evolve(row, sdev_chi=2 * row.sdev_chi)
                    for key, row in source.global_parameters.items()
                },
            ),
        ),
    )
    other = Execution(backend.name).molecule(
        case, terms=("ref",), parameter_database=changed
    )

    def samples(execution, level):
        task, sampler, concrete = na_sampling_task(
            execution, level=level, sample_syn=False
        )
        return tuple(
            execution.numpy(x)
            for x in sampler.sample_chi_for_poses(execution.pose, concrete(task))
        )

    old, new, centers = samples(backend, 1), samples(other, 1), samples(backend, 0)
    for original, updated in zip(old[:3], new[:3]):
        np.testing.assert_array_equal(original, updated)
    center_angles = np.concatenate(
        [np.tile(centers[3][centers[1] == row, 0], 3) for row in range(len(old[0]))]
    )
    np.testing.assert_allclose(
        new[3][:, 0] - center_angles,
        2 * (old[3][:, 0] - center_angles),
        rtol=2e-6,
        atol=2e-6,
    )
    assert np.max(np.abs(new[3][:, 0] - old[3][:, 0])) > 0.1
    np.testing.assert_array_equal(new[3][:, 1:], old[3][:, 1:])
    np.testing.assert_array_equal(backend.coords(), other.coords())
