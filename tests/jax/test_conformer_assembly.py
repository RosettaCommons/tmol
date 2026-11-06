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
