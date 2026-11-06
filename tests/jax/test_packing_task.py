"""Task validity follows molecular topology, not coordinate object identity."""

import jax
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.dunbrack_molecules import DUNBRACK_CASES
from tests.scientific.test_molecular_sampling import sampling_task
from tmol_jax.pack import PackerTask


def test_task_coordinate_reuse_topology_invalidation_and_eager_contract():
    backend = Execution("jax-cpu").molecule(DUNBRACK_CASES["l_peptide"], terms=("ref",))
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
