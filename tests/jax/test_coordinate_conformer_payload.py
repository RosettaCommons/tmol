"""Coordinate-producing samplers retain heterogeneous atom identity on assembly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.molecules import DOMAIN_CASES
from tmol_jax.pack import (
    PackerTask,
    SetPackerTask,
    IncludeCurrentSampler,
    FallbackSampler,
    build_rotamers,
)


class TranslatedConformers(IncludeCurrentSampler):
    truncate = False

    def create_samples_for_poses(self, pose, task):
        counts, choices, _ = super().create_samples_for_poses(pose, task)
        pieces = []
        for row in np.asarray(choices):
            p, b = int(task.cons_bt_pose[row]), int(task.cons_bt_block[row])
            start = int(pose.block_coord_offset[p, b])
            n = pose.block_type(p, b).n_atoms
            pieces.append(pose.coords[p, start : start + n] + (b + 1) * 0.25)
        coordinates = jnp.concatenate(pieces)
        if self.truncate:
            coordinates = coordinates[:-1]
        return counts, choices, {"conformer_coordinates": coordinates}


@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_coordinate_payload_preserves_identity_and_fresh_values(dtype):
    with jax.enable_x64():
        backend = Execution("jax-cpu").molecule(DOMAIN_CASES["mixed_batch"], terms=())
        pose = backend.pose.replace(coords=backend.pose.coords.astype(dtype))
        task = PackerTask(pose)
        direct = TranslatedConformers()
        mask = np.asarray([[False, True, True], [True, False, False]])
        task.add_conformer_sampler_by_block_mask(direct, mask)
        task.add_conformer_sampler(FallbackSampler())
        task.restrict_to_repacking()
        task = SetPackerTask.from_packer_task(task)
        for shift in (0.0, 0.5):
            current = pose.replace(coords=pose.coords + shift)
            _, rotamers = build_rotamers(current, task)
            assert rotamers.coords.dtype == dtype
            for rot in range(rotamers.n_rotamers_total):
                p, b = int(rotamers.pose_for_rot[rot]), int(
                    rotamers.block_ind_for_rot[rot]
                )
                assert int(rotamers.block_type_ind_for_rot[rot]) == int(
                    pose.block_type_ind[p, b]
                )
                start, target = int(pose.block_coord_offset[p, b]), int(
                    rotamers.coord_offset_for_rot[rot]
                )
                n = pose.block_type(p, b).n_atoms
                expected = np.asarray(current.coords[p, start : start + n])
                if mask[p, b]:
                    expected = expected + (b + 1) * 0.25
                np.testing.assert_array_equal(
                    rotamers.coords[target : target + n], expected
                )
        direct.truncate = True
        with pytest.raises(ValueError, match="cover every selected atom"):
            build_rotamers(pose, task)
