import pytest
import torch

from tmol import PackerPalette, PackerTask
from tmol.pack import SetPackerTask, create_dunbrack_sampler_from_database
from tmol.tests.score.test_mirror_image_scoring import _pose, MIRROR_PAIR


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("side", ["l", "d"])
def test_public_sampling_preserves_precision_and_empty_shape(
    default_database, torch_device, dtype, side
):
    pose = _pose(f"{MIRROR_PAIR}_{side}", default_database, torch_device)
    pose.coords = pose.coords.to(dtype=dtype)
    sampler = create_dunbrack_sampler_from_database(default_database, torch_device)
    task = PackerTask(pose, PackerPalette())
    task.add_conformer_sampler(sampler)
    task.restrict_to_repacking()
    counts, choices, atoms, chi = sampler.sample_chi_for_poses(
        pose, SetPackerTask.from_packer_task(task)
    )
    assert chi.dtype == dtype and chi.device == torch_device
    assert atoms.shape == chi.shape and len(choices) == int(counts.sum()) > 1
    assert torch.isfinite(chi).all()
    task.disable_packing_by_block_mask(torch.ones_like(task.is_real_block))
    empty = sampler.sample_chi_for_poses(pose, SetPackerTask.from_packer_task(task))
    assert empty[3].dtype == dtype and empty[3].shape == empty[2].shape
    assert empty[0].shape == counts.shape and not empty[0].any()
