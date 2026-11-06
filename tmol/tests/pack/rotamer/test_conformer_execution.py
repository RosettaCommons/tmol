import pytest
import torch

from tmol.pack.rotamer._conformer_execution import copy_conformer_rows


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("width,shift", [(3, 0), (9, 1)])
def test_copied_conformer_rows_preserve_source_and_untouched_derivatives(
    dtype, width, shift
):
    # Repeated origins accumulate; untouched destinations retain their original
    # derivative. Include the virtual-root offset used for nine-component DOFs.
    x = (torch.arange(6 * width, dtype=dtype).reshape(6, width) / 10).requires_grad_()
    y = (torch.arange(6 * width, dtype=dtype).reshape(6, width) / 7).requires_grad_()
    out = y.clone()
    copy_conformer_rows(
        torch.tensor([2, 1]),
        torch.tensor([1, 1]),
        torch.tensor([0, 3]),
        x,
        out,
        root_shift=shift,
    )
    src = torch.tensor([1, 2, 1]) + shift
    dst = torch.tensor([0, 1, 3]) + shift
    weights = torch.linspace(0.2, 0.8, out.numel(), dtype=dtype).reshape_as(out)
    gx, gy = torch.autograd.grad(
        (weights * out.square() / 2).sum(), (x, y), create_graph=True, allow_unused=True
    )
    assert gx is not None, "Copied coordinates must retain their source derivative"
    expected_x = torch.zeros_like(x).index_add(0, src, weights[dst] * x[src])
    expected_y = weights * y
    expected_y[dst] = 0
    torch.testing.assert_close(gx, expected_x)
    torch.testing.assert_close(gy, expected_y)
    dx, dy = torch.cos(x.detach()), torch.sin(y.detach())
    hx, hy = torch.autograd.grad((gx * dx).sum() + (gy * dy).sum(), (x, y))
    expected_hx = torch.zeros_like(x).index_add(0, src, weights[dst] * dx[src])
    expected_hy = weights * dy
    expected_hy[dst] = 0
    torch.testing.assert_close(hx, expected_hx)
    torch.testing.assert_close(hy, expected_hy)


@pytest.mark.parametrize("sampler_name", ["IncludeCurrentSampler", "FallbackSampler"])
def test_current_conformer_construction_preserves_coordinate_derivatives(
    jagged_stack_of_465_res_ubqs,
    monkeypatch,
    sampler_name,
):
    import attr
    import tmol.pack as pack
    from tmol import PackerPalette, PackerTask

    poses = jagged_stack_of_465_res_ubqs
    coords = poses.coords.detach().clone().requires_grad_()
    poses = attr.evolve(poses, coords=coords)
    monkeypatch.setattr(PackerPalette, "default_conformer_samplers", lambda self: [])
    task = PackerTask(poses, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(getattr(pack, sampler_name)())
    _, rotamers = pack.build_rotamers(
        poses,
        pack.SetPackerTask.from_packer_task(task),
        poses.packed_block_types.chem_db,
    )
    gradient = torch.autograd.grad(rotamers.coords.square().sum() / 2, coords)[0]
    expected = torch.zeros_like(coords)
    for pose in range(poses.n_poses):
        for block in range(poses.max_n_blocks):
            kind = int(poses.block_type_ind[pose, block])
            if kind < 0:
                continue
            start = int(poses.block_coord_offset[pose, block])
            size = int(poses.packed_block_types.n_atoms[kind])
            expected[pose, start : start + size] = coords[pose, start : start + size]
    torch.testing.assert_close(gradient, expected, rtol=0, atol=0)
