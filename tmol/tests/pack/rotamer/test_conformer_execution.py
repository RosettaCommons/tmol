import pytest
import torch


@pytest.mark.parametrize(
    "sampler_name", ["IncludeCurrentSampler", "FallbackSampler", "OptHSampler"]
)
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
    if sampler_name == "OptHSampler":
        with pytest.raises(NotImplementedError, match="inverse-kinematics"):
            pack.build_rotamers(
                poses,
                pack.SetPackerTask.from_packer_task(task),
                poses.packed_block_types.chem_db,
            )
        return
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_measured_chi_correction_preserves_coordinate_and_sample_derivatives(dtype):
    from tmol.numeric import coord_dihedrals
    from tmol.pack.rotamer import _conformer_execution as execution

    coordinates = torch.tensor(
        [
            [8.0, 9.0, 7.0],
            [1.2, 0.1, -0.2],
            [0.1, -0.3, 0.2],
            [0.3, 1.1, -0.1],
            [0.8, 1.3, 1.2],
        ],
        dtype=dtype,
        requires_grad=True,
    )
    chi = torch.tensor([[0.4, 0.9], [0.7, 0.8]], dtype=dtype, requires_grad=True)
    original = torch.linspace(0.1, 0.9, 63, dtype=dtype).reshape(7, 9).requires_grad_()
    four_atoms = torch.full((1, 4, 4), -1, dtype=torch.int64)
    four_atoms[0, 3] = torch.arange(4)
    output = original.clone()
    execution.correct_chi_dofs(
        torch.tensor([0]),
        torch.tensor([1]),
        torch.tensor([0, 0]),
        torch.tensor([[3, -1], [3, 2]], dtype=torch.int32),
        chi,
        coordinates,
        four_atoms,
        torch.tensor([[3, 0, 2, 1]]),
        output,
    )
    # Two samples intentionally target the same torsion. The other columns
    # represent absent atoms / an atom without a valid four-atom definition.
    xyz = coordinates[1:].double()
    measured = coord_dihedrals(*(xyz[:, None, :]), output_dtype=dtype)[0]
    delta = chi[:, 0].double() - measured.double()
    delta = (delta + torch.pi) % (2 * torch.pi) - torch.pi
    expected = (
        original.flatten()
        .index_add(0, torch.tensor([30, 30]), delta.to(dtype))
        .reshape_as(original)
    )
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    weights = torch.linspace(0.2, 0.8, 63, dtype=dtype).reshape_as(original)
    inputs = (chi, coordinates, original)
    gradients = torch.autograd.grad(
        (weights * output.square() / 2).sum(),
        inputs,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    assert all(g is not None for g in gradients)
    reference = torch.autograd.grad(
        (weights * expected.square() / 2).sum(),
        inputs,
        create_graph=True,
    )
    for actual, truth in zip(gradients, reference):
        torch.testing.assert_close(actual, truth)
    direction = [torch.sin(value.detach()) for value in inputs]
    hessian = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(gradients, direction)),
        inputs,
    )
    expected_hessian = torch.autograd.grad(
        sum((g * v).sum() for g, v in zip(reference, direction)),
        inputs,
    )
    for actual, truth in zip(hessian, expected_hessian):
        torch.testing.assert_close(actual, truth)
