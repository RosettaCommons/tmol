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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_chi_assignment_preserves_parameter_and_untouched_derivatives(dtype):
    from tmol.pack.rotamer._conformer_execution import assign_chi_dofs

    chi = torch.tensor([[0.2, 0.9], [0.4, 0.7]], dtype=dtype, requires_grad=True)
    corrections = (
        torch.linspace(0.01, 0.06, 6, dtype=dtype).reshape(2, 3).requires_grad_()
    )
    original = torch.linspace(0.1, 0.9, 90, dtype=dtype).reshape(10, 9).requires_grad_()
    out = original.clone()
    assign_chi_dofs(
        torch.tensor([1, 1]),
        torch.tensor([1, 4]),
        torch.tensor([1, 0]),
        torch.tensor([[1, -1], [0, 2]], dtype=torch.int32),
        chi,
        torch.tensor([[0, 1, 2], [2, 1, 0]]),
        corrections,
        out,
    )
    # Three independent torsions replace phi at rows 6,4,2. Padding does not
    # touch row zero and must contribute no sampled-angle derivative.
    destinations = torch.tensor([6 * 9 + 3, 4 * 9 + 3, 2 * 9 + 3])
    sources = torch.tensor([0, 2, 3])
    parameter = torch.tensor([4, 3, 5])
    weights = torch.linspace(0.2, 0.8, 90, dtype=dtype)
    gradients = torch.autograd.grad(
        (weights * out.flatten().square() / 2).sum(),
        (chi, corrections, original),
        create_graph=True,
        allow_unused=True,
    )
    assert all(g is not None for g in gradients)
    values = chi.flatten()[sources] - corrections.flatten()[parameter]
    gvalues = weights[destinations] * values
    expected_chi = (
        torch.zeros(4, dtype=dtype).index_add(0, sources, gvalues).reshape_as(chi)
    )
    expected_parameters = (
        torch.zeros(6, dtype=dtype)
        .index_add(0, parameter, -gvalues)
        .reshape_as(corrections)
    )
    expected_original = (
        (weights * original.flatten())
        .index_fill(0, destinations, 0)
        .reshape_as(original)
    )
    for actual, expected in zip(
        gradients, (expected_chi, expected_parameters, expected_original)
    ):
        torch.testing.assert_close(actual, expected)
    directions = (
        torch.cos(chi.detach()),
        torch.sin(corrections.detach()),
        torch.cos(original.detach()),
    )
    hvps = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(gradients, directions)),
        (chi, corrections, original),
    )
    values_direction = (
        directions[0].flatten()[sources] - directions[1].flatten()[parameter]
    )
    hvalues = weights[destinations] * values_direction
    expected_hvps = (
        torch.zeros(4, dtype=dtype).index_add(0, sources, hvalues).reshape_as(chi),
        torch.zeros(6, dtype=dtype)
        .index_add(0, parameter, -hvalues)
        .reshape_as(corrections),
        (weights * directions[2].flatten())
        .index_fill(0, destinations, 0)
        .reshape_as(original),
    )
    for actual, expected in zip(hvps, expected_hvps):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_opth_reset_preserves_ideal_and_untouched_dof_derivatives(dtype):
    from types import SimpleNamespace
    from tmol.pack.rotamer._conformer_execution import reset_opth_dofs

    ideal = torch.linspace(0.1, 0.8, 72, dtype=dtype).reshape(2, 4, 9).requires_grad_()
    original = torch.linspace(0.2, 0.9, 90, dtype=dtype).reshape(10, 9).requires_grad_()
    out = original.clone()
    cache = SimpleNamespace(
        nhq_downstream_kfo=torch.tensor([[-1, -1], [1, 2]], dtype=torch.int32),
        nhq_downstream_count=torch.tensor([0, 2], dtype=torch.int32),
    )
    reset_opth_dofs(
        torch.tensor([1, 1]),
        torch.tensor([0, 4]),
        torch.tensor([0, 1]),
        torch.tensor([[-1, -1], [2, -1]], dtype=torch.int32),
        cache,
        ideal,
        out,
        True,
    )
    # Only the second conformer has a chi override. Its two downstream rows
    # come from type 1; the first conformer and all other rows retain input.
    destination = torch.tensor([6, 7])
    source = torch.tensor([5, 6])
    weights = torch.linspace(0.2, 0.8, 90, dtype=dtype).reshape_as(out)
    gi, go = torch.autograd.grad(
        (weights * out.square() / 2).sum(), (ideal, original), create_graph=True
    )
    expected_i = (
        torch.zeros_like(ideal)
        .reshape(-1, 9)
        .index_add(0, source, weights[destination] * ideal.reshape(-1, 9)[source])
        .reshape_as(ideal)
    )
    expected_o = (weights * original).index_fill(0, destination, 0)
    torch.testing.assert_close(gi, expected_i)
    torch.testing.assert_close(go, expected_o)
    di, do = torch.cos(ideal.detach()), torch.sin(original.detach())
    hi, ho = torch.autograd.grad((gi * di).sum() + (go * do).sum(), (ideal, original))
    expected_hi = (
        torch.zeros_like(ideal)
        .reshape(-1, 9)
        .index_add(0, source, weights[destination] * di.reshape(-1, 9)[source])
        .reshape_as(ideal)
    )
    expected_ho = (weights * do).index_fill(0, destination, 0)
    torch.testing.assert_close(hi, expected_hi)
    torch.testing.assert_close(ho, expected_ho)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mainchain_copy_preserves_finite_source_and_ideal_derivatives(dtype):
    from tmol.pack.rotamer._conformer_execution import copy_mainchain_dofs

    source = torch.linspace(0.1, 0.9, 27, dtype=dtype).reshape(3, 9)
    source[2, 4:7] = torch.tensor(
        [float("nan"), float("inf"), -float("inf")], dtype=dtype
    )
    source.requires_grad_()
    ideal = torch.linspace(0.2, 1.0, 81, dtype=dtype).reshape(9, 9).requires_grad_()
    out = ideal.clone()
    copy_mainchain_dofs(
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 0, 0]),
        torch.tensor([1, 1, 1]),
        torch.tensor([2, 4, 0]),
        torch.tensor([0, -1]),
        torch.tensor([[0, 1], [-1, -1]]),
        torch.tensor([[[[-1, -1], [1, -1]]]]),
        torch.tensor([[1, 0], [0, 1]]),
        source,
        out,
        0,
    )
    assert torch.isfinite(out).all()
    # One mapped atom is copied into each of rows4 and6, except its nonfinite
    # components4:7, which must keep their independent ideal values.
    components = torch.tensor([0, 1, 2, 3, 7, 8])
    src = (18 + components).repeat(2)
    dst = torch.cat((36 + components, 54 + components))
    weights = torch.linspace(0.3, 0.9, 81, dtype=dtype)
    gs, gi = torch.autograd.grad(
        (weights * out.flatten().square() / 2).sum(),
        (source, ideal),
        create_graph=True,
        allow_unused=True,
    )
    assert gs is not None and gi is not None
    expected_s = (
        torch.zeros(27, dtype=dtype)
        .index_add(0, src, weights[dst] * source.flatten()[src])
        .reshape_as(source)
    )
    expected_i = (weights * ideal.flatten()).index_fill(0, dst, 0).reshape_as(ideal)
    torch.testing.assert_close(gs, expected_s)
    torch.testing.assert_close(gi, expected_i)
    ds = torch.linspace(0.2, 0.7, 27, dtype=dtype).reshape_as(source)
    di = torch.linspace(0.4, 0.8, 81, dtype=dtype).reshape_as(ideal)
    hs, hi = torch.autograd.grad((gs * ds).sum() + (gi * di).sum(), (source, ideal))
    expected_hs = (
        torch.zeros(27, dtype=dtype)
        .index_add(0, src, weights[dst] * ds.flatten()[src])
        .reshape_as(source)
    )
    expected_hi = (weights * di.flatten()).index_fill(0, dst, 0).reshape_as(ideal)
    torch.testing.assert_close(hs, expected_hs)
    torch.testing.assert_close(hi, expected_hi)
