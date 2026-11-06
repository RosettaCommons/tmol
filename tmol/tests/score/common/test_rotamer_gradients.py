"""Independent directional derivatives for weighted D-peptide rotamer pairs."""

import pytest
import torch

from tmol import PackerPalette, PackerTask, extended_pose_stack_from_sequences
from tmol.pack import IncludeCurrentSampler, SetPackerTask, build_rotamers
from tmol.score import LJLKEnergyTerm, DunbrackEnergyTerm, BackboneTorsionEnergyTerm


@pytest.mark.parametrize(
    "term_class", [LJLKEnergyTerm, DunbrackEnergyTerm, BackboneTorsionEnergyTerm]
)
def test_weighted_rotamer_directional_derivative(
    default_database, torch_device, term_class
):
    pose = extended_pose_stack_from_sequences(
        ["X[DALA]X[DVAL]", "X[DPRO]X[DPHE]X[DVAL]"], device=torch_device
    )
    task = PackerTask(pose, PackerPalette())
    task.restrict_to_repacking()
    task.add_conformer_sampler(IncludeCurrentSampler())
    pose, rotamers = build_rotamers(
        pose, SetPackerTask.from_packer_task(task), default_database.chemical
    )
    term = term_class(default_database, torch_device)
    for block in pose.packed_block_types.active_block_types:
        term.setup_block_type(block)
    term.setup_packed_block_types(pose.packed_block_types)
    term.setup_poses(pose)
    scorer = term.render_rotamer_scoring_module(pose, rotamers)
    coords = rotamers.coords.double().detach()
    # Ideal library conformers can sit on discrete rotamer classification
    # boundaries. Test the derivative within a smooth region and require
    # convergence at two independent finite-difference step sizes.
    perturbation = torch.sin(
        0.37 * torch.arange(coords.numel(), device=torch_device, dtype=coords.dtype)
    ).reshape_as(coords)
    coords = (coords + 0.017 * perturbation).requires_grad_()
    values, pairs = scorer(coords)
    weights = torch.linspace(
        -0.3, 0.7, values.numel(), device=torch_device, dtype=coords.dtype
    ).reshape_as(values)
    gradient = torch.autograd.grad((values * weights).sum(), coords)[0]
    direction = torch.sin(
        torch.arange(coords.numel(), device=torch_device, dtype=coords.dtype)
    ).reshape_as(coords)
    assert values.abs().sum() > 0 and gradient.abs().sum() > 0
    for epsilon in (1e-5, 1e-6):
        plus, plus_pairs = scorer(coords.detach() + epsilon * direction)
        minus, minus_pairs = scorer(coords.detach() - epsilon * direction)
        torch.testing.assert_close(pairs, plus_pairs, rtol=0, atol=0)
        torch.testing.assert_close(pairs, minus_pairs, rtol=0, atol=0)
        finite_difference = ((plus - minus) * weights).sum() / (2 * epsilon)
        torch.testing.assert_close(
            (gradient * direction).sum(), finite_difference, rtol=2e-5, atol=1e-5
        )
    # Internal pair forces cannot change the center of mass of the system.
    torch.testing.assert_close(
        gradient.sum(dim=0), torch.zeros_like(gradient[0]), rtol=0, atol=1e-7
    )
