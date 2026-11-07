import numpy
import torch

from tmol import PoseStackBuilder
from tmol.pose import repeat_pose_stack, select_blocks


def _single_pose(ubq_40_60_pose_stack):
    return ubq_40_60_pose_stack.split(0)


def test_repeat_pose_stack_matches_builder(ubq_40_60_pose_stack):
    single = _single_pose(ubq_40_60_pose_stack)
    repeated = repeat_pose_stack(single, 3)
    expected = PoseStackBuilder.from_poses([single] * 3, single.device)
    assert repeated.n_poses == 3
    for name in (
        "coords",
        "block_type_ind64",
        "block_coord_offset64",
        "inter_residue_connections64",
        "chain_id64",
    ):
        torch.testing.assert_close(getattr(repeated, name), getattr(expected, name))
    torch.testing.assert_close(
        repeated.inter_block_bondsep.to_dense(), expected.inter_block_bondsep.to_dense()
    )
    numpy.testing.assert_array_equal(
        repeated.pdb_info.residue_labels, expected.pdb_info.residue_labels
    )


def test_select_blocks_keeps_selected_atoms_and_clears_dropped_links(
    ubq_40_60_pose_stack,
):
    single = _single_pose(ubq_40_60_pose_stack)
    blocks = [3, 4, 9]
    selected = select_blocks(single, blocks)
    assert selected.max_n_blocks == 3

    for new, old in enumerate(blocks):
        n_atoms = single.block_type(0, old).n_atoms
        old_start = int(single.block_coord_offset64[0, old])
        new_start = int(selected.block_coord_offset64[0, new])
        torch.testing.assert_close(
            selected.coords[0, new_start : new_start + n_atoms],
            single.coords[0, old_start : old_start + n_atoms],
        )

    partners = selected.inter_residue_connections64[0, :, :, 0]
    # blocks 3 and 4 stay bonded to each other; the link 4-5 and 9's links go away
    assert set(partners.flatten().tolist()) <= {-1, 0, 1}
    assert (partners[0] == 1).any() and (partners[1] == 0).any()
    assert (partners[2] == -1).all()

    numpy.testing.assert_array_equal(
        selected.pdb_info.residue_labels[0], single.pdb_info.residue_labels[0, blocks]
    )
    torch.testing.assert_close(
        selected.inter_block_bondsep.to_dense(),
        single.inter_block_bondsep.to_dense()[:, blocks][:, :, blocks],
    )


def test_selected_repeated_coordinates_preserve_exact_cotangents_and_dtype(
    ubq_40_60_pose_stack,
):
    import attr

    single = _single_pose(ubq_40_60_pose_stack)
    coords = single.coords.to(torch.float64).detach().requires_grad_(True)
    single = attr.evolve(single, coords=coords)
    blocks = [9, 3, 4]
    selected = select_blocks(single, blocks)
    repeated = repeat_pose_stack(selected, 3)
    assert repeated.coords.dtype == coords.dtype
    assert repeated.coords.device == coords.device
    cotangent = torch.arange(
        repeated.coords.numel(), dtype=coords.dtype, device=coords.device
    ).reshape(repeated.coords.shape)
    gradient = torch.autograd.grad((repeated.coords * cotangent).sum(), coords)[0]
    expected = torch.zeros_like(coords)
    start = 0
    for block in blocks:
        old_start = int(single.block_coord_offset64[0, block])
        length = single.block_type(0, block).n_atoms
        expected[0, old_start : old_start + length] = cotangent[
            :, start : start + length
        ].sum(dim=0)
        start += length
    torch.testing.assert_close(gradient, expected, atol=0, rtol=0)
    repeated.coords[1].detach().zero_()
    torch.testing.assert_close(selected.coords, repeated.coords[0:1], atol=0, rtol=0)


def test_repeat_constraints_and_metadata_are_independent(ubq_40_60_pose_stack):
    import attr

    from tmol.pose._constraint_set import ConstraintSet
    from tmol.score.constraint._constraint_energy_term import ConstraintEnergyTerm

    single = _single_pose(ubq_40_60_pose_stack)
    device = single.device
    atoms = torch.tensor([[[0, 3, 0], [0, 4, 0]]], dtype=torch.int32, device=device)
    params = torch.tensor([[1.47, 0.1]], dtype=torch.float32, device=device)
    constraints = ConstraintSet.create_empty(device, 1).add_constraints(
        ConstraintEnergyTerm.harmonic, atoms, params
    )
    single = attr.evolve(single, constraint_set=constraints)
    repeated = repeat_pose_stack(single, 3)
    actual = repeated.constraint_set
    torch.testing.assert_close(
        actual.constraint_atoms[:, :2, 0],
        torch.arange(3, device=device, dtype=torch.int32)[:, None].expand(3, 2),
    )
    torch.testing.assert_close(actual.constraint_params, params.expand(3, 2))
    assert actual.constraint_functions == constraints.constraint_functions
    assert select_blocks(single, [3, 4]).constraint_set is None
    labels = single.pdb_info.residue_labels.copy()
    repeated.pdb_info.residue_labels[1] = -99
    numpy.testing.assert_array_equal(single.pdb_info.residue_labels, labels)
    numpy.testing.assert_array_equal(repeated.pdb_info.residue_labels[0], labels[0])
