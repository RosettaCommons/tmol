import attr
import numpy
import torch

from tmol import PoseStackBuilder
from tmol.pose import repeat_pose_stack, replace_poses, select_blocks, select_poses


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


def test_select_then_replace_poses_round_trips(ubq_40_60_pose_stack):
    stack = repeat_pose_stack(_single_pose(ubq_40_60_pose_stack), 4)
    offsets = torch.arange(4, dtype=stack.coords.dtype, device=stack.device)
    stack.coords = stack.coords + offsets[:, None, None]
    poses = torch.tensor([3, 1])
    subset = select_poses(stack, poses)
    assert subset.n_poses == 2
    torch.testing.assert_close(subset.coords, stack.coords[poses])

    edited = subset.coords + 10.0
    merged = replace_poses(stack, poses, attr.evolve(subset, coords=edited))
    torch.testing.assert_close(merged.coords[poses], edited)
    torch.testing.assert_close(merged.coords[[0, 2]], stack.coords[[0, 2]])
    for name in ("block_type_ind64", "inter_residue_connections64", "chain_id64"):
        torch.testing.assert_close(getattr(merged, name), getattr(stack, name))
    torch.testing.assert_close(
        merged.inter_block_bondsep.to_dense(), stack.inter_block_bondsep.to_dense()
    )
    numpy.testing.assert_array_equal(
        merged.pdb_info.atom_b_factor, stack.pdb_info.atom_b_factor
    )
