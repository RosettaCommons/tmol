"""Masked whole-pose scoring must equal whole-pose scoring.

Skipping pairs of frozen blocks removes an energy that cannot change while
those blocks are fixed, so a masked scorer plus the restored constant has to
reproduce the unmasked score and the gradients on movable atoms.
"""

import pytest
import torch

from tmol import pose_stack_from_file
from tmol.score import beta2016_score_function


def _movable_block_mask(pose_stack, blocks):
    mask = torch.zeros(
        pose_stack.block_type_ind64.shape, dtype=torch.int32, device=pose_stack.device
    )
    mask[:, list(blocks)] = 1
    return mask


def _atom_mask_for_blocks(pose_stack, blocks):
    mask = torch.zeros(
        pose_stack.coords.shape[:2], dtype=torch.bool, device=pose_stack.device
    )
    pbt = pose_stack.packed_block_types
    for block in blocks:
        for pose in range(pose_stack.n_poses):
            block_type = int(pose_stack.block_type_ind64[pose, block])
            if block_type < 0:
                continue
            start = int(pose_stack.block_coord_offset64[pose, block])
            mask[pose, start : start + int(pbt.n_atoms[block_type])] = True
    return mask


@pytest.fixture
def pose(ubq_pdb, default_database, torch_device):
    return pose_stack_from_file(
        ubq_pdb, torch_device, residue_start=0, residue_end=20, no_optH=True
    )


def test_masked_score_matches_whole_pose(pose, torch_device):
    """One movable block: masked score plus offset equals the unmasked score."""
    sfxn = beta2016_score_function(torch_device)
    unmasked = sfxn.render_whole_pose_scoring_module(pose)
    masked = sfxn.render_whole_pose_scoring_module(
        pose, movable_blocks=_movable_block_mask(pose, [5])
    )
    with torch.no_grad():
        reference = unmasked(pose.coords)
        restored = masked(pose.coords)
    torch.testing.assert_close(restored, reference, rtol=1e-5, atol=1e-3)


def test_masked_score_tracks_movable_motion(pose, torch_device):
    """The offset stays valid as movable atoms move; frozen blocks do not."""
    sfxn = beta2016_score_function(torch_device)
    movable = [5]
    unmasked = sfxn.render_whole_pose_scoring_module(pose)
    masked = sfxn.render_whole_pose_scoring_module(
        pose, movable_blocks=_movable_block_mask(pose, movable)
    )
    atom_mask = _atom_mask_for_blocks(pose, movable)
    moved = pose.coords.clone()
    moved[atom_mask] += 0.25
    with torch.no_grad():
        torch.testing.assert_close(masked(moved), unmasked(moved), rtol=1e-5, atol=1e-3)


def test_masked_gradients_match_on_movable_atoms(pose, torch_device):
    sfxn = beta2016_score_function(torch_device)
    movable = [5]
    atom_mask = _atom_mask_for_blocks(pose, movable)

    grads = []
    for kwargs in ({}, {"movable_blocks": _movable_block_mask(pose, movable)}):
        scorer = sfxn.render_whole_pose_scoring_module(pose, **kwargs)
        coords = pose.coords.detach().clone().requires_grad_(True)
        scorer(coords).sum().backward()
        grads.append(coords.grad[atom_mask])
    torch.testing.assert_close(grads[1], grads[0], rtol=1e-4, atol=1e-4)


def test_all_blocks_movable_is_a_no_op(pose, torch_device):
    """A fully movable mask drops no pair, so the offset is zero."""
    sfxn = beta2016_score_function(torch_device)
    every_block = range(pose.block_type_ind64.shape[1])
    masked = sfxn.render_whole_pose_scoring_module(
        pose, movable_blocks=_movable_block_mask(pose, every_block)
    )
    unmasked = sfxn.render_whole_pose_scoring_module(pose)
    with torch.no_grad():
        torch.testing.assert_close(
            masked.offset, torch.zeros_like(masked.offset), rtol=0, atol=1e-4
        )
        torch.testing.assert_close(
            masked(pose.coords), unmasked(pose.coords), rtol=0, atol=1e-4
        )


def test_distant_movable_blocks_keep_their_pairs(pose, torch_device):
    """Movability is declared, not inferred: a far movable block still scores.

    Two movable blocks at opposite ends of the chain stand in for a flexible
    sidechain outside any ligand-centered radius. A proximity-based crop would
    drop one; the mask must not.
    """
    sfxn = beta2016_score_function(torch_device)
    n_blocks = pose.block_type_ind64.shape[1]
    movable = [0, n_blocks - 1]
    masked = sfxn.render_whole_pose_scoring_module(
        pose, movable_blocks=_movable_block_mask(pose, movable)
    )
    unmasked = sfxn.render_whole_pose_scoring_module(pose)
    atom_mask = _atom_mask_for_blocks(pose, movable)
    moved = pose.coords.clone()
    moved[atom_mask] += 0.3
    with torch.no_grad():
        torch.testing.assert_close(masked(moved), unmasked(moved), rtol=1e-5, atol=1e-3)


def test_atom_mask_selects_the_same_blocks(pose, torch_device):
    """An atom-shaped mask marks a block movable when any of its atoms move."""
    sfxn = beta2016_score_function(torch_device)
    movable = [5, 9]
    from_blocks = sfxn.render_whole_pose_scoring_module(
        pose, movable_blocks=_movable_block_mask(pose, movable)
    )
    from_atoms = sfxn.render_whole_pose_scoring_module(
        pose, movable_blocks=_atom_mask_for_blocks(pose, movable)
    )
    torch.testing.assert_close(from_atoms.offset, from_blocks.offset, rtol=0, atol=1e-4)


def test_atom_mask_ignores_padding_in_a_heterogeneous_batch(pose, torch_device):
    from tmol import PoseStackBuilder
    from tmol.pose import select_blocks
    from tmol.score._score_function import _normalized_movable_blocks

    short = select_blocks(pose, [0, 1, 2])
    batch = PoseStackBuilder.from_poses([short, pose], torch_device)
    atoms = ~batch.real_atoms
    # Exercise both ends of a block and the boundary to the next block.
    atoms[0, batch.block_coord_offset64[0, 2] - 1] = True
    atoms[1, batch.block_coord_offset64[1, 7]] = True
    expected = torch.zeros_like(batch.block_type_ind)
    expected[0, 1] = expected[1, 7] = 1
    torch.testing.assert_close(_normalized_movable_blocks(batch, atoms), expected)


def test_frozen_scores_restore_term_lanes_after_reweighting(
    pose, torch_device, default_database
):
    sfxn = beta2016_score_function(torch_device, default_database)
    mask = _movable_block_mask(pose, [5])
    full = sfxn.render_whole_pose_scoring_module(pose)
    masked = sfxn.render_whole_pose_scoring_module(pose, movable_blocks=mask)
    with torch.no_grad():
        # The two rendered modules can share the source weight storage.
        full.weights[0].fill_(0.37)
        masked.weights[0].fill_(0.37)
        for sum_terms in (True, False):
            for apply_weights in (True, False):
                torch.testing.assert_close(
                    masked(pose.coords, sum_terms, apply_weights),
                    full(pose.coords, sum_terms, apply_weights),
                    rtol=1e-5,
                    atol=1e-3,
                )
        torch.testing.assert_close(
            masked.unweighted_scores(pose.coords),
            full.unweighted_scores(pose.coords),
            rtol=1e-5,
            atol=1e-3,
        )


def test_frozen_cartesian_reset_rejects_changed_context(pose, torch_device):
    import attrs
    from tmol.optimization._sfxn_modules import CartesianSfxnNetwork

    sfxn = beta2016_score_function(torch_device)
    mask = _atom_mask_for_blocks(pose, [5])
    network = CartesianSfxnNetwork(sfxn, pose, coord_mask=mask)
    moved = pose.coords.clone()
    moved[mask] += 0.1
    assert network._reset(sfxn, attrs.evolve(pose, coords=moved), mask)
    moved[_atom_mask_for_blocks(pose, [17])] += 0.7
    assert not network._reset(sfxn, attrs.evolve(pose, coords=moved), mask)


def test_masked_block_pair_unweighted_entry_points(pose, torch_device):
    from tmol.score import ScoreFunction
    from tmol.score._score_types import ScoreType
    from tmol.database import ParameterDatabase

    mask = _movable_block_mask(pose, [5])
    single = ScoreFunction(ParameterDatabase.get_default(), torch_device)
    single.set_weight(ScoreType.fa_elec, 1.0)
    for sfxn in (single, beta2016_score_function(torch_device)):
        scorer = sfxn.render_block_pair_scoring_module(pose, movable_blocks=mask)
        with torch.no_grad():
            lanes = scorer(pose.coords, sum_terms=False, apply_weights=False)
            torch.testing.assert_close(scorer.unweighted_scores(pose.coords), lanes)
            if sfxn is single:
                frozen = mask[0] == 0
                assert torch.count_nonzero(lanes[..., frozen, :][..., frozen]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph replay")
def test_masked_cuda_graph_agrees_with_eager(ubq_pdb):
    device = torch.device("cuda")
    pose = pose_stack_from_file(ubq_pdb, device, residue_end=20, no_optH=True)
    mask = _movable_block_mask(pose, [5])
    sfxn = beta2016_score_function(device)
    eager = sfxn.render_whole_pose_scoring_module(pose, movable_blocks=mask)
    graph = sfxn.render_whole_pose_scoring_module(
        pose, movable_blocks=mask, cuda_graph="both"
    )
    x = pose.coords.detach().requires_grad_(True)
    with torch.no_grad():
        torch.testing.assert_close(graph(x), eager(x), rtol=1e-5, atol=1e-3)
    assert graph.enable_cuda_graphs(x) is graph
    expected, actual = eager(x), graph(x)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-3)
    (actual_grad,) = torch.autograd.grad(actual.sum(), x)
    (expected_grad,) = torch.autograd.grad(expected.sum(), x)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-4, atol=1e-3)


def test_kinematic_reset_checks_declared_mask_and_frozen_coordinates(
    pose, torch_device
):
    import attrs
    from tmol.kinematics._fold_forest import FoldForest
    from tmol.kinematics._move_map import MoveMap
    from tmol.optimization._minimizers import _build_kinforest_network

    sfxn = beta2016_score_function(torch_device)
    ff = FoldForest.reasonable_fold_forest(pose)
    mm = MoveMap.from_pose_stack(pose)
    mask = _movable_block_mask(pose, [5])
    mm.set_move_all_sc_tors_for_blocks(mask.bool())
    net = _build_kinforest_network(pose, sfxn, ff, mm, movable_blocks=mask)
    assert net._reset(sfxn, pose, ff, mm, movable_blocks=mask)
    assert not net._reset(
        sfxn, pose, ff, mm, movable_blocks=_movable_block_mask(pose, [6])
    )
    coords = pose.coords.clone()
    coords[_atom_mask_for_blocks(pose, [17])] += 0.7
    assert not net._reset(
        sfxn, attrs.evolve(pose, coords=coords), ff, mm, movable_blocks=mask
    )
