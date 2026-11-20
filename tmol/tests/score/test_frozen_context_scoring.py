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
