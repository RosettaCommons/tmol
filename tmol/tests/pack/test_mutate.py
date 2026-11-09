"""Public mutation preserves frozen identities, coordinates and input ownership."""

import pytest
import torch

from tmol import (
    beta2016_score_function,
    extended_pose_stack_from_sequences,
    get_sequence,
    mutate_residue,
    mutate_to_sequence,
)


def test_mutation_noop_and_invalid_sequence(torch_device):
    pose = extended_pose_stack_from_sequences(["AKG", "A"], device=torch_device)
    assert mutate_to_sequence(pose, 0, "AKG") is pose
    with pytest.raises(ValueError, match="3 real blocks"):
        mutate_to_sequence(pose, 0, "AK")
    with pytest.raises(ValueError, match="valid one-letter"):
        mutate_to_sequence(pose, 0, "AXG")


def test_mutation_preserves_ligand_other_pose_and_input(torch_device):
    pose, context = extended_pose_stack_from_sequences(
        ["AKG:X(CCO)", "GG"], device=torch_device, return_context=True
    )
    original = pose.coords.clone()
    score = beta2016_score_function(torch_device, param_db=context.parameter_database)
    result = mutate_residue(pose, 0, 1, "ALA", score, seed=4)
    assert get_sequence(result, 0) == "AAGX"
    assert get_sequence(result, 1) == "GG"
    assert torch.equal(pose.coords, original)
    for p, blocks in ((0, [0, 2, 3]), (1, [0, 1])):
        for b in blocks:
            bt = pose.block_type(p, b)
            assert result.block_type(p, b).name == bt.name
            before = int(pose.block_coord_offset64[p, b])
            after = int(result.block_coord_offset64[p, b])
            torch.testing.assert_close(
                result.coords[p, after : after + bt.n_atoms],
                original[p, before : before + bt.n_atoms],
                rtol=0,
                atol=0,
            )
