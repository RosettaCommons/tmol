"""Independent retained-choice/layout and coordinate-derivative invariants."""

import numpy as np
import pytest
import torch

from tmol.pack.rotamer._conformer_compiled import (
    plan_pose_imposition,
    select_pose_rotamers,
)
from tmol_core.pack.pose_imposition import impose_pdb_atom_metadata


def _inputs(device):
    values = [
        ([[0, 1, -1], [2, 0, 1]], torch.int64),
        ([[0, -1, -1], [-1, 4, -1]], torch.int64),
        ([1, 2], torch.int64),
        ([[0, -1], [2, 4]], torch.int64),
        ([1, 2, 3, 5, 6, 7], torch.int64),
        ([[1, 123], [0, 1]], torch.int32),
        ([0, 1, 2, 2, 0, 1, 1, 0], torch.int64),
        ([2, 3, 4], torch.int32),
        ([0, 2, 5, 9, 13, 15, 18, 21], torch.int32),
        ([[1, 5, -1], [0, 6, 10]], torch.int32),
    ]
    return [torch.tensor(value, dtype=dtype, device=device) for value, dtype in values]


@pytest.mark.parametrize("change_types", [False, True])
def test_pose_imposition_layout_and_coordinate_ad(torch_device, change_types):
    if torch_device.type not in ("cpu", "cuda"):
        pytest.skip("native planning metadata is staged on CPU on other devices")
    a = _inputs(torch_device)
    if not change_types:
        a[6][2], a[6][7] = 1, 1
    selected, types, offsets, source, target, metadata = plan_pose_imposition(a, 16, 24)
    assert selected.tolist() == [[0, 2, -1], [3, 4, 7]]
    torch.testing.assert_close(select_pose_rotamers(a[:6]), selected)
    if change_types:
        assert metadata.tolist() == [8, 1]
        assert types.tolist() == [[0, 2, -1], [2, 0, 0]]
        assert offsets.tolist() == [[0, 2, 6], [0, 4, 6]]
        assert source.tolist() == [0, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 21, 22]
        assert target.tolist() == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]
    else:
        assert metadata.tolist() == [16, 0]
        assert torch.equal(offsets, a[9])
        assert target.tolist() == [1, 2, 5, 6, 7, 16, 17, 18, 19, 22, 23, 26, 27, 28]
    # Copying selected physical atoms is linear and leaves unused atoms at zero.
    coords = (
        torch.arange(72, device=torch_device, dtype=torch.float64)
        .reshape(24, 3)
        .requires_grad_()
    )
    out = torch.zeros(
        (2 * metadata[0].item(), 3), dtype=coords.dtype, device=torch_device
    )
    out[target] = coords[source]
    grad = torch.autograd.grad(out.square().sum(), coords, create_graph=True)[0]
    expected = torch.zeros_like(coords)
    expected[source] = 2 * coords[source]
    torch.testing.assert_close(grad, expected, atol=0, rtol=0)
    hvp = torch.autograd.grad(grad.sum(), coords)[0]
    expected_hvp = torch.zeros_like(coords)
    expected_hvp[source] = 2
    torch.testing.assert_close(hvp, expected_hvp, atol=0, rtol=0)


@pytest.mark.parametrize(
    "field,index,value",
    [
        (2, (0,), 2),
        (5, (0, 0), -1),
        (5, (0, 0), 100),
        (4, (1,), -1),
        (4, (1,), 100),
        (6, (2,), 100),
        (8, (2,), 100),
        (7, (2,), -1),
    ],
)
def test_pose_imposition_rejects_invalid_indices(field, index, value):
    a = _inputs("cpu")
    a[field][index] = value
    with pytest.raises(RuntimeError, match="Invalid pose imposition"):
        plan_pose_imposition(a, 16, 24)


@pytest.mark.parametrize("poses,blocks", [(0, 0), (0, 3), (2, 0)])
def test_pose_imposition_empty_axes(poses, blocks):
    a = _inputs("cpu")
    a[0] = torch.empty((poses, blocks), dtype=torch.int64)
    a[1] = torch.empty_like(a[0])
    a[2] = torch.zeros(poses, dtype=torch.int64)
    a[3] = torch.empty((poses, 0), dtype=torch.int64)
    a[4] = torch.empty(0, dtype=torch.int64)
    a[5] = torch.empty((poses, 0), dtype=torch.int32)
    a[9] = torch.empty((poses, blocks), dtype=torch.int32)
    selected, types, offsets, source, target, metadata = plan_pose_imposition(a, 7, 24)
    assert selected.shape == types.shape == offsets.shape == (poses, blocks)
    assert source.numel() == target.numel() == 0
    assert metadata.tolist() == [7, 0]


def test_designed_residue_pdb_metadata_defaults_and_padding():
    original = np.array([[0, 1, -1], [2, 0, 1]])
    new = np.array([[0, 2, -1], [2, 0, 0]])
    old_offsets = np.array([[1, 5, -1], [0, 6, 10]])
    new_offsets = np.array([[0, 2, 6], [0, 4, 6]])
    occ = np.arange(32, dtype=np.float32).reshape(2, 16) / 32
    b = 40 + np.arange(32, dtype=np.float32).reshape(2, 16)
    new_occ, new_b = impose_pdb_atom_metadata(
        original,
        new,
        old_offsets,
        new_offsets,
        np.array([2, 3, 4]),
        occ,
        b,
        8,
    )
    np.testing.assert_array_equal(new_occ[0], [occ[0, 1], occ[0, 2], 1, 1, 1, 1, 0, 0])
    np.testing.assert_array_equal(new_b[0], [41, 42, 30, 30, 30, 30, 0, 0])
    np.testing.assert_array_equal(new_b[1], [56, 57, 58, 59, 62, 63, 30, 30])
