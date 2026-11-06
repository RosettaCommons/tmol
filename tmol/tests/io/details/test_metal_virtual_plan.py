"""A fixed cluster fit preserves the metal translation and coordinate derivative."""

from types import SimpleNamespace as S

import numpy as np
import pytest
import torch

from tmol.io.details._metal_detection import _place_cluster_virtuals


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cluster_virtuals_follow_the_metal_translation(dtype):
    names = ("M", "V1", "V2")
    bt = S(
        ideal_coords=np.array([[1.0, 2, 3], [1.125, 2.25, 3.5], [0.5, 2.125, 3.25]]),
        icoors_index=dict(zip(names, range(3))),
        atom_to_idx=dict(zip(names, range(3))),
        atoms=[S(name=name) for name in names],
        metal_sites=[S(metal_atom="M", site_virts=names[1:])],
    )
    x = torch.tensor(
        [[10000.25, 0.1, -0.2], [0.0, 0, 0], [0.0, 0, 0]],
        dtype=dtype,
        requires_grad=True,
    )
    out = x.clone()
    missing = torch.tensor([False, True, True])
    rotation = np.array([[0.0, 1, 0], [-1.0, 0, 0], [0.0, 0, 1]])
    _place_cluster_virtuals(bt, out, missing, (rotation,))
    expected = (
        x[0].detach().double().numpy()
        + (bt.ideal_coords[1:] - bt.ideal_coords[0]) @ rotation
    ).astype(np.float32 if dtype == torch.float32 else np.float64)
    np.testing.assert_array_equal(out[1:].detach().numpy(), expected)
    assert not missing.any()
    weights = torch.tensor([[1.0, 2, 3], [4.0, 5, 6]], dtype=dtype)
    gradient = torch.autograd.grad((out[1:] * weights).sum(), x)[0]
    torch.testing.assert_close(gradient[0], weights.sum(dim=0), rtol=0, atol=0)
    assert torch.count_nonzero(gradient[1:]) == 0
