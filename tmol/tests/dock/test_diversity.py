"""Energy-ranked RMS-diverse selection."""

import math

import pytest
import torch

from tmol.dock._diversity import select_diverse_low_energy


def _sequential_selection(energies, coords, npool, rmsdthreshold):
    """Greedy walk down the energy ranking, one candidate at a time."""
    n_candidates = energies.shape[0]
    order = torch.argsort(energies, stable=True)
    flat = coords[order].reshape(n_candidates, -1)
    distance = torch.cdist(flat, flat, compute_mode="donot_use_mm_for_euclid_dist")
    far_enough = (distance / math.sqrt(coords.shape[1]) >= rmsdthreshold).cpu().numpy()
    kept: list[int] = []
    for rank in range(n_candidates):
        if len(kept) == npool:
            break
        if far_enough[rank, kept].all():
            kept.append(rank)
    skipped = set(kept)
    fill = [r for r in range(n_candidates) if r not in skipped][: npool - len(kept)]
    return order[torch.tensor(kept + fill, device=energies.device)], len(kept)


# 0.5 keeps nearly everything, 12.0 keeps almost nothing: the second exercises
# the fill path, where fewer than npool candidates satisfy the cutoff.
@pytest.mark.parametrize("rmsdthreshold", [0.5, 2.0, 6.0, 12.0])
@pytest.mark.parametrize("n_candidates,npool", [(400, 200), (257, 128)])
def test_matches_sequential_selection(torch_device, rmsdthreshold, n_candidates, npool):
    torch.manual_seed(1)
    coords = 3.0 * torch.randn(
        (n_candidates, 30, 3), dtype=torch.float32, device=torch_device
    )
    energies = torch.randn(n_candidates, dtype=torch.float32, device=torch_device)

    selected, n_diverse = select_diverse_low_energy(
        energies, coords, npool=npool, rmsdthreshold=rmsdthreshold
    )
    expected, expected_diverse = _sequential_selection(
        energies, coords, npool, rmsdthreshold
    )
    assert int(n_diverse) == expected_diverse
    assert selected.shape == (npool,)
    torch.testing.assert_close(selected, expected, rtol=0.0, atol=0.0)


def test_rejects_npool_above_candidate_count(torch_device):
    coords = torch.randn((4, 30, 3), dtype=torch.float32, device=torch_device)
    energies = torch.randn(4, dtype=torch.float32, device=torch_device)
    with pytest.raises(ValueError):
        select_diverse_low_energy(energies, coords, npool=5, rmsdthreshold=1.0)
