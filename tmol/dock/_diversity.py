"""Energy-ranked RMS-diverse selection shared by the aligner and the GA."""

import math

import torch


def ligand_heavy_atom_rms(coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
    """Unsymmetrized heavy-atom coordinate RMS with broadcasting."""
    delta = coords1 - coords2
    return torch.sqrt(torch.mean(torch.sum(delta * delta, dim=-1), dim=-1))


def _greedy_diverse_mask(too_close: torch.Tensor) -> torch.Tensor:
    """Greedy independent set over a conflict matrix, in rank order.

    Accepting every undecided candidate that no better-ranked undecided
    candidate conflicts with, then rejecting that candidate's neighbors, picks
    the same set as a sequential walk down the ranking: the priority order is
    fixed, so a candidate's fate never depends on the order decisions are made.
    """
    n_candidates = too_close.shape[0]
    device = too_close.device
    rank = torch.arange(n_candidates, device=device)
    blocked_by = too_close & (rank[:, None] > rank[None, :])

    undecided = torch.ones(n_candidates, dtype=torch.bool, device=device)
    kept = torch.zeros(n_candidates, dtype=torch.bool, device=device)
    # Each round accepts at least the best undecided candidate, so the loop
    # cannot run more times than there are candidates.
    for _ in range(n_candidates):
        if not bool(undecided.any()):
            break
        accepted = undecided & ~(blocked_by & undecided[None, :]).any(dim=1)
        kept |= accepted
        undecided &= ~accepted
        undecided &= ~(too_close & accepted[None, :]).any(dim=1)
    return kept


def select_diverse_low_energy(
    energies: torch.Tensor,
    coords: torch.Tensor,
    *,
    npool: int,
    rmsdthreshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select energy-ranked RMS niches, then fill from rejected candidates."""
    n_candidates = energies.shape[0]
    if npool > n_candidates:
        raise ValueError("npool cannot exceed the number of candidates")

    order = torch.argsort(energies, stable=True)
    if rmsdthreshold == 0 and torch.isfinite(coords).all():
        selected = order[:npool]
        return selected, energies.new_tensor(selected.numel(), dtype=torch.int64)
    flat = coords[order].reshape(n_candidates, -1)
    distance = torch.cdist(flat, flat, compute_mode="donot_use_mm_for_euclid_dist")
    too_close = ~(distance / math.sqrt(coords.shape[1]) >= rmsdthreshold)
    too_close.fill_diagonal_(False)

    kept = _greedy_diverse_mask(too_close)
    kept_ranks = torch.nonzero(kept).flatten()[:npool]
    n_kept = int(kept_ranks.numel())
    if n_kept < npool:
        rejected = torch.nonzero(~kept).flatten()[: npool - n_kept]
        kept_ranks = torch.cat((kept_ranks, rejected))
    return order[kept_ranks], torch.tensor(n_kept, device=energies.device)
