"""Gradients through atom37 loading are good enough to minimize with.

The same structure, loaded with optH, is minimized two ways with the same
torch optimizer:

1. on the coordinates of the PoseStack loaded from atom37, and
2. on the atom37 coordinates themselves, reloading (optH included) and
   rescoring the pose on every evaluation, so that the gradients come back
   through loading into the original atoms.

If loading backpropagates correctly, the second minimization reaches nearly
the energy of the first. If gradients through loading are wrong, the line
search stalls near the starting energy.
"""

import os

import torch

from tmol import beta2016_score_function
from tmol.io import ATOMWORKS_ATOM37_NAMES, pose_stack_from_canonical_aa_atom37
from tmol.io._pose_stack_from_atomworks import ATOMWORKS_NAME3S

_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "atomworks", "ubq_atomworks.pt"
)


def _lbfgs(x, energy, max_iter):
    optimizer = torch.optim.LBFGS(
        [x],
        lr=1,
        max_iter=max_iter,
        history_size=50,
        line_search_fn="strong_wolfe",
        tolerance_grad=1e-5,
        tolerance_change=1e-7,
    )

    def closure():
        optimizer.zero_grad()
        total = energy(x)
        total.backward()
        assert torch.isfinite(total)
        assert torch.isfinite(x.grad).all()
        return total

    optimizer.step(closure)
    with torch.no_grad():
        return float(energy(x))


def _used_atom37_slots(atom37, residue_type, chain_iid):
    """Given atom37 slots that loading reads: OXT only at a chain's C-terminus."""
    used = (atom37 != 0).any(-1)
    n_res = residue_type.shape[1]
    for r in range(n_res):
        names = ATOMWORKS_ATOM37_NAMES[ATOMWORKS_NAME3S[int(residue_type[0, r])]]
        c_terminal = r == n_res - 1 or chain_iid[0, r] != chain_iid[0, r + 1]
        if "OXT" in names and not c_terminal:
            used[0, r, names.index("OXT")] = False
    return used


def _load_and_minimize_both_ways(atom37, residue_type, chain_iid, no_optH):
    """(start, minimized pose coords, minimized atom37) energies, and the grad
    of the starting energy with respect to atom37."""
    device = atom37.device
    max_iter = 100

    def load(coords):
        return pose_stack_from_canonical_aa_atom37(
            coords, residue_type, chain_iid, no_optH=no_optH, packer_seed=0
        )

    sfxn = beta2016_score_function(device)
    pose = load(atom37)
    # Loading atom37 neither flips nor retypes, so one rendering scores every
    # pose loaded below.
    scorer = sfxn.render_whole_pose_scoring_module(pose)

    def through_loading(x):
        loaded = load(x)
        assert torch.equal(loaded.block_type_ind, pose.block_type_ind)
        return scorer(loaded.coords).sum()

    atom37_coords = atom37.clone().requires_grad_(True)
    (grad,) = torch.autograd.grad(through_loading(atom37_coords), atom37_coords)

    # 1. minimize the loaded pose's coordinates
    pose_coords = pose.coords.detach().clone().requires_grad_(True)
    with torch.no_grad():
        start = float(scorer(pose_coords).sum())
    pose_min = _lbfgs(pose_coords, lambda x: scorer(x).sum(), max_iter)

    # 2. minimize the atom37 coordinates, through loading
    with torch.no_grad():
        torch.testing.assert_close(
            float(through_loading(atom37_coords)), start, rtol=1e-5, atol=1e-3
        )
    atom37_min = _lbfgs(atom37_coords, through_loading, max_iter)

    return start, pose_min, atom37_min, grad


def test_minimizing_atom37_through_loading_matches_minimizing_the_pose(
    torch_device,
):
    data = torch.load(_FIXTURE_PATH, weights_only=True)
    atom37 = data["coords"].to(torch_device)
    residue_type = data["residue_type"].to(torch_device)
    chain_iid = data["chain_iid"].to(torch_device)

    start, pose_min, atom37_min, grad = _load_and_minimize_both_ways(
        atom37, residue_type, chain_iid, no_optH=False
    )
    off_start, off_pose_min, off_atom37_min, _ = _load_and_minimize_both_ways(
        atom37, residue_type, chain_iid, no_optH=True
    )
    report = (
        "energies (optH on / off): "
        f"start {start:.2f} / {off_start:.2f}, "
        f"minimized pose coords {pose_min:.2f} / {off_pose_min:.2f}, "
        f"minimized atom37 through loading {atom37_min:.2f} / {off_atom37_min:.2f}"
    )

    # the score backpropagates into every atom37 atom loading uses, and only those
    torch.testing.assert_close(
        (grad != 0).any(-1), _used_atom37_slots(atom37, residue_type, chain_iid)
    )

    assert pose_min < start, report

    # Minimizing pose coordinates moves every hydrogen freely, with or without
    # optH. Minimizing atom37 with optH, the gradients reach the original atoms
    # and optH re-places polar hydrogens, so it ends as low (on 1UBQ, all three
    # within ~2.4 after 100 iterations).
    finals = torch.tensor([atom37_min, pose_min, off_pose_min])
    assert finals.max() - finals.min() < 5.0, report

    # Without optH, the atom37 scheme can turn a polar hydrogen only by moving
    # the heavy atoms it is built from (on 1UBQ, it ends ~36 higher).
    assert off_atom37_min - atom37_min > 30.0, report
