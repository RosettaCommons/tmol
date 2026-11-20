"""The per-atom grid kernel on Metal, and a CPU scorer offloading to it."""

from __future__ import annotations

import pytest
import torch

from tmol.score._mps_offload import mps_offload_enabled
from tmol.score.gridscore import GridScorer, build_grid
from tmol.score.gridscore.grid import (
    OUT_OF_BOUNDS_PENALTY,
    probe_atom_types_from_name3,
)
from tmol.score.gridscore.potentials.compiled import grid_score_atoms
from tmol.tests.ligand.test_protein_ligand_ddg import LIGAND_RES_NAME
from tmol.tests.score.gridscore.test_gridscore import (
    _ligand_scoring_inputs,
    _load_cdk2_pose,
)


@pytest.fixture
def offload(monkeypatch):
    monkeypatch.setenv("TMOL_MPS_OFFLOAD", "1")
    if not mps_offload_enabled():
        pytest.skip("needs an Apple GPU and a Metal-enabled build")


@pytest.fixture(scope="module")
def cdk2_case():
    pose_stack, param_db = _load_cdk2_pose()
    probe_names = probe_atom_types_from_name3(pose_stack, LIGAND_RES_NAME)
    grids = build_grid(
        pose_stack, probe_names, param_db, ligand_name3=LIGAND_RES_NAME
    ).to(torch.device("cpu"), dtype=torch.float32)
    base_coords, atom_types, charges = _ligand_scoring_inputs(pose_stack)
    generator = torch.Generator().manual_seed(0)
    base = base_coords.to(torch.float32)
    coords = base[None] + 1.0 * torch.randn((64, *base.shape), generator=generator)
    # One atom far off the grid exercises the out-of-bounds penalty.
    coords[0, 0] += 1.0e3
    return grids, atom_types, charges.to(torch.float32), coords


def test_grid_score_atoms_on_metal_matches_cpu(offload, cdk2_case):
    grids, atom_types, charges, coords = cdk2_case
    scorer = GridScorer(grids, atom_types, charges, device="cpu", dtype=torch.float32)
    g = scorer.grids
    host_args = (
        coords,
        scorer._grid_atom_index,
        scorer._grid_index,
        scorer._charges,
        g.fa_atr_coeffs,
        g.fa_rep_coeffs,
        g.fa_sol_coeffs,
        g.fa_elec_coeffs[None],
        scorer._lk_heavy_coeffs,
        g.geometry.origin,
        g.geometry.voxel_spacing,
        scorer._weights,
        OUT_OF_BOUNDS_PENALTY,
    )
    metal_args = tuple(
        a.to("mps") if isinstance(a, torch.Tensor) else a for a in host_args
    )
    host_energy, host_grad = grid_score_atoms(*host_args)
    metal_energy, metal_grad = grid_score_atoms(*metal_args)
    torch.testing.assert_close(metal_energy.cpu(), host_energy, rtol=1e-5, atol=1e-4)
    torch.testing.assert_close(metal_grad.cpu(), host_grad, rtol=1e-5, atol=1e-4)


def test_cpu_grid_scorer_offloads_and_matches_host(offload, monkeypatch, cdk2_case):
    grids, atom_types, charges, coords = cdk2_case
    offloaded = GridScorer(
        grids, atom_types, charges, device="cpu", dtype=torch.float32
    )
    monkeypatch.setenv("TMOL_MPS_OFFLOAD", "0")
    host = GridScorer(grids, atom_types, charges, device="cpu", dtype=torch.float32)
    assert offloaded._mps is not None and host._mps is None

    for scale in (1.0, 0.1):
        # The repulsive ramp must reach the GPU copy of the weights.
        offloaded.set_rep_scale(scale)
        host.set_rep_scale(scale)
        e_off, g_off = offloaded.value_and_grad(coords)
        e_host, g_host = host.value_and_grad(coords)
        assert e_off.device.type == "cpu" and g_off.device.type == "cpu"
        torch.testing.assert_close(e_off, e_host, rtol=1e-5, atol=1e-3)
        torch.testing.assert_close(g_off, g_host, rtol=1e-5, atol=1e-3)
