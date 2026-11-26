"""The docking backends are interchangeable.

A caller builds either backend the same way and docks with the same calls; the
only differences are the ones grids inherently carry (building or sharing a
grid set, and aligner initialization, which places the ligand in the grid box).
"""

from __future__ import annotations

import pytest
import torch

from tmol.dock import (
    ContinuousBackend,
    GASchedule,
    GridBackend,
    run_docking,
    select_pocket_residues,
)
from tmol.dock.ga import search_layout
from tmol.dock.scoring_backend import ScoringBackend
from tmol.tests.dock.test_ligand_dof_roundtrip import _ligand_block, _load_cdk2_pose


def _flexible_residue(pose_stack, ligand_block):
    """One pocket residue whose sidechain the search can move."""
    selected = select_pocket_residues(pose_stack, ligand_block, cutoff=6.0)
    for block in selected.detach().cpu().tolist():
        if pose_stack.block_type(0, block).n_sc_torsions:
            return (block,)
    raise AssertionError("cdk2's pocket has no flexible sidechain")


@pytest.fixture(scope="module")
def cdk2_case():
    pose_stack, param_db = _load_cdk2_pose()
    ligand_block = _ligand_block(pose_stack)
    return (
        pose_stack,
        param_db,
        ligand_block,
        _flexible_residue(pose_stack, ligand_block),
    )


@pytest.fixture(scope="module")
def built(cdk2_case):
    """Both backends, built for the same ligand and the same flexible residue."""
    pose_stack, param_db, ligand_block, flexible = cdk2_case
    return {
        "continuous": ContinuousBackend.from_pose_stack(
            pose_stack, ligand_block, param_db, flexible_residues=flexible
        ),
        "grid": GridBackend.from_pose_stack(
            pose_stack, ligand_block, param_db, flexible_residues=flexible
        ),
    }


def test_both_backends_declare_the_same_docking_site(cdk2_case, built):
    """from_pose_stack resolves the ligand, sidechains and metals identically."""
    _, param_db, ligand_block, flexible = cdk2_case

    for name, backend in built.items():
        assert isinstance(backend, ScoringBackend), name
        assert backend.ligand_block == ligand_block, name
        assert backend.sidechain_blocks == flexible, name
        assert backend.param_db is param_db, name
    assert (
        built["continuous"].metal_blocks == built["grid"].metal_blocks
    ), "backends disagree on the reachable metals"


def test_a_backend_may_be_built_from_a_residue_name(cdk2_case):
    """Either backend takes the ligand as a name, index, or atom mask."""
    pose_stack, param_db, ligand_block, _ = cdk2_case
    name3 = pose_stack.block_type(0, ligand_block).name3
    for backend in (ContinuousBackend, GridBackend):
        assert (
            backend.from_pose_stack(pose_stack, name3, param_db).ligand_block
            == ligand_block
        ), backend.__name__


def test_search_layouts_agree_on_what_moves(cdk2_case, built):
    """Each backend searches its own pose; both map onto the same reference."""
    pose_stack, _, ligand_block, _ = cdk2_case
    layouts = {
        name: search_layout(pose_stack, backend) for name, backend in built.items()
    }

    # The grid searches a compact pose, the continuous one the whole complex.
    assert layouts["grid"].pose_stack.max_n_blocks < pose_stack.max_n_blocks
    assert layouts["continuous"].pose_stack.max_n_blocks == pose_stack.max_n_blocks
    assert layouts["grid"].ligand_block == 0
    assert layouts["continuous"].ligand_block == ligand_block

    for name, layout in layouts.items():
        assert layout.reference_ligand_block == ligand_block, name
    # Same reference residues, same moving atoms within them, either way.
    moving = {
        name: tuple(
            (block, tuple(atoms.tolist())) for _, block, atoms in layout.sidechain_atoms
        )
        for name, layout in layouts.items()
    }
    assert moving["continuous"] == moving["grid"]
    assert moving["grid"], "the flexible residue contributed no moving atoms"


def test_backends_run_the_same_flexible_docking_call(cdk2_case, built):
    """The same run_docking call drives either backend, ligand and sidechain."""
    pose_stack, _, ligand_block, flexible = cdk2_case
    schedule = GASchedule(generations=1, npool=4, maxiter=2, rmsdthreshold=0.0)

    results = {}
    for name, backend in built.items():
        results[name] = run_docking(
            pose_stack,
            backend,
            schedule=schedule,
            seed=20261126,
            initialization="jitter",
            finalize=True,
            top_k=1,
        )

    reference = pose_stack.coords[0]
    for name, result in results.items():
        assert result.best_pose_stack.max_n_blocks == pose_stack.max_n_blocks, name
        assert torch.isfinite(result.best_energies).all(), name
        displacement = (result.best_pose_stack.coords[0] - reference).norm(dim=1)

        ligand_start = int(pose_stack.block_coord_offset64[0, ligand_block])
        n_ligand_atoms = pose_stack.block_type(0, ligand_block).n_atoms
        assert torch.any(
            displacement[ligand_start : ligand_start + n_ligand_atoms] > 1.0e-4
        ), f"{name} left the ligand where it started"

        moved = torch.zeros_like(displacement, dtype=torch.bool)
        moved[ligand_start : ligand_start + n_ligand_atoms] = True
        for block in flexible:
            start = int(pose_stack.block_coord_offset64[0, block])
            moved[start : start + pose_stack.block_type(0, block).n_atoms] = True
        assert torch.all(displacement[~moved] < 1.0e-6), f"{name} moved the receptor"

    # Both search the same objective, so they rank the pocket the same way.
    energies = [float(result.best_energies[0]) for result in results.values()]
    assert abs(energies[0] - energies[1]) < 2.0
