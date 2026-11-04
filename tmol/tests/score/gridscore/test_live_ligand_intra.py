"""Correctness and derivative gates for live ligand-intra grid scoring."""

from __future__ import annotations

import attrs
import pytest
import torch

from tmol.score import beta2016_score_function
from tmol.score.gridscore.live_ligand_intra import (
    build_live_ligand_intra_data,
    score_ligand_intra,
    score_ligand_intra_with_gradients,
)
from tmol.tests.ligand.test_ligand_receptor_perterm_baseline import (
    _GOLDEN_PER_TERM,
    _TARGETS,
    _TERM_ORDER,
)
from tmol.tests.score.gridscore.test_gridscore import _load_target_pose
from tmol.tests.score.gridscore.test_live_terms import (
    _ligand_block,
    _ligand_coords,
)
from tmol.pose import repeat_pose_stack
from tmol.tests.dock.test_finalization import _flexible_sidechains

_BASE_INTRA_TERMS = (
    "fa_ljatr",
    "fa_ljrep",
    "fa_lk",
    "fa_elec",
    "gen_torsions",
)
_CART_INTRA_TERMS = (
    "cart_lengths",
    "cart_angles",
    "cart_torsions",
    "cart_impropers",
    "cart_hxltorsions",
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
def test_flexible_context_intra_cuda_graph_replays_values_and_gradients():
    reference, param_db = _load_target_pose("cdk2")
    ligand_block = _ligand_block(reference)
    data = build_live_ligand_intra_data(
        reference,
        ligand_block,
        param_db,
        weight_profile="search",
        sidechain_blocks=_flexible_sidechains(reference, ligand_block),
    ).to("cuda", torch.float32)
    assert data.context_block_indices
    coords = data.ligand_pose_stack.coords.repeat(2, 1, 1)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            score_ligand_intra_with_gradients(data, coords)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        energy, gradient = score_ligand_intra_with_gradients(data, coords)
    for displacement in (0.0, 0.01):
        coords[:, 0, 0].add_(displacement)
        expected = score_ligand_intra_with_gradients(data, coords)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.isfinite(energy).all() and torch.isfinite(gradient).all()
        torch.testing.assert_close(energy, expected[0])
        torch.testing.assert_close(gradient, expected[1])


def _perturbed_ligand_coords(pose_stack, ligand_block: int) -> torch.Tensor:
    base = _ligand_coords(pose_stack, ligand_block)[0]
    atom = torch.arange(base.shape[0], dtype=base.dtype)[:, None]
    axes = torch.arange(3, dtype=base.dtype)[None, :]
    delta = 0.035 * torch.sin(0.7 * atom + 1.3 * axes)
    return torch.stack((base, base + delta, base - 0.6 * delta))


def _full_complex_ligand_intra_oracle(
    pose_stack,
    ligand_block: int,
    param_db,
    ligand_coords: torch.Tensor,
    terms: tuple[str, ...],
) -> torch.Tensor:
    population = repeat_pose_stack(pose_stack, ligand_coords.shape[0])
    pose_indices = torch.arange(population.n_poses, dtype=torch.int64)
    type_index = int(population.block_type_ind64[0, ligand_block])
    n_atoms = population.packed_block_types.active_block_types[type_index].n_atoms
    atom_indices = (
        population.block_coord_offset64[:, ligand_block, None]
        + torch.arange(n_atoms, dtype=torch.int64)[None, :]
    )
    coords = population.coords.clone()
    coords[pose_indices[:, None], atom_indices] = ligand_coords
    population = attrs.evolve(population, coords=coords)
    score_function = beta2016_score_function(torch.device("cpu"), param_db=param_db)
    scorer = score_function.render_block_pair_scoring_module(population)
    per_term = scorer(population.coords, sum_terms=False)
    term_order = tuple(
        score_type.name for score_type in score_function.all_score_types()
    )
    assert term_order == _TERM_ORDER
    term_indices = [term_order.index(term) for term in terms]
    return per_term[term_indices, :, ligand_block, ligand_block].sum(dim=0)


@pytest.mark.parametrize("target", _TARGETS)
def test_full_live_ligand_intra_matches_exact_oracle_native_and_perturbed(target):
    pose_stack, param_db = _load_target_pose(target)
    ligand_block = _ligand_block(pose_stack)
    data = build_live_ligand_intra_data(
        pose_stack, ligand_block, param_db, weight_profile="full"
    )
    ligand_coords = _perturbed_ligand_coords(pose_stack, ligand_block)
    # All five committed ligand parameter files contain bonds explicitly marked
    # as ring bonds, independently pinning the cart-bonded inclusion decision.
    assert data.include_cart_bonded
    terms = _BASE_INTRA_TERMS + _CART_INTRA_TERMS

    actual = score_ligand_intra(data, ligand_coords)
    padded = torch.empty((*ligand_coords.shape[:2], 6), dtype=ligand_coords.dtype)
    noncontiguous = padded[..., ::2]
    noncontiguous.copy_(ligand_coords)
    assert not noncontiguous.is_contiguous()
    noncontiguous_actual = score_ligand_intra(data, noncontiguous)
    expected = _full_complex_ligand_intra_oracle(
        pose_stack,
        ligand_block,
        param_db,
        ligand_coords,
        terms,
    )
    golden_native = sum(
        _GOLDEN_PER_TERM[target]["ligand_intra"][_TERM_ORDER.index(term)]
        for term in terms
    )
    oracle_error = torch.max(torch.abs(actual - expected))
    golden_error = abs(float(actual[0]) - golden_native)
    print(
        f"{target} ligand-intra: max live/exact error={float(oracle_error):.3e} "
        f"native golden error={golden_error:.3e}"
    )
    torch.testing.assert_close(noncontiguous_actual, actual, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=1.0e-4)
    torch.testing.assert_close(
        actual[0],
        torch.tensor(golden_native, dtype=torch.float64),
        rtol=0.0,
        atol=1.0e-4,
    )
