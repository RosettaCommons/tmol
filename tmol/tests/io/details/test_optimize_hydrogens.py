import copy

import torch

from tmol import beta2016_score_function, pose_stack_from_atom_array
from tmol.io.details._build_missing_leaf_atoms import _build_coords_from_icoors
from tmol.io.details._optimize_hydrogens import (
    _block_atoms,
    _optimize_hydrogens,
    _resolve_uaids,
)
from tmol.numeric import coord_dihedrals
from tmol.pack._build_missing_sidechains import _build_missing_sidechains


def _atoms_named(pose, res_name, atom_name):
    """(pose, block, block atom, pose atom) of each atom_name in a res_name block."""
    pbt = pose.packed_block_types
    rows = []
    for b, t in enumerate(pose.block_type_ind[0].tolist()):
        if t < 0:
            continue
        bt = pbt.active_block_types[t]
        if bt.base_name == res_name and atom_name in bt.atom_to_idx:
            a = bt.atom_to_idx[atom_name]
            rows.append((0, b, a, int(pose.block_coord_offset[0, b]) + a))
    return torch.tensor(rows, dtype=torch.int64, device=pose.device).unbind(1)


def _is_hydrogen(pose):
    """Whether each real atom of a pose, in pose order, is a hydrogen."""
    p, b, a = _block_atoms(pose)
    return pose.packed_block_types.atom_is_hydrogen[
        pose.block_type_ind64[p, b], a
    ].bool()


def _primary_frame_dihedral(pose, coords, p, b, a, i):
    uaids = pose.packed_block_types.build_missing_leaf_atom_icoor_ann.anc_uaids
    types = pose.block_type_ind64[p, b]
    frame = [_resolve_uaids(pose, p, b, uaids[types, a, k]) for k in range(3)]
    coords = coords.double()
    return coord_dihedrals(
        coords[p, frame[2]], coords[p, frame[1]], coords[p, frame[0]], coords[p, i]
    ).float()


def test_dihedrals_replace_stored_phi(biotite_1ubq, torch_device):
    pose = pose_stack_from_atom_array(biotite_1ubq, torch_device, no_optH=True)
    p, b, a, i = _atoms_named(pose, "SER", "HG")
    pbt = pose.packed_block_types

    coords = pose.coords.clone()
    coords[p, i] = torch.nan
    targets = torch.zeros(
        (*pose.block_type_ind.shape, pbt.max_n_atoms),
        dtype=torch.bool,
        device=torch_device,
    )
    targets[p, b, a] = True

    def build(dihedrals=None):
        return _build_coords_from_icoors(
            pbt,
            coords,
            targets,
            torch.isnan(coords).any(-1),
            pose.block_coord_offset,
            pose.block_type_ind,
            pose.inter_residue_connections,
            dihedrals=dihedrals,
        )

    default = build()
    unchanged = torch.full_like(coords[..., 0], torch.nan)
    torch.testing.assert_close(build(unchanged), default)

    wanted = torch.linspace(-3.0, 3.0, i.shape[0], device=torch_device)
    dihedrals = unchanged.clone()
    dihedrals[p, i] = wanted
    built = build(dihedrals)
    torch.testing.assert_close(
        _primary_frame_dihedral(pose, built, p, b, a, i), wanted, atol=1e-4, rtol=0
    )
    # every other atom is copied, and the bond to the parent keeps its length
    others = torch.ones_like(built[..., 0], dtype=torch.bool)
    others[p, i] = False
    torch.testing.assert_close(built[others], default[others], equal_nan=True)
    og = _atoms_named(pose, "SER", "OG")[3]
    torch.testing.assert_close(
        (built[p, i] - built[p, og]).norm(dim=-1),
        (default[p, i] - default[p, og]).norm(dim=-1),
    )


def test_optimized_hydrogens_match_packing_and_carry_gradients(
    biotite_1ubq, torch_device
):
    pose = pose_stack_from_atom_array(biotite_1ubq, torch_device, no_optH=True)
    sfxn = beta2016_score_function(torch_device)
    packed = _build_missing_sidechains(
        pose,
        sfxn,
        None,
        torch.zeros_like(pose.block_type_ind64, dtype=torch.bool),
        no_optH=False,
        has_missing_atoms=False,
        seed=0,
    )

    coords = pose.coords.detach().clone().requires_grad_(True)
    pose = copy.copy(pose)
    pose.coords = coords
    optimized = _optimize_hydrogens(pose, sfxn, seed=0)

    real = optimized.real_atoms
    torch.testing.assert_close(
        optimized.coords[real], packed.coords[real], atol=1e-3, rtol=0
    )
    # no NHQ flips: types (so histidine tautomers) and every heavy atom stay put
    torch.testing.assert_close(optimized.block_type_ind, pose.block_type_ind)
    is_h = _is_hydrogen(optimized)
    torch.testing.assert_close(
        optimized.coords[real][~is_h], coords.detach()[real][~is_h], rtol=0, atol=0
    )

    # a hydroxyl hydrogen optH turned follows the atoms it is built from
    hg = _atoms_named(optimized, "SER", "HG")[3]
    og = _atoms_named(optimized, "SER", "OG")[3]
    turned = (optimized.coords[0, hg] - coords[0, hg]).norm(dim=-1) > 1e-3
    assert torch.any(turned)
    (grad,) = torch.autograd.grad(
        optimized.coords[0, hg[turned]].sum(), coords, allow_unused=True
    )
    assert torch.count_nonzero(grad[0, hg[turned]]) == 0
    assert torch.all(grad[0, og[turned]].abs().sum(-1) > 0)


def test_construction_optimizes_hydrogens_differentiably(biotite_1ubq, torch_device):
    from tmol.io import build_context_from_atom_array, canonical_form_from_biotite
    from tmol.io._pose_stack_construction import _pose_stack_from_canonical_form

    context = build_context_from_atom_array(biotite_1ubq, torch_device)
    cf = canonical_form_from_biotite(
        biotite_1ubq, torch_device, co=context.canonical_ordering
    )
    coords = cf.coords.clone().requires_grad_(True)
    cf.coords = coords

    pose, details = _pose_stack_from_canonical_form(
        context.canonical_ordering,
        context.packed_block_types,
        *cf,
        return_atom_mapping=True,
        opth_score_function=context._opth_score_function,
        packer_seed=0,
    )
    assert torch.isfinite(pose.coords[pose.real_atoms]).all()
    pose.coords[pose.real_atoms].sum().backward()
    assert torch.count_nonzero(coords.grad) > 0

    # the mapping points each input atom at the pose atom it became
    can = details["can_atom_mapping"]
    ps = details["ps_atom_mapping"]
    source = cf.coords[can[:, 0], can[:, 1], can[:, 2]]
    given = torch.isfinite(source).all(-1)
    moved = (pose.coords[ps[:, 0], ps[:, 1]] - source).norm(dim=-1) > 1e-3
    assert torch.count_nonzero(given & ~moved) > 0.9 * torch.count_nonzero(given)


def test_minimized_hydrogens_lower_the_energy_and_keep_gradients(
    biotite_1ubq, torch_device
):
    pose = pose_stack_from_atom_array(biotite_1ubq, torch_device, no_optH=True)
    sfxn = beta2016_score_function(torch_device)
    coords = pose.coords.detach().clone().requires_grad_(True)
    pose = copy.copy(pose)
    pose.coords = coords

    sampled = _optimize_hydrogens(pose, sfxn, seed=0)
    minimized = _optimize_hydrogens(pose, sfxn, seed=0, minimize=True)

    def energy(p):
        return sfxn.render_whole_pose_scoring_module(p)(p.coords.detach()).sum()

    assert energy(minimized) < energy(sampled)
    torch.testing.assert_close(minimized.block_type_ind, sampled.block_type_ind)
    real = minimized.real_atoms
    is_h = _is_hydrogen(minimized)
    shift = (minimized.coords[real] - sampled.coords[real]).norm(dim=-1)
    assert torch.any(shift[is_h] > 1e-2)
    torch.testing.assert_close(
        minimized.coords[real][~is_h], sampled.coords[real][~is_h]
    )

    minimized.coords[real].sum().backward()
    assert torch.isfinite(coords.grad).all()
    assert torch.count_nonzero(coords.grad) > 0


def test_loader_minimizes_optimized_hydrogens_on_request(biotite_1ubq, torch_device):
    # the input's hydrogens would otherwise be kept as given
    sfxn = beta2016_score_function(torch_device)
    sampled = pose_stack_from_atom_array(
        biotite_1ubq, torch_device, packer_seed=0, rebuild_hydrogens=True
    )
    minimized = pose_stack_from_atom_array(
        biotite_1ubq,
        torch_device,
        packer_seed=0,
        rebuild_hydrogens=True,
        optH_minimize=True,
    )
    for p in (sampled, minimized):
        assert torch.isfinite(p.coords[p.real_atoms]).all()
    assert sfxn.render_whole_pose_scoring_module(minimized)(minimized.coords) < (
        sfxn.render_whole_pose_scoring_module(sampled)(sampled.coords)
    )


def test_input_hydrogens_are_kept_unless_rebuilt(biotite_1ubq, torch_device):
    # the 1ubq fixture gives ideal hydrogens, proton-chi ones included; shift
    # them off ideal so keeping and rebuilding them differ
    structure = biotite_1ubq.copy()
    hydrogens = structure.element == "H"
    assert hydrogens.sum() > 0
    structure.coord[hydrogens] += [0.1, 0.0, 0.0]
    sfxn = beta2016_score_function(torch_device)

    def load(**kwargs):
        return pose_stack_from_atom_array(
            structure, torch_device, packer_seed=0, **kwargs
        )

    def energy(p):
        return sfxn.render_whole_pose_scoring_module(p)(p.coords).sum()

    given = load(no_optH=True)
    kept = load()
    torch.testing.assert_close(kept.block_type_ind, given.block_type_ind)
    torch.testing.assert_close(kept.coords, given.coords, rtol=0, atol=0)

    # rebuilt from ideal geometry: heavy atoms stay, hydrogens move
    ideal = load(no_optH=True, rebuild_hydrogens=True)
    torch.testing.assert_close(ideal.block_type_ind, given.block_type_ind)
    real = given.real_atoms
    is_h = _is_hydrogen(given)
    torch.testing.assert_close(
        ideal.coords[real][~is_h], given.coords[real][~is_h], rtol=0, atol=0
    )
    assert torch.any(
        (ideal.coords[real][is_h] - given.coords[real][is_h]).norm(dim=-1) > 0.05
    )

    # and then optimized
    optimized = load(rebuild_hydrogens=True)
    assert energy(optimized) < energy(ideal)
    torch.testing.assert_close(
        optimized.coords[real][~is_h], given.coords[real][~is_h], rtol=0, atol=0
    )
