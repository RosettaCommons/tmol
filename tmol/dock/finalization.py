"""RosettaVS-style exact refinement and binding-score finalization."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import attr
import numpy
import torch

from tmol.dock.ga import (
    GAResult,
    SearchLayout,
    _place_ligand_in_reference,
    search_layout,
)
from tmol.dock.scoring_backend import ScoringBackend
from tmol.dock.ligand_dofs import per_pose_fold_forest_and_move_map
from tmol.optimization._minimizers import (
    CartesianMinimizer,
    KinematicMinimizer,
    run_cart_min,
)
from tmol.kinematics._move_map import CartesianMoveMap
from tmol.relax._fast_relax import (
    CartesianRelaxMinimizer,
    default_relax_task_operation,
    fast_relax,
)
from tmol.io._metal_coordination import redetect_metal_coordination
from tmol.pose._constraint_set import ConstraintSet

from tmol.pose._pose_stack import PoseStack
from tmol.pose._pose_stack_builder import PoseStackBuilder
from tmol.pose import repeat_pose_stack, select_blocks, select_poses
from tmol.database import ParameterDatabase
from tmol.score import beta2016_score_function
from tmol.score.constraint._constraint_energy_term import ConstraintEnergyTerm
from tmol.score._score_types import ScoreType
from tmol.score.gridscore._movable import movable_pose_stack

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RefineProtocol:
    """Exact-stage refinement, in the vocabulary of Rosetta's relax scripts.

    ``steps`` are ``fast_relax`` schedule entries: each repacks the movable
    sidechains at ``fa_rep_pack_frac`` and minimizes at ``fa_rep_min_frac``,
    one entry per Rosetta ``ramp_repack_min`` line. ``cycles`` is its
    ``repeat``, and the best-scoring state across cycles is kept.

    ``sidechains`` chooses what moves: ``"backend"`` takes the set the backend
    declared, ``()`` holds the receptor rigid, and an explicit sequence of
    block indices overrides both -- the relax stage need not use the same
    freedom the search did.
    """

    steps: tuple[dict[str, float], ...]
    cycles: int = 1
    max_iter: int = 50
    top_k: int = 20
    space: str = "cartesian"
    opt_h: bool = False
    backbone_neighbors: int = 0
    sidechains: str | tuple[int, ...] = "backend"
    #: Rebuild the movable sidechains from the rotamer library at each step;
    #: off leaves them at the conformation the search produced.
    repack: bool = False
    #: Redeclare metal coordination from each promoted candidate's own
    #: coordinates before refining it, so a ligand that reached an open site
    #: coordinates it.
    redetect_metal: bool = True

    def ramp(self) -> tuple[tuple[float, float, int], ...]:
        """(fa_rep multiplier, tolerance, iterations) per step."""
        return tuple(
            (
                float(step["fa_rep_min_frac"]),
                float(step.get("min_rtol", 1e-5)),
                self.max_iter,
            )
            for step in self.steps
        )


# Rosetta dockrigid: the default FastRelax script, torsion space, no receptor
# sidechain repacking because its movemap frees none.
RIGID_REFINE = RefineProtocol(
    steps=(
        {"fa_rep_pack_frac": 0.02, "fa_rep_min_frac": 0.02, "min_rtol": 0.01},
        {"fa_rep_pack_frac": 1.0, "fa_rep_min_frac": 1.0, "min_rtol": 1e-5},
    ),
    cycles=3,
    max_iter=50,
    space="torsion",
    sidechains=(),
)

# Rosetta dockflex "bbsc1": one cartesian ramp_repack_min at full fa_rep.
FLEX_REFINE = RefineProtocol(
    steps=({"fa_rep_pack_frac": 1.0, "fa_rep_min_frac": 1.0, "min_rtol": 1e-5},),
    cycles=1,
    max_iter=200,
    backbone_neighbors=1,
    repack=True,
)


@dataclass(frozen=True)
class DockingFinalization:
    """Exact-refined candidates and their RosettaVS ranking components.

    Candidate tensors have shape ``[1, top_k]`` and are sorted by the selected
    metric (exact complex energy by default, Rosetta's ``top_pose_metric="score"``)
    for the single input ligand. ``pose_stack`` uses
    the same candidate order.
    """

    pose_stack: PoseStack
    genes: torch.Tensor
    pre_refine_energies: torch.Tensor
    complex_energies: torch.Tensor
    free_ligand_energies: torch.Tensor
    free_receptor_energies: torch.Tensor
    nchi: torch.Tensor
    dH: torch.Tensor
    TdS: torch.Tensor
    dG: torch.Tensor


def _score_pose_stack(pose_stack: PoseStack, sfxn) -> torch.Tensor:
    scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
    with torch.no_grad():
        return scorer(pose_stack.coords)


def _refine_fa_rep_scaled(base_fa_ljrep, param_db, device, rep_mult):
    """Fresh beta2016 with scaled fa_ljrep and all other weights identical."""
    sfxn = beta2016_score_function(device, param_db=param_db)
    sfxn.set_weight(ScoreType.fa_ljrep, float(base_fa_ljrep) * float(rep_mult))
    return sfxn


def _soft_hard_refine(
    promoted, minimize, *, param_db, device, exact_sfxn, schedule, cycles
):
    """Run soft-to-hard cycles, accepting each pose's best exact-energy state.

    ``minimize(pose_stack, step_sfxn, tol, max_iter)`` runs one ramp step.
    """
    # beta2016_score_function MEMOIZES when param_db is None, so a None db would
    # make the soft/hard/exact score functions ALIAS one object (set_weight on
    # "hard" would clobber "soft"). Resolve an explicit database so every stage
    # is an independent, freshly built score function.
    if param_db is None:
        param_db = ParameterDatabase.get_default()
    base_fa_ljrep = float(exact_sfxn.get_weight(ScoreType.fa_ljrep))
    step_sfxns = [
        (
            _refine_fa_rep_scaled(base_fa_ljrep, param_db, device, rep_mult),
            tol,
            max_iter,
        )
        for rep_mult, tol, max_iter in schedule
    ]
    # The refined topology never changes, so render the exact scorer once and
    # feed it coordinates; rendering per cycle is pure setup.
    exact_scorer = exact_sfxn.render_whole_pose_scoring_module(promoted)

    def exact_energy(pose):
        with torch.no_grad():
            return exact_scorer(pose.coords)

    best_coords = promoted.coords.clone()
    # A non-finite pre-refine score (e.g. a severe clash scoring to +/-inf/NaN)
    # must not block accept-to-best: seed it to +inf so any finite refined-cycle
    # energy is accepted. A candidate that stays non-finite ranks last (dH/dG=inf)
    # rather than poisoning the sort with NaN.
    best_energy = exact_energy(promoted)
    best_energy = torch.where(
        torch.isfinite(best_energy),
        best_energy,
        torch.full_like(best_energy, float("inf")),
    )
    current = promoted
    for _ in range(cycles):
        for step_sfxn, tol, max_iter in step_sfxns:
            current = minimize(current, step_sfxn, tol, max_iter)
        energy = exact_energy(current)
        improved = torch.isfinite(energy) & (energy < best_energy)
        best_coords = torch.where(improved[:, None, None], current.coords, best_coords)
        best_energy = torch.where(improved, energy, best_energy)
        current = attr.evolve(current, coords=best_coords.clone())
    return attr.evolve(promoted, coords=best_coords), best_energy


def _ligand_cartmin(
    pose_stack, exact_sfxn, ligand_coord_mask, *, max_iter=50, rtol=1e-4, minimizer=None
):
    """Ligand-only Cartesian minimization at exact hard weights."""
    kwargs = {"max_iter": max_iter, "rtol": rtol}
    if minimizer is not None:
        return minimizer(
            pose_stack, exact_sfxn, ligand_coord_mask, optimizer_kwargs=kwargs
        )
    return run_cart_min(
        pose_stack,
        exact_sfxn,
        coord_mask=ligand_coord_mask,
        optimizer_kwargs=kwargs,
    )


def _ligand_atom_indices(pose_stack: PoseStack, ligand_block: int) -> torch.Tensor:
    block_type_indices = pose_stack.block_type_ind64[:, ligand_block]
    if bool(torch.any(block_type_indices < 0)):
        raise ValueError("ligand block is absent from at least one pose")
    if not bool(torch.all(block_type_indices == block_type_indices[0])):
        raise ValueError("finalization requires one homogeneous ligand type")
    block_type = pose_stack.packed_block_types.active_block_types[
        int(block_type_indices[0])
    ]
    local_atoms = torch.arange(
        block_type.n_atoms, dtype=torch.int64, device=pose_stack.device
    )
    return pose_stack.block_coord_offset64[:, ligand_block, None] + local_atoms


def _ligand_atom_mask(pose_stack: PoseStack, ligand_block: int) -> torch.Tensor:
    """[pose, atom] mask of the ligand block; its form, and size, may vary by pose."""
    block_type_indices = pose_stack.block_type_ind64[:, ligand_block]
    if bool(torch.any(block_type_indices < 0)):
        raise ValueError("ligand block is absent from at least one pose")
    start = pose_stack.block_coord_offset64[:, ligand_block, None]
    end = start + pose_stack.packed_block_types.n_atoms[block_type_indices, None]
    atoms = torch.arange(pose_stack.coords.shape[1], device=pose_stack.device)
    return (atoms >= start) & (atoms < end)


def _stack_poses(poses: list[PoseStack]) -> PoseStack:
    """Single-pose stacks, whose block layouts may differ, as one stack in their dtype."""
    stacked = PoseStackBuilder.from_poses(poses, poses[0].device)
    coords = torch.zeros_like(stacked.coords, dtype=poses[0].coords.dtype)
    for i, pose in enumerate(poses):
        coords[i, : pose.coords.shape[1]] = pose.coords[0]
    return attr.evolve(stacked, coords=coords)


def _reordered_poses(pose_stack: PoseStack, order: torch.Tensor) -> PoseStack:
    """The poses in ``order``; whole poses move when their layouts differ."""
    types = pose_stack.block_type_ind64
    connections = pose_stack.inter_residue_connections64
    if bool(torch.all(types == types[:1])) and bool(
        torch.all(connections == connections[:1])
    ):
        return attr.evolve(pose_stack, coords=pose_stack.coords[order].detach().clone())
    return _stack_poses([pose_stack.split(int(i)) for i in order])


def _free_ligand_pose_stack(pose_stack: PoseStack, ligand_block: int) -> PoseStack:
    """Each pose's ligand alone, in the form it takes in that complex.

    A ligand that coordinates a metal stays the conjugate base it bound as; only
    its connections go.
    """
    connected = pose_stack.inter_residue_connections64[:, ligand_block, :, 0] >= 0
    if not bool(connected.any()):
        return movable_pose_stack(pose_stack, ligand_block)
    return _stack_poses(
        [
            select_blocks(pose_stack.split(i), [ligand_block])
            for i in range(pose_stack.n_poses)
        ]
    )


def _promote_top_candidates(
    reference_pose_stack: PoseStack,
    ga_result: GAResult,
    top_k: int,
    layout: SearchLayout,
) -> tuple[PoseStack, torch.Tensor]:
    """Copy the top ``top_k`` pool poses' moving atoms into full complexes."""

    if reference_pose_stack.n_poses != 1:
        raise ValueError("finalization requires one reference pose")
    pool = ga_result.pool.pose_stack
    if pool.max_n_blocks != layout.pose_stack.max_n_blocks:
        raise ValueError("GA pool topology does not match the backend's search view")
    flat = torch.argsort(ga_result.pool.energies)[:top_k]
    ligand_atom_map = ga_result.pool_ligand_atoms
    promoted = _place_ligand_in_reference(
        repeat_pose_stack(reference_pose_stack, int(flat.numel())),
        select_poses(pool, flat),
        layout,
        ligand_atom_map=(
            None
            if ligand_atom_map is None
            else ligand_atom_map[flat.to(ligand_atom_map.device)]
        ),
    )
    return promoted, ga_result.pool.genes[flat].unsqueeze(0)


def _sidechain_block_atoms(
    layout: SearchLayout,
) -> tuple[tuple[int, torch.Tensor], ...]:
    """(reference block, block-local moving atoms) of each flexible sidechain."""
    return tuple((block, atoms) for _, block, atoms in layout.sidechain_atoms)


def _mapped_atom_indices(
    pose_stack: PoseStack,
    block_local_atoms: tuple[tuple[int, torch.Tensor], ...],
) -> torch.Tensor:
    """Return per-pose absolute atom indices for block-local atom selections."""

    return torch.cat(
        [
            pose_stack.block_coord_offset64[:, block, None]
            + local_atoms.to(device=pose_stack.device)[None, :]
            for block, local_atoms in block_local_atoms
        ],
        dim=1,
    )


def _receptor_only_pose_stack(pose_stack: PoseStack, ligand_block: int) -> PoseStack:
    """Delete one disconnected ligand block while preserving receptor topology."""

    if bool(
        torch.any(pose_stack.inter_residue_connections64[:, ligand_block, :, 0] >= 0)
    ):
        raise ValueError("the finalized ligand must be disconnected from the receptor")
    real_blocks = torch.nonzero(pose_stack.block_type_ind64[0] >= 0).flatten().tolist()
    return select_blocks(pose_stack, [b for b in real_blocks if b != ligand_block])


def _with_coordinate_constraints(pose_stack: PoseStack) -> PoseStack:
    """Constrain every ligand atom to its current coordinate with sigma 1 A."""

    n_poses, n_atoms = pose_stack.coords.shape[:2]
    pose = torch.arange(n_poses, device=pose_stack.device)[:, None].expand(-1, n_atoms)
    atom = torch.arange(n_atoms, device=pose_stack.device)[None, :].expand(n_poses, -1)
    block = torch.zeros_like(atom)
    constraint_atoms = (
        torch.stack((pose, block, atom), dim=2).reshape(-1, 1, 3).to(torch.int32)
    )
    params = torch.zeros(
        (n_poses * n_atoms, 5), dtype=torch.float32, device=pose_stack.device
    )
    params[:, 1:4] = pose_stack.coords.reshape(-1, 3)
    params[:, 4] = 1.0
    constraints = ConstraintSet.create_empty(pose_stack.device, n_poses)
    constraints = constraints.add_constraints(
        ConstraintEnergyTerm.harmonic_coordinate, constraint_atoms, params
    )
    return attr.evolve(pose_stack, constraint_set=constraints)


def count_non_proton_ligand_chis(pose_stack: PoseStack, ligand_block: int) -> int:
    """Count named ligand chis whose outer atoms are not polar hydrogens.

    Rosetta params encode proton chis as a named chi ending at the rotating
    OH/SH/NH hydrogen. Checking both outer atoms also handles an equivalent
    reversed torsion declaration.
    """

    block_type_index = int(pose_stack.block_type_ind64[0, ligand_block])
    block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
    atom_is_hydrogen = (
        pose_stack.packed_block_types.atom_is_hydrogen[block_type_index]
        .to(torch.bool)
        .detach()
        .cpu()
        .numpy()
    )
    nchi = 0
    for torsion_index in range(block_type.n_torsions):
        atoms = block_type.ordered_torsions[torsion_index, :, 0]
        if numpy.any(atoms < 0):
            raise ValueError("ligand chi atoms must be local to the ligand block")
        if bool(atom_is_hydrogen[int(atoms[0])]) or bool(
            atom_is_hydrogen[int(atoms[3])]
        ):
            continue
        nchi += 1
    return nchi


def _refine_flexible(promoted, exact_sfxn, param_db, coord_mask, protocol, pack_blocks):
    """Rosetta dockflex: cartesian ramp_repack_min over ligand and sidechains.

    Runs tmol's own FastRelax so the repack at each ramp step, and the
    accept-to-best across cycles, are the shared implementation rather than a
    second one. ``pack_blocks`` selects the sidechains the packer may rebuild;
    empty leaves the receptor fixed and the stage is minimization only.
    """
    minimizer = CartesianRelaxMinimizer(
        CartesianMoveMap(coord_mask), max_iter=protocol.max_iter
    )
    operations = refine_task_operations(
        promoted, exact_sfxn, pack_blocks, protocol.repack
    )
    refined = fast_relax(
        promoted,
        exact_sfxn,
        minimizer=minimizer,
        num_repeats=protocol.cycles,
        schedule=list(protocol.steps),
        task_operations=operations or None,
    )
    return refined, _score_pose_stack(refined, exact_sfxn)


def refine_task_operations(pose_stack, sfxn, pack_blocks, repack):
    """Packer task operations for the refinement stage.

    Without ``repack`` the only sampler is the fallback one, whose single
    rotamer is the input conformation, so the packer cannot move a sidechain.
    """
    operations = []
    if pack_blocks is not None:
        if repack:
            operations.append(default_relax_task_operation(pose_stack, sfxn))
        operations.append(_repack_only_blocks(pose_stack, pack_blocks))
    return operations


def _repack_only_blocks(pose_stack, blocks):
    """Task operation restricting repacking to ``blocks``.

    Composed after the relax default, whose samplers it keeps.
    """

    def operation(task):
        mask = torch.zeros(
            (pose_stack.n_poses, pose_stack.max_n_blocks),
            dtype=torch.bool,
            device=pose_stack.device,
        )
        if len(blocks):
            mask[:, torch.as_tensor(list(blocks), device=pose_stack.device)] = True
        task.restrict_to_repacking()
        task.disable_packing_by_block_mask(~mask)

    return operation


def _refine_rigid(
    promoted,
    reference,
    exact_sfxn,
    param_db,
    ligand_block,
    ligand_mask,
    refine_schedule,
    refine_cycles,
    movable_blocks=None,
):
    """Torsion-space soft-to-hard refinement bracketed by ligand Cartesian min."""
    # One minimizer per distinct score function, reused across refine cycles:
    # the rendered network and its topology tensors are setup, not numerics.
    cartesian = CartesianMinimizer()
    torsion_minimizers: dict[int, KinematicMinimizer] = {}

    promoted = _ligand_cartmin(promoted, exact_sfxn, ligand_mask, minimizer=cartesian)
    # Redetection can leave the promoted poses holding different ligand forms.
    fold_forest, move_map = per_pose_fold_forest_and_move_map(promoted, ligand_block)

    def torsion_step(pose, step_sfxn, tol, max_iter):
        # Fixed-hydrogen harness: no whole-pose optH, ramp steps minimize only.
        # rtol maps Rosetta's fractional-energy tolerance (in float32 1e-5 is
        # below the dtype floor, so the hard step stops on gradient/atol/max_iter).
        minimizer = torsion_minimizers.get(id(step_sfxn))
        if minimizer is None:
            minimizer = KinematicMinimizer(
                kin_dtype=pose.coords.dtype, movable_blocks=movable_blocks
            )
            torsion_minimizers[id(step_sfxn)] = minimizer
        return minimizer(
            pose,
            step_sfxn,
            fold_forest,
            move_map,
            optimizer_kwargs={"max_iter": max_iter, "rtol": tol},
        )

    promoted, _ = _soft_hard_refine(
        promoted,
        torsion_step,
        param_db=param_db,
        device=promoted.device,
        exact_sfxn=exact_sfxn,
        schedule=refine_schedule,
        cycles=refine_cycles,
    )
    refined = _ligand_cartmin(promoted, exact_sfxn, ligand_mask, minimizer=cartesian)
    # run_kin_min refolds the whole pose and float32 kinematics leave ~1e-3 A
    # noise on the nominally fixed receptor; restore every non-ligand atom.
    refined = attr.evolve(
        refined,
        coords=torch.where(
            ligand_mask.unsqueeze(-1),
            refined.coords,
            reference.coords.expand(refined.n_poses, -1, -1),
        ),
    )
    return refined, _score_pose_stack(refined, exact_sfxn)


def _free_state_energies(
    refined, reference, exact_sfxn, param_db, ligand_block, block_local_atoms
):
    """Energies of the relaxed free ligand and of the (flex-relaxed) receptor."""
    # Constraint weights belong to the minimizer, not the memoized exact scorer.
    if param_db is None:
        param_db = ParameterDatabase.get_default()
    constrained_sfxn = beta2016_score_function(refined.device, param_db=param_db)
    constrained_sfxn.set_weight(ScoreType.constraint, 1.0)
    relaxed_ligand = run_cart_min(
        _with_coordinate_constraints(_free_ligand_pose_stack(refined, ligand_block)),
        constrained_sfxn,
        optimizer_kwargs={"max_iter": 30},
    )
    free_ligand_energy = _score_pose_stack(relaxed_ligand, exact_sfxn)

    receptor = _receptor_only_pose_stack(reference, ligand_block)
    if block_local_atoms:
        kept = torch.nonzero(reference.block_type_ind64[0] >= 0).flatten().tolist()
        kept = [block for block in kept if block != ligand_block]
        receptor_block = {source: index for index, source in enumerate(kept)}
        mask = torch.zeros(
            receptor.coords.shape[:2], dtype=torch.bool, device=receptor.device
        )
        mask.scatter_(
            1,
            _mapped_atom_indices(
                receptor,
                tuple((receptor_block[b], atoms) for b, atoms in block_local_atoms),
            ),
            True,
        )
        receptor = run_cart_min(
            receptor,
            exact_sfxn,
            coord_mask=mask,
            optimizer_kwargs={"max_iter": 50},
        )
    return free_ligand_energy, _score_pose_stack(receptor, exact_sfxn)[0]


def finalize_ga_docking(
    reference_pose_stack: PoseStack,
    backend: ScoringBackend,
    ga_result: GAResult,
    *,
    refine: RefineProtocol | None = None,
    top_k: int | None = None,
    top_pose_metric: str = "score",
) -> DockingFinalization:
    """Refine the top candidates and rank them by dG or exact complex score.

    ``refine`` defaults to the protocol matching the backend's mode:
    :data:`FLEX_REFINE` when it declares flexible sidechains, else
    :data:`RIGID_REFINE`. ``top_k`` overrides the protocol's own count.
    ``top_pose_metric`` selects ``"dG"`` (``dH + 0.4*nchi``) or ``"score"``.
    """

    if reference_pose_stack.n_poses != 1:
        raise ValueError("finalization requires one reference pose")
    if top_pose_metric not in ("dG", "score"):
        raise ValueError('top_pose_metric must be "dG" or "score"')
    npool = ga_result.ranked_energies.shape[1]
    if top_k is not None and top_k > npool:
        raise ValueError(f"top_k ({top_k}) cannot exceed GA pool size ({npool})")
    if refine is None:
        refine = _default_refine(backend)
    if top_k is None:
        # The protocol's count is a default, not a demand on the pool size.
        top_k = min(refine.top_k, npool)

    layout = search_layout(reference_pose_stack, backend)
    promoted, genes = _promote_top_candidates(
        reference_pose_stack, ga_result, top_k, layout
    )
    return _finalize_promoted(
        reference_pose_stack,
        backend,
        refine,
        top_pose_metric,
        promoted,
        genes,
        layout,
    )


def _default_refine(backend) -> RefineProtocol:
    """FLEX_REFINE when the backend moves sidechains, else RIGID_REFINE."""
    return FLEX_REFINE if backend.sidechain_blocks else RIGID_REFINE


def _finalize_promoted(
    reference_pose_stack: PoseStack,
    backend: ScoringBackend,
    refine: RefineProtocol,
    top_pose_metric: str,
    promoted: PoseStack,
    genes: torch.Tensor,
    layout: SearchLayout,
) -> DockingFinalization:
    """Refine complexes laid out as the reference, score dH and dG, and rank them."""
    ligand_block = backend.ligand_block
    sidechain_blocks = tuple(backend.sidechain_blocks)
    block_local_atoms = _sidechain_block_atoms(layout)
    if refine.redetect_metal:
        try:
            promoted = redetect_metal_coordination(promoted)
        except Exception as error:
            # A complex that cannot be rebuilt from its canonical form keeps the
            # coordination it was built with; losing the candidate is worse.
            logger.warning("metal coordination not redetected: %s", error)
    exact_sfxn = beta2016_score_function(promoted.device, param_db=backend.param_db)
    pre_refine = _score_pose_stack(promoted, exact_sfxn)
    coord_mask = _ligand_atom_mask(promoted, ligand_block)
    if sidechain_blocks:
        coord_mask.scatter_(1, _mapped_atom_indices(promoted, block_local_atoms), True)
        # Reference-complex block indices: refinement runs on full complexes.
        pack_blocks = (
            sidechain_blocks
            if refine.sidechains == "backend"
            else tuple(refine.sidechains)
        )
        refined, complex_energy = _refine_flexible(
            promoted, exact_sfxn, backend.param_db, coord_mask, refine, pack_blocks
        )
    else:
        # promoted, not the reference: redetection can change its atom layout
        refined, complex_energy = _refine_rigid(
            promoted,
            promoted,
            exact_sfxn,
            backend.param_db,
            ligand_block,
            coord_mask,
            refine.ramp(),
            refine.cycles,
            movable_blocks=coord_mask,
        )
    free_ligand_energy, free_receptor_energy = _free_state_energies(
        refined,
        reference_pose_stack,
        exact_sfxn,
        backend.param_db,
        ligand_block,
        sidechain_blocks and block_local_atoms,
    )
    free_receptor_energies = free_receptor_energy.expand_as(complex_energy)

    nchi_value = count_non_proton_ligand_chis(reference_pose_stack, ligand_block)
    nchi = torch.full(
        complex_energy.shape,
        nchi_value,
        dtype=torch.int64,
        device=promoted.device,
    )
    TdS = 0.4 * nchi.to(complex_energy.dtype)
    dH = complex_energy - free_ligand_energy - free_receptor_energies
    dG = dH + TdS

    n_poses = promoted.n_poses
    sort_key = dG if top_pose_metric == "dG" else complex_energy
    sort_order = torch.argsort(sort_key.reshape(1, n_poses), dim=1)

    def rank(values: torch.Tensor) -> torch.Tensor:
        matrix = values.reshape(1, n_poses)
        return matrix.gather(1, sort_order)

    ranked_genes = genes.gather(
        1, sort_order[:, :, None].expand(-1, -1, genes.shape[2])
    )
    ranked_pose_stack = _reordered_poses(refined, sort_order.reshape(-1))
    return DockingFinalization(
        pose_stack=ranked_pose_stack,
        genes=ranked_genes,
        pre_refine_energies=rank(pre_refine),
        complex_energies=rank(complex_energy),
        free_ligand_energies=rank(free_ligand_energy),
        free_receptor_energies=rank(free_receptor_energies),
        nchi=rank(nchi),
        dH=rank(dH),
        TdS=rank(TdS),
        dG=rank(dG),
    )


def _finalize_reference_pose(
    reference_pose_stack: PoseStack,
    backend: ScoringBackend,
    *,
    refine: RefineProtocol | None = None,
    top_pose_metric: str = "score",
) -> DockingFinalization:
    """The reference complex itself, refined and scored as a docked candidate is."""
    if reference_pose_stack.n_poses != 1:
        raise ValueError("finalization requires one reference pose")
    genes = torch.zeros(
        (1, 1, 0),
        dtype=reference_pose_stack.coords.dtype,
        device=reference_pose_stack.device,
    )
    return _finalize_promoted(
        reference_pose_stack,
        backend,
        _default_refine(backend) if refine is None else refine,
        top_pose_metric,
        reference_pose_stack,
        genes,
        search_layout(reference_pose_stack, backend),
    )


__all__ = [
    "DockingFinalization",
    "count_non_proton_ligand_chis",
    "finalize_ga_docking",
]
