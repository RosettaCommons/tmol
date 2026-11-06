import time
import warnings
from collections.abc import Callable, Sequence
from typing import Protocol

import attr
import numpy
import torch

from tmol.pose._pose_stack import PoseStack
from tmol.score._score_function import ScoreFunction
from tmol.score._score_types import ScoreType
from tmol.kinematics._move_map import CartesianMoveMap, MoveMap
from tmol.kinematics._fold_forest import FoldForest
from tmol.pack._pack_rotamers import pack_rotamers
from tmol.pack._packer_task import PackerPalette, PackerTask
from tmol.pack import FixedAAChiSampler, IncludeCurrentSampler, NaChiRotamerSampler
from tmol.optimization._minimizers import CartesianMinimizer, KinematicMinimizer
from tmol.types._torch import Tensor
from tmol.utility._device import _synchronize_device

RelaxScheduleEntry = float | int | dict[str, float]
PackerTaskOperation = Callable[[PackerTask], None]


class RelaxMinimizer(Protocol):
    """Callable contract for a FastRelax minimization stage."""

    def __call__(
        self,
        pose_stack: PoseStack,
        sfxn: ScoreFunction,
        *,
        verbose: bool,
        rtol: float | None = None,
    ) -> PoseStack:
        """Minimize and return the updated poses."""


# Jack Maguire's tuned MonomerRelax2019 schedule. Fractions scale the score
# function's initial fa_rep and constraint weights at each pack-minimize stage;
# min_rtol is the stage's minimizer relative tolerance.
DEFAULT_RELAX_SCHEDULE = [
    {
        "fa_rep_pack_frac": 0.040,
        "fa_rep_min_frac": 0.051,
        "cst_frac": 1.0,
        "min_rtol": 0.01,
    },
    {
        "fa_rep_pack_frac": 0.265,
        "fa_rep_min_frac": 0.280,
        "cst_frac": 0.5,
        "min_rtol": 0.01,
    },
    {
        "fa_rep_pack_frac": 0.559,
        "fa_rep_min_frac": 0.581,
        "cst_frac": 0.0,
        "min_rtol": 0.01,
    },
    {
        "fa_rep_pack_frac": 1.000,
        "fa_rep_min_frac": 1.000,
        "cst_frac": 0.0,
        "min_rtol": 5e-4,
    },
]


def _normalize_schedule(
    schedule: Sequence[RelaxScheduleEntry],
    constrain: bool = False,
    ramp_constraints: bool = False,
) -> list[tuple[float, float, float, float | None]]:
    """Normalize user schedule entries into pack, minimize, and constraint weights.

    Args:
        schedule: Numbers, used for both fa_rep stages, or dictionaries with
            ``fa_rep_pack_frac``, ``fa_rep_min_frac``, and optional
            ``cst_frac`` and ``min_rtol``.
        constrain: Whether the score function has an active constraint term.
        ramp_constraints: Whether to ramp its weight to zero.

    Returns:
        ``(pack_fraction, minimize_fraction, constraint_fraction, min_rtol)``
        per step.

    Raises:
        ValueError: If an entry is neither a float/int nor a dict with the
            required keys.
    """

    def constraint_fraction(step_index: int) -> float:
        """Determine the constraint fraction for a particular step.

        If ramping, ramp constraint frations down from 1.0 to 0.0 over
        the first half of the schedule, then keep at 0.0.
        """
        if not constrain:
            return 0.0
        if not ramp_constraints:
            return 1.0
        n_steps = len(schedule)
        if step_index > n_steps // 2:
            return 0.0
        return 1 - step_index / (n_steps / 2)

    normalized = []
    for i, entry in enumerate(schedule):
        if isinstance(entry, (int, float)):
            constraint = constraint_fraction(i)
            normalized.append((float(entry), float(entry), constraint, None))
        elif isinstance(entry, dict):
            pack_frac = float(entry["fa_rep_pack_frac"])
            min_frac = float(entry["fa_rep_min_frac"])
            constraint = (
                constraint_fraction(i)
                if "cst_frac" not in entry
                else float(entry["cst_frac"])
            )
            min_rtol = entry.get("min_rtol")
            min_rtol = None if min_rtol is None else float(min_rtol)
            normalized.append((pack_frac, min_frac, constraint, min_rtol))
        else:
            raise ValueError(
                f"Schedule entry must be a number or a dict with keys"
                f" 'fa_rep_pack_frac', 'fa_rep_min_frac', and optionally 'cst_frac'"
                f" and 'min_rtol', got {type(entry)}"
            )
    return normalized


def _graph_mode(cuda_graph: bool | None, pose_stack: PoseStack) -> bool:
    """Resolve a minimizer's graph setting; None replays graphs on CUDA."""
    return pose_stack.device.type == "cuda" if cuda_graph is None else cuda_graph


# Kinematic relax convergence: one small relative step at max |gradient| below
# 1 kcal/mol per degree (expressed per radian); rtol is overridden per stage.
KINEMATIC_RELAX_LBFGS_OPTIONS = {
    "rtol": 0.01,
    "atol": None,
    "patience": 1,
    "gradtol": 180.0 / numpy.pi,
}


class CartesianRelaxMinimizer:
    """FastRelax minimization of atom coordinates.

    Reuses the rendered scorer while the pose topology is unchanged.

    Args:
        move_map: Atoms free to move; None frees every atom.
        cuda_graph: Capture and replay the CUDA forward/backward scoring path.
            None turns it on for CUDA poses and off on CPU.
    """

    def __init__(
        self,
        move_map: CartesianMoveMap | None = None,
        *,
        cuda_graph: bool | None = None,
    ):
        if move_map is None:
            move_map = CartesianMoveMap()
        if not isinstance(move_map, CartesianMoveMap):
            raise TypeError(
                f"CartesianRelaxMinimizer needs a CartesianMoveMap, got "
                f"{type(move_map).__name__}; use KinematicRelaxMinimizer for a MoveMap"
            )
        self.move_map = move_map
        self.cuda_graph = cuda_graph
        self.minimizer = CartesianMinimizer()

    def release_retained_state(self) -> None:
        """Release scorer and optimizer storage between relax stages."""
        self.minimizer.release_retained_state()

    def __call__(
        self,
        pose_stack: PoseStack,
        sfxn: ScoreFunction,
        *,
        verbose: bool = False,
        rtol: float | None = None,
    ) -> PoseStack:
        # Cartesian relax keeps the LBFGS default convergence
        del rtol
        self.minimizer.cuda_graph = _graph_mode(self.cuda_graph, pose_stack)
        return self.minimizer(
            pose_stack,
            sfxn,
            coord_mask=self.move_map.coord_mask,
            verbose=verbose,
            optimizer_kwargs={"verbose": verbose},
        )


class KinematicRelaxMinimizer:
    """FastRelax minimization of kinematic DOFs (torsions and jumps).

    Reuses the rendered scorer while the pose topology is unchanged.

    Args:
        move_map: Torsions and jumps free to move; None frees every mainchain
            and sidechain torsion and every jump of each minimized pose stack.
        fold_forest: Kinematic tree; None uses
            FoldForest.reasonable_fold_forest of each minimized pose stack.
        cuda_graph: Capture and replay the CUDA forward/backward scoring path.
            None turns it on for CUDA poses and off on CPU.
    """

    def __init__(
        self,
        move_map: MoveMap | None = None,
        fold_forest: FoldForest | None = None,
        *,
        cuda_graph: bool | None = None,
    ):
        if move_map is not None and not isinstance(move_map, MoveMap):
            raise TypeError(
                f"KinematicRelaxMinimizer needs a MoveMap, got "
                f"{type(move_map).__name__}; use CartesianRelaxMinimizer for a "
                "CartesianMoveMap"
            )
        self.move_map = move_map
        self.fold_forest = fold_forest
        self.cuda_graph = cuda_graph
        self.minimizer = KinematicMinimizer()

    def release_retained_state(self) -> None:
        """Release scorer and optimizer storage between relax stages."""
        self.minimizer.release_retained_state()

    def __call__(
        self,
        pose_stack: PoseStack,
        sfxn: ScoreFunction,
        *,
        verbose: bool = False,
        rtol: float | None = None,
    ) -> PoseStack:
        move_map = self.move_map
        if move_map is None:
            move_map = MoveMap.from_pose_stack(pose_stack)
            move_map.move_all_mc = True
            move_map.move_all_sc = True
            move_map.move_all_jumps = True
        fold_forest = self.fold_forest
        if fold_forest is None:
            fold_forest = FoldForest.reasonable_fold_forest(pose_stack)
        self.minimizer.cuda_graph = _graph_mode(self.cuda_graph, pose_stack)
        options = dict(KINEMATIC_RELAX_LBFGS_OPTIONS, verbose=verbose)
        if rtol is not None:
            options["rtol"] = rtol
        return self.minimizer(
            pose_stack,
            sfxn,
            fold_forest,
            move_map,
            verbose=verbose,
            optimizer_kwargs=options,
        )


def fast_relax(  # noqa: C901
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    *,
    minimizer: RelaxMinimizer | None = None,
    packer_palette: PackerPalette | None = None,
    task_operations: Sequence[PackerTaskOperation] | None = None,
    num_repeats: int = 2,
    ramp_constraints: bool | None = None,  # default True
    schedule: Sequence[RelaxScheduleEntry] | None = None,
    release_between_stages: bool = True,
    verbose: bool = False,
) -> PoseStack:
    """Relax poses through repeated side-chain packing and minimization.

    Each repeat applies the MonomerRelax2019 fa_rep ramp and retains the
    lowest-scoring conformation independently for every pose in the batch.

    Args:
        pose_stack: Input poses to relax.
        sfxn: Packing and minimization score function. Constraints are active
            only when its constraint weight is nonzero.
        minimizer: The minimization mode and its settings: a
            ``CartesianRelaxMinimizer`` (atom coordinates) or a
            ``KinematicRelaxMinimizer`` (torsions and jumps along a fold
            forest), or any callable following ``RelaxMinimizer``. If it
            provides ``release_retained_state()``, FastRelax calls that hook
            after extracting each minimized pose so packing does not overlap
            retained scorer or optimizer storage. Defaults to a
            ``CartesianRelaxMinimizer`` with every atom free.
        packer_palette: Residue types available to the packer. Defaults to
            ``PackerPalette()``.
        task_operations: In-place task configuration callbacks. By default,
            restrict to repacking with the scoring database’s Dunbrack, fixed-AA,
            current and nucleic-acid rotamers, sampling covalent groups jointly.
        num_repeats: Number of complete pack-minimize ramps.
        ramp_constraints: Ramp an active constraint weight to zero. Defaults
            to true; the input weight is restored after relaxation.
        schedule: Numeric fa_rep fractions or dictionaries with separate pack,
            minimize, and optional constraint fractions and ``min_rtol``.
            Defaults to ``DEFAULT_RELAX_SCHEDULE``.
        release_between_stages: Drop the rendered scorer, CUDA graph and L-BFGS
            history after each stage's minimization. On by default: a large
            system is bound by that retained state through the next packing
            phase, which is why the release exists. Turn it off when the setup
            cost dominates instead -- a small pose on the default four-stage
            schedule otherwise pays four network constructions, and four graph
            captures, where one would do.
        verbose: Print timing information for each step.

    Returns:
        Best-scoring relaxed poses across all repeats.
    """
    if schedule is None:
        schedule = DEFAULT_RELAX_SCHEDULE

    constraint_weight_start = sfxn.get_weight(ScoreType.constraint)
    use_constraints = constraint_weight_start != 0
    if not use_constraints and ramp_constraints:
        print(
            "Warning: ramp_constraints is True but sfxn's 'constraint' weight is 0; no constraints will be used."
        )
    if ramp_constraints is None:
        ramp_constraints = True

    steps = _normalize_schedule(schedule, use_constraints, ramp_constraints)

    if not steps:
        raise ValueError("Relax schedule must contain at least one step.")

    # Warn if the final step doesn't restore fa_rep to its full weight.
    final_min_frac = steps[-1][1]
    if abs(final_min_frac - 1.0) > 1e-6:
        warnings.warn(
            f"Final schedule step has fa_rep min fraction {final_min_frac:.4f},"
            f" not 1.0!",
            stacklevel=2,
        )

    if task_operations is None:
        torch_device = pose_stack.device
        from tmol.pack import create_dunbrack_sampler_from_database
        from tmol.pack.rotamer._conjugated_groups import _add_conjugated_group_sampler

        samplers = [
            create_dunbrack_sampler_from_database(sfxn._param_db, torch_device),
            FixedAAChiSampler(),
            IncludeCurrentSampler(),
        ]
        if any(
            bt.properties.polymer.backbone_type in ("dna", "rna")
            for bt in pose_stack.packed_block_types.active_block_types
        ):
            samplers.append(
                NaChiRotamerSampler.from_database(sfxn._param_db, torch_device)
            )

        def default_op(task: PackerTask) -> None:
            task.restrict_to_repacking()
            for sampler in samplers:
                task.add_conformer_sampler(sampler)
            _add_conjugated_group_sampler(task, pose_stack)

        task_operations = [default_op]

    fa_rep_start = float(sfxn.get_weight(ScoreType.fa_ljrep))

    wpsm = sfxn.render_whole_pose_scoring_module(pose_stack)
    best_score = wpsm(pose_stack.coords)
    del wpsm
    best_ps = pose_stack.clone_sharing_topology()

    if minimizer is None:
        minimizer = CartesianRelaxMinimizer()
    if packer_palette is None:
        packer_palette = PackerPalette()

    ps = pose_stack
    for _ in range(num_repeats):
        for pack_frac, min_frac, cst_frac, min_rtol in steps:
            ps = _relax_pack_min_step(
                pose_stack=ps,
                sfxn=sfxn,
                packer_palette=packer_palette,
                fa_rep_pack_weight=pack_frac * fa_rep_start,
                fa_rep_min_weight=min_frac * fa_rep_start,
                cst_weight=cst_frac * constraint_weight_start,
                task_operations=task_operations,
                minimizer=minimizer,
                verbose=verbose,
                release_between_stages=release_between_stages,
                min_rtol=min_rtol,
            )

        best_ps, best_score = _accept_best(sfxn, best_ps, best_score, ps, verbose)
        ps = best_ps.clone_sharing_topology()
    if use_constraints:
        # Restore original constraint weight to the score function
        sfxn.set_weight(ScoreType.constraint, constraint_weight_start)
    return ps


def _relax_pack_min_step(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    packer_palette: PackerPalette,
    fa_rep_pack_weight: float,
    fa_rep_min_weight: float,
    cst_weight: float,
    task_operations: Sequence[PackerTaskOperation],
    minimizer: RelaxMinimizer,
    verbose: bool,
    release_between_stages: bool = True,
    min_rtol: float | None = None,
) -> PoseStack:
    """Execute one weighted packing and minimization stage.

    Args:
        pose_stack: Current poses.
        sfxn: Score function whose repulsive and constraint weights are updated.
        packer_palette: Residue types available during packing.
        fa_rep_pack_weight: Repulsive weight for packing.
        fa_rep_min_weight: Repulsive weight for minimization.
        cst_weight: Constraint weight for both operations.
        task_operations: In-place task configuration callbacks.
        minimizer: Minimization callable.
        verbose: Print synchronized stage timings.
        min_rtol: Minimizer relative tolerance (None: minimizer default).

    Returns:
        The minimized pose stack.
    """

    if verbose:
        _synchronize_device(pose_stack.device)
    start_time = time.perf_counter()
    task = PackerTask(pose_stack, packer_palette)
    for op in task_operations:
        op(task)

    sfxn.set_weight(ScoreType.fa_ljrep, fa_rep_pack_weight)
    sfxn.set_weight(ScoreType.constraint, cst_weight)
    if verbose:
        print(
            f"packing with fa_rep of {fa_rep_pack_weight: .2f} and constraint weight of {cst_weight: .2f}"
        )
    if verbose:
        _synchronize_device(pose_stack.device)
    end_time1 = time.perf_counter()
    packed_pose_stack = pack_rotamers(pose_stack, sfxn, task, verbose)

    sfxn.set_weight(ScoreType.fa_ljrep, fa_rep_min_weight)
    if verbose:
        print(
            f"minimizing with fa_rep of {fa_rep_min_weight: .2f} and constraint weight of {cst_weight: .2f}"
        )
    if verbose:
        _synchronize_device(pose_stack.device)
    end_time2 = time.perf_counter()
    stage_kwargs = {} if min_rtol is None else {"rtol": min_rtol}
    minimized_pose_stack = minimizer(
        packed_pose_stack,
        sfxn,
        verbose=verbose,
        **stage_kwargs,
    )
    # Releasing costs a network construction and, with cuda_graph, a capture at
    # every stage; retaining costs the scorer and L-BFGS history through the
    # next packing phase. Large systems are bound by the second -- it is why
    # this release exists -- and small ones by the first.
    if release_between_stages:
        release_retained_state = getattr(minimizer, "release_retained_state", None)
        if release_retained_state is not None:
            release_retained_state()
    if verbose:
        _synchronize_device(pose_stack.device)
    end_time3 = time.perf_counter()

    if verbose:
        print(
            f"pack-min {end_time3 - start_time: .2f} task-init {end_time1 - start_time: .2f}"
            + f" packing {end_time2 - end_time1: .2f} min {end_time3 - end_time2: .2f}"
        )

    return minimized_pose_stack


def _accept_best(
    sfxn: ScoreFunction,
    best_pose_stack: PoseStack,
    best_pose_score: Tensor[torch.float32][:],
    candidate_pose_stack: PoseStack,
    verbose: bool = False,
) -> tuple[PoseStack, Tensor[torch.float32][:]]:
    """Keep the lower-scoring conformation independently for each pose.

    Args:
        sfxn: Score function used for comparison.
        best_pose_stack: Best poses from previous repeats.
        best_pose_score: Best scores shaped ``[n_poses]``.
        candidate_pose_stack: Newly minimized poses.
        verbose: Print accepted scores.

    Returns:
        Updated best poses and scores shaped ``[n_poses]``.
    """
    wpsm = sfxn.render_whole_pose_scoring_module(candidate_pose_stack)
    candidate_score = wpsm(candidate_pose_stack.coords)
    better_mask = candidate_score < best_pose_score

    def select_better(tensor_name: str) -> torch.Tensor:
        tensor = getattr(best_pose_stack, tensor_name)
        candidate = getattr(candidate_pose_stack, tensor_name)
        if tensor_name == "coords":
            # A packed block type can change a pose's atom count.
            width = max(tensor.shape[1], candidate.shape[1])
            tensor = torch.nn.functional.pad(tensor, (0, 0, 0, width - tensor.shape[1]))
            candidate = torch.nn.functional.pad(
                candidate, (0, 0, 0, width - candidate.shape[1])
            )
        new_tensor = tensor.detach().clone()
        new_tensor[better_mask] = candidate[better_mask]
        return new_tensor

    def select_better_atom_info(field_name: str, width: int) -> numpy.ndarray:
        """A per-atom pdb_info field, padded with zeros to the merged atom count."""
        best = getattr(best_pose_stack.pdb_info, field_name)
        candidate = getattr(candidate_pose_stack.pdb_info, field_name)
        better = better_mask.cpu().numpy()
        merged = numpy.zeros((best.shape[0], width), dtype=best.dtype)
        merged[~better, : best.shape[1]] = best[~better]
        merged[better, : candidate.shape[1]] = candidate[better]
        return merged

    if better_mask.any():
        if verbose:
            print("accepting new best scores")
            print(f" old best score: {best_pose_score[better_mask]}")
            print(f" new best score: {candidate_score[better_mask]}")

        new_coords = select_better("coords")
        new_block_coord_offset = select_better("block_coord_offset")
        new_block_coord_offset64 = select_better("block_coord_offset64")
        new_block_type_ind = select_better("block_type_ind")
        new_block_type_ind64 = select_better("block_type_ind64")
        new_pdb_info = attr.evolve(
            best_pose_stack.pdb_info,
            atom_occupancy=select_better_atom_info(
                "atom_occupancy", new_coords.shape[1]
            ),
            atom_b_factor=select_better_atom_info("atom_b_factor", new_coords.shape[1]),
        )
        new_best_pose_stack = attr.evolve(
            best_pose_stack,
            coords=new_coords,
            pdb_info=new_pdb_info,
            block_coord_offset=new_block_coord_offset,
            block_coord_offset64=new_block_coord_offset64,
            block_type_ind=new_block_type_ind,
            block_type_ind64=new_block_type_ind64,
        )
        new_best_pose_score = best_pose_score.detach().clone()
        new_best_pose_score[better_mask] = candidate_score[better_mask]
        return new_best_pose_stack, new_best_pose_score
    return best_pose_stack, best_pose_score
