from collections.abc import Callable, Sequence

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
from tmol.utility._device import _synchronize_device

from tmol_core.relax import (  # noqa: F401; preserve existing re-exports
    RelaxMinimizer as RelaxMinimizer,
    DEFAULT_RELAX_SCHEDULE as DEFAULT_RELAX_SCHEDULE,
    _normalize_schedule as _normalize_schedule,
    fast_relax as _shared_fast_relax,
    relax_pack_min_step as _shared_relax_step,
    accept_best as _shared_accept_best,
    default_task_operation,
    KINEMATIC_RELAX_LBFGS_OPTIONS,
)

RelaxScheduleEntry = float | int | dict[str, float]
PackerTaskOperation = Callable[[PackerTask], None]


def _graph_mode(cuda_graph: bool | None, pose_stack: PoseStack) -> bool:
    """Resolve a minimizer's graph setting; None replays graphs on CUDA."""
    return pose_stack.device.type == "cuda" if cuda_graph is None else cuda_graph


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

    return _shared_fast_relax(
        pose_stack,
        sfxn,
        execution=_TorchRelaxExecution,
        minimizer=minimizer,
        packer_palette=packer_palette,
        task_operations=task_operations,
        num_repeats=num_repeats,
        ramp_constraints=ramp_constraints,
        schedule=schedule,
        release_between_stages=release_between_stages,
        verbose=verbose,
    )


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

    return _shared_relax_step(
        pose_stack=pose_stack,
        sfxn=sfxn,
        packer_palette=packer_palette,
        fa_rep_pack_weight=fa_rep_pack_weight,
        fa_rep_min_weight=fa_rep_min_weight,
        cst_weight=cst_weight,
        task_operations=task_operations,
        minimizer=minimizer,
        verbose=verbose,
        release_between_stages=release_between_stages,
        min_rtol=min_rtol,
        execution=_TorchRelaxExecution,
    )


def _accept_best(
    sfxn, best_pose_stack, best_pose_score, candidate_pose_stack, verbose=False
):
    return _shared_accept_best(
        _TorchRelaxExecution(),
        sfxn,
        best_pose_stack,
        best_pose_score,
        candidate_pose_stack,
        verbose,
    )


def _select_better_pose(best_pose_stack, candidate_pose_stack, better_mask):
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

    new_coords = select_better("coords")
    new_block_coord_offset = select_better("block_coord_offset")
    new_block_coord_offset64 = select_better("block_coord_offset64")
    new_block_type_ind = select_better("block_type_ind")
    new_block_type_ind64 = select_better("block_type_ind64")
    new_pdb_info = attr.evolve(
        best_pose_stack.pdb_info,
        atom_occupancy=select_better_atom_info("atom_occupancy", new_coords.shape[1]),
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
    return new_best_pose_stack


class _TorchRelaxExecution:
    score_type = ScoreType
    task = staticmethod(PackerTask)
    select_pose = staticmethod(_select_better_pose)

    @staticmethod
    def clone(pose):
        return pose.clone_sharing_topology()

    @staticmethod
    def synchronize(pose):
        _synchronize_device(pose.device)

    @staticmethod
    def pack(pose, sfxn, task, verbose):
        return pack_rotamers(pose, sfxn, task, verbose)

    @staticmethod
    def select_scores(best, candidate, better):
        result = best.detach().clone()
        result[better] = candidate[better]
        return result

    @staticmethod
    def default_minimizer():
        return CartesianRelaxMinimizer()

    default_palette = staticmethod(PackerPalette)

    @staticmethod
    def default_task_operations(pose, sfxn):
        from tmol.pack import create_dunbrack_sampler_from_database
        from tmol.pack.rotamer._conjugated_groups import _add_conjugated_group_sampler

        samplers = [
            create_dunbrack_sampler_from_database(sfxn._param_db, pose.device),
            FixedAAChiSampler(),
            IncludeCurrentSampler(),
        ]
        if any(
            bt.properties.polymer.backbone_type in ("dna", "rna")
            for bt in pose.packed_block_types.active_block_types
        ):
            samplers.append(
                NaChiRotamerSampler.from_database(sfxn._param_db, pose.device)
            )
        return default_task_operation(pose, samplers, _add_conjugated_group_sampler)
