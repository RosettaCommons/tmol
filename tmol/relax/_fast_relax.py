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
from tmol.optimization._minimizers import (
    CartesianMinimizer,
    KinematicMinimizer,
    run_cart_min,
    run_kin_min,
)
from tmol.utility._device import _synchronize_device

from tmol_core.relax import (  # noqa: F401; preserve existing re-exports
    DEFAULT_RELAX_SCHEDULE as DEFAULT_RELAX_SCHEDULE,
    _normalize_schedule as _normalize_schedule,
    fast_relax as _shared_fast_relax,
    relax_pack_min_step as _shared_relax_step,
    accept_best as _shared_accept_best,
    default_task_operation,
)

RelaxScheduleEntry = float | int | dict[str, float]
PackerTaskOperation = Callable[[PackerTask], None]


class RelaxMinimizer(Protocol):
    """Callable contract for a FastRelax minimization stage."""

    def __call__(
        self,
        pose_stack: PoseStack,
        sfxn: ScoreFunction,
        *,
        fold_forest: FoldForest,
        move_map: MoveMap | CartesianMoveMap,
        verbose: bool,
    ) -> PoseStack:
        """Minimize and return the updated poses."""


def _default_cart_min_fn(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    *,
    fold_forest: FoldForest,
    move_map: MoveMap | CartesianMoveMap,
    verbose: bool,
    cuda_graph: bool = False,
) -> PoseStack:
    """Run Cartesian LBFGS using a Cartesian move map's coordinate mask."""
    coord_mask = move_map.coord_mask if isinstance(move_map, CartesianMoveMap) else None
    return run_cart_min(
        pose_stack,
        sfxn,
        coord_mask=coord_mask,
        verbose=verbose,
        cuda_graph=cuda_graph,
        optimizer_kwargs={"verbose": verbose},
    )


class DefaultCartesianMinimizer:
    """Cartesian minimizer that reuses rendering for unchanged pose topology."""

    def __init__(self, cuda_graph: bool):
        self.minimizer = CartesianMinimizer(cuda_graph=cuda_graph)

    def release_retained_state(self) -> None:
        """Release scorer and optimizer storage between relax stages."""
        self.minimizer.release_retained_state()

    def __call__(
        self,
        pose_stack: PoseStack,
        sfxn: ScoreFunction,
        *,
        fold_forest: FoldForest,
        move_map: MoveMap | CartesianMoveMap,
        verbose: bool,
    ) -> PoseStack:
        del fold_forest
        coord_mask = (
            move_map.coord_mask if isinstance(move_map, CartesianMoveMap) else None
        )
        return self.minimizer(
            pose_stack,
            sfxn,
            coord_mask=coord_mask,
            verbose=verbose,
            optimizer_kwargs={"verbose": verbose},
        )


def _default_kin_min_fn(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    *,
    fold_forest: FoldForest,
    move_map: MoveMap | CartesianMoveMap,
    verbose: bool,
    cuda_graph: bool = False,
) -> PoseStack:
    """Default minimization function: kinematic (torsion-space) LBFGS."""
    return run_kin_min(
        pose_stack,
        sfxn,
        fold_forest,
        move_map,
        verbose=verbose,
        cuda_graph=cuda_graph,
        optimizer_kwargs={"verbose": verbose},
    )


class DefaultKinematicMinimizer:
    """Kinematic minimizer that reuses rendering for unchanged pose topology."""

    def __init__(self, cuda_graph: bool | str = False):
        self.cuda_graph = cuda_graph
        self.minimizer = KinematicMinimizer(cuda_graph=cuda_graph)

    def release_retained_state(self) -> None:
        """Release scorer and optimizer storage between relax stages."""
        self.minimizer.release_retained_state()

    def __call__(
        self,
        pose_stack: PoseStack,
        sfxn: ScoreFunction,
        *,
        fold_forest: FoldForest,
        move_map: MoveMap | CartesianMoveMap,
        verbose: bool,
    ) -> PoseStack:

        return self.minimizer(
            pose_stack,
            sfxn,
            fold_forest,
            move_map,
            verbose=verbose,
            optimizer_kwargs={"verbose": verbose},
        )


def _resolve_cuda_graph_mode(pose_stack: PoseStack, cuda_graph: bool | None) -> bool:
    """Choose graph replay automatically where launch overhead dominates."""
    if cuda_graph is not None:
        return cuda_graph
    if pose_stack.device.type != "cuda":
        return False

    used_block_types = torch.unique(pose_stack.block_type_ind).tolist()
    active_block_types = pose_stack.packed_block_types.active_block_types
    return any(
        block_type_ind >= 0
        and active_block_types[block_type_ind].properties.polymer.backbone_type
        in ("dna", "rna")
        for block_type_ind in used_block_types
    )


def fast_relax(  # noqa: C901
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    packer_pallete: PackerPalette,
    move_map: MoveMap | CartesianMoveMap,
    fold_forest: FoldForest,
    *,
    task_operations: Sequence[PackerTaskOperation] | None = None,
    num_repeats: int = 2,
    ramp_constraints: bool | None = None,  # default True
    schedule: Sequence[RelaxScheduleEntry] | None = None,
    min_fn: RelaxMinimizer | None = None,
    cuda_graph: bool | None = None,
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
        packer_pallete: Residue types available to the packer.
        move_map: Specifies which DOFs are free to move during minimization.
        fold_forest: Fold forest defining the kinematic connectivity.
        task_operations: In-place task configuration callbacks. By default,
            restrict to repacking with the scoring database’s Dunbrack, fixed-AA,
            current and nucleic-acid rotamers, sampling covalent groups jointly.
        num_repeats: Number of complete pack-minimize ramps.
        ramp_constraints: Ramp an active constraint weight to zero. Defaults
            to true; the input weight is restored after relaxation.
        schedule: Numeric fa_rep fractions or dictionaries with separate pack,
            minimize, and optional constraint fractions. Defaults to
            ``DEFAULT_RELAX_SCHEDULE``.
        min_fn: Minimizer called with the pose, score function, fold forest,
            move map, and verbosity. If it provides ``release_retained_state()``,
            FastRelax calls that hook after extracting each minimized pose so
            packing does not overlap retained scorer or optimizer storage.
            Defaults to Cartesian minimization.
        release_between_stages: Drop the rendered scorer, CUDA graph and L-BFGS
            history after each stage's minimization. On by default: a large
            system is bound by that retained state through the next packing
            phase, which is why the release exists. Turn it off when the setup
            cost dominates instead -- a small pose on the default four-stage
            schedule otherwise pays four network constructions, and four graph
            captures, where one would do.
        cuda_graph: Capture the default Cartesian minimizer's repeated CUDA
            scoring path. By default, enable it automatically for CUDA poses
            containing DNA or RNA, where replay savings exceed capture setup.
            Pass ``True`` for repeated protein workloads where steady-state
            throughput matters more than first-call latency, or ``False`` to
            disable it. It cannot be combined with a custom ``min_fn``.
        verbose: Print timing information for each step.

    Returns:
        Best-scoring relaxed poses across all repeats.
    """
    return _shared_fast_relax(
        execution=_TorchRelaxExecution(),
        pose_stack=pose_stack,
        sfxn=sfxn,
        packer_pallete=packer_pallete,
        move_map=move_map,
        fold_forest=fold_forest,
        task_operations=task_operations,
        num_repeats=num_repeats,
        ramp_constraints=ramp_constraints,
        schedule=schedule,
        min_fn=min_fn,
        cuda_graph=cuda_graph,
        release_between_stages=release_between_stages,
        verbose=verbose,
    )


def _relax_pack_min_step(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    fold_forest: FoldForest,
    move_map: MoveMap | CartesianMoveMap,
    packer_pallete: PackerPalette,
    fa_rep_pack_weight: float,
    fa_rep_min_weight: float,
    cst_weight: float,
    task_operations: Sequence[PackerTaskOperation],
    min_fn: RelaxMinimizer,
    verbose: bool,
    release_between_stages: bool = True,
) -> PoseStack:
    """Execute one weighted packing and minimization stage.

    Args:
        pose_stack: Current poses.
        sfxn: Score function whose repulsive and constraint weights are updated.
        fold_forest: Connectivity passed to the minimizer.
        move_map: Movable degrees of freedom passed to the minimizer.
        packer_pallete: Residue types available during packing.
        fa_rep_pack_weight: Repulsive weight for packing.
        fa_rep_min_weight: Repulsive weight for minimization.
        cst_weight: Constraint weight for both operations.
        task_operations: In-place task configuration callbacks.
        min_fn: Minimization callable.
        verbose: Print synchronized stage timings.

    Returns:
        The minimized pose stack.
    """

    return _shared_relax_step(
        execution=_TorchRelaxExecution(),
        pose_stack=pose_stack,
        sfxn=sfxn,
        fold_forest=fold_forest,
        move_map=move_map,
        packer_pallete=packer_pallete,
        fa_rep_pack_weight=fa_rep_pack_weight,
        fa_rep_min_weight=fa_rep_min_weight,
        cst_weight=cst_weight,
        task_operations=task_operations,
        min_fn=min_fn,
        verbose=verbose,
        release_between_stages=release_between_stages,
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
    def default_minimizer(pose, cuda_graph):
        return DefaultCartesianMinimizer(_resolve_cuda_graph_mode(pose, cuda_graph))

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
