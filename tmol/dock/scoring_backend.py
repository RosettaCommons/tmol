"""Pluggable scoring and local-minimization backends for docking."""

from __future__ import annotations

from abc import ABC, abstractmethod
from contextlib import contextmanager
import copy
from dataclasses import dataclass, replace
from typing import Optional, Sequence

import attrs
import torch

from tmol.database import ParameterDatabase
from tmol.dock._batched_lbfgs import BatchedLBFGS
from tmol.dock._metal_context import (
    ligand_donor_patches,
    ligand_variant_atom_maps,
    reachable_metal_blocks,
)
from tmol.dock.ligand_dofs import (
    GeneIndex,
    LigandDockingDOFs,
    _block_kinforest_nodes,
    _ligand_kinforest_nodes,
    _sidechain_coordinate_indices,
    build_ligand_docking_dofs,
    merge_ligand_variant_dofs,
    per_pose_fold_forest_and_move_map,
    repeat_docking_dofs,
    LigandVariantDOFs,
)
from tmol.kinematics._datatypes import KinForest, NodeType
from tmol.kinematics.compiled._compiled_ops import (
    _forward_only_op,
    _forward_with_ht_op,
    _inverse_kin,
    _kin_deriv_op,
)
from tmol.kinematics._move_map import MinimizerMap
from tmol.kinematics import PoseStackKinematicsModule
from tmol.optimization._minimizers import _build_kinforest_network, _run_min
from tmol.pose._pose_stack import PoseStack
from tmol.pose import repeat_pose_stack
from tmol.pose._inter_block_bondsep import InterBlockBondsep
from tmol.score import beta2016_score_function
from tmol.score.gridscore import GridScorer, GridSet, build_grid
from tmol.score.gridscore.grid import _setup_scoring_parameters
from tmol.score.gridscore.live_hbond import (
    LiveHBondData,
    build_live_hbond_data,
    set_hbond_pose_variants,
)
from tmol.score.gridscore.live_ligand_intra import (
    LiveLigandIntraData,
    build_live_ligand_intra_data,
)
from tmol.score.gridscore.live_lk_ball import (
    LiveLKBallData,
    build_live_lk_ball_data,
    set_lk_ball_pose_variants,
)
from tmol.score import ScoreFunction
from tmol.score._score_types import ScoreType
from tmol.score.gridscore._movable import (
    movable_block_mapping,
    movable_atoms,
    movable_pose_stack,
    resolve_ligand_block,
    sidechain_context_blocks,
    validated_sidechain_blocks,
)


@dataclass(frozen=True)
class MinimizationResult:
    """A minimized pose and its optimizer-authored freed-DOF gene.

    ``genes`` is ``None`` only for backends that do not participate in the GA.
    Continuous minimization always returns the fixed-order jump, ligand-chi,
    and selected-sidechain-chi tensor directly from its final kinematic DOFs.
    """

    pose_stack: PoseStack
    genes: Optional[torch.Tensor]
    masked_dofs: Optional[torch.Tensor] = None
    # Final objective values, when the optimizer already has them (else the GA
    # rescores ``pose_stack``).
    energies: Optional[torch.Tensor] = None


#: How far clear of the structure an absent union slot is parked, in Angstroms.
#: Well past every live-term cutoff, so hbond, LK-ball and ligand-intra drop it
#: on distance alone; the grid masks it instead.


@dataclass(frozen=True)
class LigandVariantAtoms:
    """One ligand block type's union-slot atom data.

    Binding a metal can deprotonate a donor, which retypes it, shifts its
    charges and drops a hydrogen; slots are the union over the block types a
    search can reach.
    """

    #: [slot] ligand-block-local atom index, -1 where this type has no atom
    local_atom: torch.Tensor
    #: [grid atom] grid atom type index, in the backend's grid-scored order
    grid_index: torch.Tensor
    #: [grid atom] partial charge, in the backend's grid-scored order
    charges: torch.Tensor
    #: [grid atom] False where the grid-scored slot has no atom
    grid_present: torch.Tensor


def _genes_from_kinforest_dofs(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
    kf_network,
    variants: Optional[LigandVariantDOFs] = None,
) -> torch.Tensor:
    """Map final optimizer DOFs to six jump DOFs plus logical torsions."""

    kmd = kf_network.kin_module.kmd
    raw_dofs = kf_network.full_dofs.detach().clone()
    raw_dofs.reshape(-1)[kf_network._dof_flat_idx] = kf_network.masked_dofs.detach()
    return _genes_from_raw_kinforest_dofs(
        pose_stack,
        docking_dofs,
        kmd.forest,
        kf_network.dof_mask,
        raw_dofs,
        variants=variants,
    )


def _genes_from_raw_kinforest_dofs(
    pose_stack: PoseStack,
    docking_dofs: LigandDockingDOFs,
    forest: KinForest,
    dof_mask: torch.Tensor,
    raw_dofs: torch.Tensor,
    ligand_nodes: Optional[torch.Tensor] = None,
    variants: Optional[LigandVariantDOFs] = None,
) -> torch.Tensor:
    """Map raw kinematic DOFs directly to the docking gene layout."""

    index = (
        GeneIndex.build(
            pose_stack, docking_dofs, forest, dof_mask, raw_dofs.dtype, ligand_nodes
        )
        if variants is None
        else GeneIndex.build_union(
            pose_stack, variants, forest, dof_mask, raw_dofs.dtype, ligand_nodes
        )
    )
    return index.genes_from_raw_dofs(raw_dofs).to(dtype=pose_stack.coords.dtype)


@attrs.frozen
class _DockingSite:
    """The ligand, flexible sidechains and reachable metals of one complex."""

    #: The reference, on the complete block types when a metal is in reach
    pose_stack: PoseStack
    ligand_block: int
    sidechain_blocks: tuple[int, ...]
    metal_blocks: tuple[int, ...]
    #: Packed block types holding every bound form a search can reach, or None
    complete_block_types: object


def _docking_site(
    reference_pose_stack: PoseStack,
    ligand,
    param_db: ParameterDatabase,
    *,
    flexible_residues: Sequence[int] | None,
    metal_context: bool,
    voxel_spacing: float,
    bbox_padding: float,
) -> _DockingSite:
    """Resolve what a backend docks, the same way for every backend."""
    if reference_pose_stack.n_poses != 1:
        raise ValueError("a docking backend is built from one reference pose")
    ligand_block = resolve_ligand_block(reference_pose_stack, ligand)
    sidechain_blocks = validated_sidechain_blocks(
        reference_pose_stack,
        ligand_block,
        () if flexible_residues is None else flexible_residues,
    )
    metal_blocks = (
        reachable_metal_blocks(
            reference_pose_stack,
            ligand_block,
            voxel_spacing=voxel_spacing,
            bbox_padding=bbox_padding,
        )
        if metal_context
        else ()
    )
    complete_block_types = None
    if metal_blocks:
        # Every bound form the search can reach, built before anything is
        # rendered: detection then only picks among these block types, so
        # no scoring module is rebuilt and no captured graph is invalidated.
        from tmol.io.details._metal_detection import _with_donor_patches

        pbt = reference_pose_stack.packed_block_types
        complete = _with_donor_patches(
            pbt,
            ligand_donor_patches(reference_pose_stack, ligand_block, param_db.chemical),
        )
        if complete is not pbt:
            reference_pose_stack = attrs.evolve(
                reference_pose_stack, packed_block_types=complete
            )
            complete_block_types = complete
    return _DockingSite(
        pose_stack=reference_pose_stack,
        ligand_block=ligand_block,
        sidechain_blocks=tuple(sidechain_blocks),
        metal_blocks=tuple(metal_blocks),
        complete_block_types=complete_block_types,
    )


def _on_complete_block_types(backend, pose_stack: PoseStack) -> PoseStack:
    """``pose_stack`` on the block types every bound form was enumerated into.

    Detection then never grows them. Their leading run is the caller's. The
    palette is built on the reference pose's device; the search runs on its
    own, so a copy is kept per device.
    """
    if backend.complete_block_types is None:
        return pose_stack
    complete = backend._complete_on.get(pose_stack.device)
    if complete is None:
        complete = backend.complete_block_types.to(pose_stack.device)
        backend._complete_on[pose_stack.device] = complete
    return attrs.evolve(pose_stack, packed_block_types=complete)


class ScoringBackend(ABC):
    """The scoring contract used by the docking loop.

    ``score`` returns one search-objective energy per pose, with shape
    ``[n_poses]``. ``minimize`` performs one batched local minimization over
    exactly the freedoms described by ``docking_dofs`` and returns a
    :class:`MinimizationResult`. A grid implementation may use a different
    internal objective and gradient representation, but it must preserve these
    input and output semantics so the population loop does not depend on the
    backend.

    A backend is built for one ligand: ``ligand_block`` is its block index in
    the reference complex, and ``sidechain_blocks`` the receptor residues
    (block indices) whose sidechains move with it. Every backend is built the
    same way, with ``from_pose_stack``, and docks through the same calls.
    """

    ligand_block: int
    sidechain_blocks: tuple[int, ...] = ()
    #: Receptor metals in the ligand's reach, which a search may coordinate
    metal_blocks: tuple[int, ...] = ()
    #: Ligand-extended parameter database; None uses tmol's default
    param_db: Optional[ParameterDatabase] = None

    def search_view(
        self, pose_stack: PoseStack
    ) -> tuple[PoseStack, "ScoringBackend", tuple[int, ...]]:
        """The pose, backend and flexible blocks the GA actually searches over.

        The default searches the whole complex with this backend's own
        sidechains. A backend that searches a smaller pose returns it with a
        backend addressing that pose's block numbering.
        """
        return pose_stack, self, tuple(self.sidechain_blocks)

    @contextmanager
    def rep_scale(self, scale: float):
        """Scale fa_rep for the duration of the block (the GA packing ramp).

        A backend without a repulsive term to soften leaves this a no-op.
        """
        yield

    def set_ligand_variants(
        self, variants: "LigandVariantDOFs", pose_stack: PoseStack
    ) -> None:
        """Register the block types a search can give this ligand.

        A backend that scores one fixed ligand layout ignores this; one that
        scores a pose as the form it holds records the union here.
        """

    def apply_pose_variants(self, pose_stack: PoseStack) -> None:
        """Point the scorer at the ligand form each pose of this population holds.

        A backend that reads its chemistry from the PoseStack on every
        evaluation needs nothing here.
        """

    def set_metal_term(self, pose_stack: PoseStack, weight: float) -> None:
        """Score the ligand's bonds to receptor metals at ``weight``.

        Only a backend whose objective lacks the term needs to render one;
        beta2016 already carries ``metal_coordination`` at its database weight.
        """

    @abstractmethod
    def score(self, pose_stack: PoseStack) -> torch.Tensor:
        """Return the total energy of each pose as a ``[n_poses]`` tensor."""

    @abstractmethod
    def minimize(
        self,
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        *,
        maxiter: Optional[int] = None,
        initial_raw_dofs: Optional[torch.Tensor] = None,
    ) -> MinimizationResult:
        """Minimize and return coordinates plus the final freed-DOF gene."""


class ContinuousBackend(ScoringBackend):
    """Full continuous beta2016 scoring and kinematic minimization.

    The search runs on the whole complex, so every receptor interaction is
    scored exactly; it is slower per evaluation than :class:`GridBackend` and
    needs no grid. Build it with :meth:`from_pose_stack`, as for the grid
    backend; the two dock through the same calls and options.

    ``param_db`` should be the ligand-extended parameter database returned by
    ligand preparation; omitted, tmol's default database is used. The backend
    scores with its own copy of beta2016 per device, so the repulsive ramp and
    the search's metal weight never write into the shared memoized one.

    Args:
        ligand_block: Block index of the ligand in the complex to be docked.
        param_db: Optional ligand-extended scoring parameter database.
        sidechain_blocks: Receptor residues whose chis the search moves.
        metal_blocks: Receptor metals the ligand may coordinate.
        complete_block_types: Packed block types holding every bound form the
            search can reach (see :meth:`from_pose_stack`).
        kin_dtype: Internal-coordinate dtype for minimization. By default the
            input coordinate dtype is used.
        value_only_line_search: Evaluate Armijo trials without gradients and
            refresh the gradient at the accepted point (default: False).
    """

    def __init__(
        self,
        ligand_block: int,
        param_db: Optional[ParameterDatabase] = None,
        *,
        sidechain_blocks: Sequence[int] = (),
        metal_blocks: Sequence[int] = (),
        complete_block_types=None,
        kin_dtype: Optional[torch.dtype] = None,
        value_only_line_search: bool = False,
    ) -> None:
        if kin_dtype not in (None, torch.float32, torch.float64):
            raise ValueError("kin_dtype must be torch.float32 or torch.float64")
        self.ligand_block = ligand_block
        self.param_db = param_db
        self.sidechain_blocks = tuple(int(b) for b in sidechain_blocks)
        self.metal_blocks = tuple(int(b) for b in metal_blocks)
        self.complete_block_types = complete_block_types
        self._complete_on: dict = {}
        self.kin_dtype = kin_dtype
        self.value_only_line_search = value_only_line_search
        self._score_functions: dict[torch.device, ScoreFunction] = {}
        self._base_weights: dict = {}
        self._rep_scale = 1.0
        #: Search weight on metal coordination; None keeps beta2016's own
        self._metal_weight: Optional[float] = None
        self.ligand_variant_dofs: Optional[LigandVariantDOFs] = None
        self._single_variant_dofs: dict = {}

    @classmethod
    def from_pose_stack(
        cls,
        reference_pose_stack: PoseStack,
        ligand,
        param_db: ParameterDatabase,
        *,
        flexible_residues: Sequence[int] | None = None,
        metal_context: bool = True,
        voxel_spacing: float = 0.25,
        bbox_padding: float = 4.0,
        kin_dtype: Optional[torch.dtype] = None,
        value_only_line_search: bool = False,
    ) -> "ContinuousBackend":
        """Build the backend for one ligand of a complex.

        Takes the same arguments as :meth:`GridBackend.from_pose_stack` where
        they apply. ``voxel_spacing`` and ``bbox_padding`` only set the box
        whose metals count as within the ligand's reach, as the grid's box does.
        """
        site = _docking_site(
            reference_pose_stack,
            ligand,
            param_db,
            flexible_residues=flexible_residues,
            metal_context=metal_context,
            voxel_spacing=voxel_spacing,
            bbox_padding=bbox_padding,
        )
        return cls(
            site.ligand_block,
            param_db,
            sidechain_blocks=site.sidechain_blocks,
            metal_blocks=site.metal_blocks,
            complete_block_types=site.complete_block_types,
            kin_dtype=kin_dtype,
            value_only_line_search=value_only_line_search,
        )

    def search_view(
        self, pose_stack: PoseStack
    ) -> tuple[PoseStack, "ContinuousBackend", tuple[int, ...]]:
        """The whole complex, on the block types every bound form needs."""
        return (
            _on_complete_block_types(self, pose_stack),
            self,
            self.sidechain_blocks,
        )

    def _score_function(self, device: torch.device) -> ScoreFunction:
        device = torch.device(device)
        sfxn = self._score_functions.get(device)
        if sfxn is None:
            param_db = self.param_db or ParameterDatabase.get_default()
            beta = beta2016_score_function(device, param_db=param_db)
            sfxn = ScoreFunction(param_db, device)
            for score_type in beta.all_score_types():
                weight = float(beta.get_weight(score_type))
                if weight:
                    sfxn.set_weight(score_type, weight)
            self._base_weights = {
                ScoreType.fa_ljrep: float(beta.get_weight(ScoreType.fa_ljrep)),
                ScoreType.metal_coordination: float(
                    beta.get_weight(ScoreType.metal_coordination)
                ),
            }
            self._score_functions[device] = sfxn
            self._apply_weights(sfxn)
        return sfxn

    def _apply_weights(self, sfxn: ScoreFunction) -> None:
        sfxn.set_weight(
            ScoreType.fa_ljrep, self._base_weights[ScoreType.fa_ljrep] * self._rep_scale
        )
        sfxn.set_weight(
            ScoreType.metal_coordination,
            (
                self._base_weights[ScoreType.metal_coordination]
                if self._metal_weight is None
                else self._metal_weight
            ),
        )

    @contextmanager
    def rep_scale(self, scale: float):
        """Scale fa_rep for the duration of the block, as the grid backend does."""
        previous = self._rep_scale
        self._rep_scale = float(scale)
        for sfxn in self._score_functions.values():
            self._apply_weights(sfxn)
        try:
            yield
        finally:
            self._rep_scale = previous
            for sfxn in self._score_functions.values():
                self._apply_weights(sfxn)

    def set_metal_term(self, pose_stack: PoseStack, weight: float) -> None:
        """Score the ligand's metal bonds at the search's ``weight``.

        beta2016 already carries the term; only its weight changes.
        """
        self._metal_weight = float(weight)
        for sfxn in self._score_functions.values():
            self._apply_weights(sfxn)

    def set_ligand_variants(
        self, variants: "LigandVariantDOFs", pose_stack: PoseStack
    ) -> None:
        """Record the ligand forms this search can reach.

        Scoring reads each pose's chemistry from the PoseStack, so only
        minimization needs these: they carry the per-form kinematics.
        """
        self.ligand_variant_dofs = variants

    def _minimized_variants(
        self, pose_stack: PoseStack, docking_dofs: LigandDockingDOFs
    ) -> Optional[LigandVariantDOFs]:
        """The union layout, when this population holds more than one form.

        A population agreeing on its ligand form minimizes on the DOFs it was
        handed, which is both the common case and the one the caller built.
        """
        if self.ligand_variant_dofs is None:
            return None
        block_types = pose_stack.block_type_ind64[:, docking_dofs.ligand_block]
        present = set(block_types.tolist())
        if len(present) == 1:
            return None
        unknown = sorted(present - set(self.ligand_variant_dofs.dofs))
        if unknown:
            raise ValueError(f"ligand block types {unknown} have no docking DOFs")
        return self.ligand_variant_dofs

    def score(self, pose_stack: PoseStack) -> torch.Tensor:
        """Score a PoseStack with the weighted continuous beta2016 function."""

        sfxn = self._score_function(pose_stack.device)
        scorer = sfxn.render_whole_pose_scoring_module(pose_stack)
        with torch.no_grad():
            energies = scorer(pose_stack.coords)
        expected_shape = (pose_stack.n_poses,)
        if energies.shape != expected_shape:
            raise RuntimeError(
                "continuous whole-pose scorer returned shape "
                f"{tuple(energies.shape)}; expected {expected_shape}"
            )
        return energies

    def minimize(
        self,
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        *,
        maxiter: Optional[int] = None,
        initial_raw_dofs: Optional[torch.Tensor] = None,
    ) -> MinimizationResult:
        """Run docking-DOF minimization and return optimizer-authored genes."""
        kin_dtype = self.kin_dtype or pose_stack.coords.dtype
        variants = self._minimized_variants(pose_stack, docking_dofs)
        if variants is None:
            fold_forest, move_map = docking_dofs.fold_forest, docking_dofs.move_map
        else:
            # The poses hold different ligand forms, whose torsions and jump
            # stubs differ: each takes its own form's rows.
            fold_forest, move_map = per_pose_fold_forest_and_move_map(
                pose_stack,
                docking_dofs.ligand_block,
                sidechain_blocks=docking_dofs.sidechain_blocks,
            )
        kf_network = _build_kinforest_network(
            pose_stack,
            self._score_function(pose_stack.device),
            fold_forest,
            move_map,
            kin_dtype=kin_dtype,
        )
        if initial_raw_dofs is not None:
            base_dofs = kf_network._kinematic_scoring.base_dofs
            if initial_raw_dofs.shape != base_dofs.shape:
                raise ValueError(
                    "initial raw DOF shape does not match the minimizer: "
                    f"{tuple(initial_raw_dofs.shape)} != {tuple(base_dofs.shape)}"
                )
            initial = initial_raw_dofs.to(dtype=kin_dtype).detach()
            with torch.no_grad():
                base_dofs.copy_(initial)
                kf_network.masked_dofs.copy_(initial[kf_network.dof_mask])
        optimizer_kwargs = {"value_only_line_search": self.value_only_line_search}
        if maxiter is not None:
            optimizer_kwargs["max_iter"] = maxiter
        minimized = _run_min(
            kf_network,
            optimizer_kwargs=optimizer_kwargs,
        )
        minimized.coords = minimized.coords.detach()
        genes = _genes_from_kinforest_dofs(
            pose_stack, docking_dofs, kf_network, variants=variants
        ).detach()
        return MinimizationResult(
            pose_stack=minimized,
            genes=genes,
            masked_dofs=kf_network.masked_dofs.detach().clone(),
        )


class _LightweightGridObjective:
    """Ligand-grid energy and analytic docking-DOF gradient of a population."""

    def __init__(
        self,
        backend: "GridBackend",
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        kin_dtype: torch.dtype,
        initial_raw_dofs: Optional[torch.Tensor],
        variants: Optional[LigandVariantDOFs] = None,
    ) -> None:
        self.backend = backend
        self.pose_stack = pose_stack
        self.docking_dofs = docking_dofs
        kin_module = PoseStackKinematicsModule(pose_stack, docking_dofs.fold_forest)
        full_kmd = kin_module.kmd
        self.forest = full_kmd.forest
        self.dof_mask = MinimizerMap(
            pose_stack, full_kmd, docking_dofs.move_map
        ).dof_mask

        if initial_raw_dofs is not None:
            expected_shape = (full_kmd.forest.id.shape[0], 9)
            if initial_raw_dofs.shape != expected_shape:
                raise ValueError(
                    "initial raw DOF shape does not match the minimizer: "
                    f"{tuple(initial_raw_dofs.shape)} != {expected_shape}"
                )
            full_dofs = None
        else:
            kin_coords = torch.zeros(
                (full_kmd.forest.id.shape[0], 3),
                dtype=kin_dtype,
                device=pose_stack.device,
            )
            kin_coords[1:] = pose_stack.coords.reshape(-1, 3)[
                full_kmd.forest.id[1:].to(torch.int64)
            ].to(kin_dtype)
            full_dofs = _inverse_kin(
                kin_coords,
                full_kmd.forest.parent,
                full_kmd.forest.frame_x,
                full_kmd.forest.frame_y,
                full_kmd.forest.frame_z,
                full_kmd.forest.doftype,
            )
        self.variants = variants
        ligand_map = backend.ligand_local_atom_map(pose_stack)
        ligand_block_type = pose_stack.block_type(0, docking_dofs.ligand_block)
        self.ligand_nodes = _ligand_kinforest_nodes(
            pose_stack,
            docking_dofs.ligand_block,
            ligand_block_type.n_atoms,
            self.forest.id,
            ligand_map,
        )
        # Reuse the already-validated full-pose ordering. All compiled operations
        # run under no_grad, so retaining fixed receptor nodes costs forward
        # buffers but does not retain an autograd graph across L-BFGS iterations.
        # The full ordering is retained so ligand and selected sidechain
        # coordinate gradients can both be propagated to their freed DOFs.
        self.nodes_f = kin_module.nodes_f
        self.scans_f = kin_module.scans_f
        self.gens_f = kin_module.gens_f
        self.nodes_b = kin_module.nodes_b
        self.scans_b = kin_module.scans_b
        self.gens_b = kin_module.gens_b
        self.kintree = kin_module.kinforest

        self.dof_flat_idx, self.dof_pad_width = self._freed_dofs(pose_stack)
        if initial_raw_dofs is None:
            self.full_dofs = full_dofs.detach().clone()
        else:
            self.full_dofs = initial_raw_dofs.to(dtype=kin_dtype).detach().clone()
        self.initial_masked_dofs = self.full_dofs.reshape(-1)[self.dof_flat_idx].clone()

        self.ligand_atom_indices = backend._ligand_atom_indices(
            pose_stack, docking_dofs.ligand_block
        )
        self.sidechain_nodes, self.sidechain_atom_indices = (
            _sidechain_coordinate_indices(pose_stack, docking_dofs, self.forest)
        )
        if backend.sidechain_blocks or backend.metal_blocks:
            movable_node_parts = []
            ligand_map = backend.ligand_local_atom_map(pose_stack)
            for block in range(pose_stack.max_n_blocks):
                block_type = pose_stack.block_type(0, block)
                is_ligand = block == docking_dofs.ligand_block
                movable_node_parts.append(
                    _block_kinforest_nodes(
                        pose_stack,
                        block,
                        block_type.n_atoms,
                        self.forest.id,
                        ligand_map if is_ligand else None,
                    )
                )
            self.movable_nodes = torch.cat(movable_node_parts, dim=1)
        else:
            self.movable_nodes = self.ligand_nodes
        # A pose whose ligand deprotonated has no atom for some union slots;
        # those are parked clear of every cutoff instead of being scored.
        self.movable_present = self.movable_nodes >= 0
        self.movable_all_present = bool(self.movable_present.all())
        self.movable_present_flat = (
            self.movable_present.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        )
        self.movable_present_nodes = self.movable_nodes.reshape(-1)[
            self.movable_present_flat
        ]

        pose_for_node = torch.div(
            self.forest.id,
            pose_stack.max_n_pose_atoms,
            rounding_mode="floor",
        )
        pose_for_dof = (
            pose_for_node.unsqueeze(1).expand(self.full_dofs.shape).reshape(-1)
        )
        if self.dof_pad_width is None:
            self.segment_ids = pose_for_dof[self.dof_flat_idx].to(torch.int64)
        else:
            self.segment_ids = torch.arange(
                pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
            ).repeat_interleave(self.dof_pad_width)
        if bool(torch.any(self.segment_ids < 0)):
            raise ValueError("root node DOFs cannot be minimized")

    def _freed_dofs(self, pose_stack: PoseStack):
        """Flat indices of the freed DOFs, one equal-length run per pose.

        Poses holding different metal variants free different numbers of DOFs,
        and the batched line search reads its vector as ``[trial, pose, dof]``.
        Short poses are padded with an unused slot of a bond node, which the
        kinematics never reads, so every pose contributes the same width.
        """
        n_columns = self.dof_mask.shape[1]
        flat = self.dof_mask.reshape(-1).nonzero(as_tuple=False).squeeze(-1)
        pose_of = (
            torch.div(
                self.forest.id, pose_stack.max_n_pose_atoms, rounding_mode="floor"
            )
            .to(torch.int64)
            .repeat_interleave(n_columns)[flat]
        )
        counts = torch.bincount(pose_of.clamp(min=0), minlength=pose_stack.n_poses)[
            : pose_stack.n_poses
        ]
        if pose_stack.n_poses < 2 or bool((counts == counts[0]).all()):
            return flat, None
        bond = torch.nonzero(self.forest.doftype != int(NodeType.jump))
        if bond.numel() == 0:
            raise ValueError("no bond node to park a padded DOF on")
        dead = int(bond[0]) * n_columns + n_columns - 1
        width = int(counts.max())
        runs = [flat[pose_of == pose] for pose in range(pose_stack.n_poses)]
        return (
            torch.cat(
                [
                    (
                        run
                        if run.numel() == width
                        else torch.cat(
                            (run, run.new_full((width - run.numel(),), dead))
                        )
                    )
                    for run in runs
                ]
            ),
            width,
        )

    def _set_masked_dofs(self, masked_dofs: torch.Tensor) -> None:
        if masked_dofs.shape != self.initial_masked_dofs.shape:
            raise ValueError(
                "masked DOFs have shape "
                f"{tuple(masked_dofs.shape)}; expected "
                f"{tuple(self.initial_masked_dofs.shape)}"
            )
        self.full_dofs.reshape(-1).index_copy_(
            0, self.dof_flat_idx, masked_dofs.detach()
        )

    def _forward_only(self, masked_dofs: torch.Tensor) -> torch.Tensor:
        self._set_masked_dofs(masked_dofs)
        return _forward_only_op(
            self.full_dofs,
            self.nodes_f,
            self.scans_f,
            self.gens_f,
            self.kintree,
        )

    def energy_and_gradient(
        self, masked_dofs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self._set_masked_dofs(masked_dofs)
            kin_coords, hts = _forward_with_ht_op(
                self.full_dofs,
                self.nodes_f,
                self.scans_f,
                self.gens_f,
                self.kintree,
            )
            # A slot this pose's ligand has no atom for keeps whatever its
            # node held: every term is told the slot is absent and reads none
            # of it, so no placeholder coordinate is needed.
            movable_coords = kin_coords[self.movable_nodes.clamp(min=0)]
            energies, movable_gradients = (
                self.backend._score_ligand_coordinates_with_gradients(movable_coords)
            )
            coordinate_gradients = torch.zeros_like(kin_coords)
            if self.movable_all_present:
                coordinate_gradients[self.movable_nodes.reshape(-1)] = (
                    movable_gradients.reshape(-1, 3)
                )
            else:
                # Integer indices, resolved once off the capture path: a
                # boolean mask would size its output on the device.
                coordinate_gradients[self.movable_present_nodes] = (
                    movable_gradients.reshape(-1, 3)[self.movable_present_flat]
                )
            dof_gradients = _kin_deriv_op(
                coordinate_gradients,
                hts,
                self.full_dofs,
                self.nodes_b,
                self.scans_b,
                self.gens_b,
                self.kintree,
            )
            masked_gradients = dof_gradients.reshape(-1)[self.dof_flat_idx]
        return energies, masked_gradients

    def final_result(self, masked_dofs: torch.Tensor) -> MinimizationResult:
        """Build output coordinates and gene directly from final raw DOFs."""

        with torch.no_grad():
            kin_coords = self._forward_only(masked_dofs)
            full_coords = self.pose_stack.coords.detach().clone()
            pose_indices = torch.arange(
                self.pose_stack.n_poses,
                dtype=torch.int64,
                device=self.pose_stack.device,
            )
            ligand_xyz = kin_coords[self.ligand_nodes.clamp(min=0)].to(
                full_coords.dtype
            )
            present = self.ligand_nodes >= 0
            if bool(present.all()):
                full_coords[pose_indices[:, None], self.ligand_atom_indices] = (
                    ligand_xyz
                )
            else:
                rows = pose_indices[:, None].expand_as(self.ligand_atom_indices)
                full_coords[rows[present], self.ligand_atom_indices[present]] = (
                    ligand_xyz[present]
                )
            if self.sidechain_nodes.shape[1]:
                full_coords[pose_indices[:, None], self.sidechain_atom_indices] = (
                    kin_coords[self.sidechain_nodes].to(full_coords.dtype)
                )
            minimized_pose = attrs.evolve(self.pose_stack, coords=full_coords)
            genes = _genes_from_raw_kinforest_dofs(
                self.pose_stack,
                self.docking_dofs,
                self.forest,
                self.dof_mask,
                self.full_dofs,
                self.ligand_nodes,
                self.variants,
            ).detach()
        return MinimizationResult(
            pose_stack=minimized_pose,
            genes=genes,
            masked_dofs=masked_dofs.detach().clone(),
        )


def variant_docking_dofs(
    backend, pose: PoseStack, ligand_block: int, sidechain_blocks, n_poses: int
) -> LigandDockingDOFs:
    """DOFs of ``pose``'s ligand block type for ``n_poses`` poses.

    Resolved once per block type on the single ``pose`` and tiled; the result
    depends on neither its coordinates nor the batch size beyond the tiling.
    """
    index = int(pose.block_type_ind64[0, ligand_block])
    key = (index, tuple(sidechain_blocks), ligand_block)
    cache = backend._single_variant_dofs
    if key not in cache:
        cache[key] = build_ligand_docking_dofs(
            pose, ligand_block, sidechain_blocks=tuple(sidechain_blocks)
        )
    return repeat_docking_dofs(cache[key], n_poses)


def _candidate_variants(
    backend, candidates: PoseStack, ligand_block: int, sidechain_blocks
) -> Optional[LigandVariantDOFs]:
    """Union DOFs resolved against the line search's own, wider stack.

    Torsion offsets, the fold forest and the move map are all per pose, so the
    pool's table cannot be reused at a different batch size. A variant's DOFs depend only on the batch size and the flexible sidechains,
    not on any coordinate nor on which pose holds it, so each is resolved once
    on a single pose, tiled, and cached; only the merge is redone.
    """
    if backend.ligand_variant_dofs is None:
        return None
    block_types = candidates.block_type_ind64[:, ligand_block]
    present = sorted(set(block_types.tolist()))
    sidechain_blocks = tuple(sidechain_blocks)
    dofs = {}
    for index in present:
        key = (candidates.n_poses, index, sidechain_blocks, ligand_block)
        if key not in backend._candidate_variant_cache:
            pose = int(torch.nonzero(block_types == index)[0])
            backend._candidate_variant_cache[key] = variant_docking_dofs(
                backend,
                candidates.split(pose),
                ligand_block,
                sidechain_blocks,
                candidates.n_poses,
            )
        dofs[index] = backend._candidate_variant_cache[key]
    return merge_ligand_variant_dofs(
        dofs, candidates.packed_block_types.active_block_types, candidates.device
    )


def _tiled_candidates(pose_stack: PoseStack, n_trials: int) -> PoseStack:
    """``n_trials`` copies of the pool, pose-major, each keeping its own type.

    The line search scores every pose at several step sizes at once and reads
    the result as ``[trial, pose]``. Replicating one pose would give the whole
    batch that pose's topology, which is wrong as soon as detection leaves the
    pool holding different metal variants of the ligand.
    """
    if pose_stack.n_poses == 1:
        return repeat_pose_stack(pose_stack, n_trials)

    def tile(value: torch.Tensor) -> torch.Tensor:
        return value.repeat((n_trials,) + (1,) * (value.ndim - 1))

    tensors = (
        "coords",
        "block_coord_offset",
        "block_coord_offset64",
        "inter_residue_connections",
        "inter_residue_connections64",
        "block_type_ind",
        "block_type_ind64",
        "chain_id",
        "chain_id64",
    )
    bondsep = pose_stack.inter_block_bondsep
    return attrs.evolve(
        pose_stack,
        **{name: tile(getattr(pose_stack, name)) for name in tensors},
        inter_block_bondsep=InterBlockBondsep.concatenate(
            [bondsep] * n_trials, bondsep.max_n_blocks, bondsep.max_n_conn
        ),
    )


def _write_into(buffers: Optional[tuple], values: Optional[tuple]) -> Optional[tuple]:
    """``values`` copied into ``buffers`` when every shape matches, else ``values``."""
    if values is None or buffers is None or len(buffers) != len(values):
        return values
    if any(b.shape != v.shape or b.dtype != v.dtype for b, v in zip(buffers, values)):
        return values
    for b, v in zip(buffers, values):
        b.copy_(v)
    return buffers


_HBOND_POSE_TABLES = (
    "ligand_donor_atoms",
    "ligand_donor_types",
    "ligand_acceptor_atoms",
    "ligand_acceptor_types",
)
_LK_BALL_POSE_TABLES = (
    "ligand_polar_atoms",
    "ligand_water_kinds",
    "ligand_water_atoms",
    "ligand_water_params",
    "ligand_water_is_ring",
)


def _kinematic_layout(objective: _LightweightGridObjective) -> tuple:
    """Every index tensor a population's kinematics and gradient scatter read."""
    return (
        objective.nodes_f,
        objective.scans_f,
        objective.gens_f,
        objective.nodes_b,
        objective.scans_b,
        objective.gens_b,
        objective.kintree,
        objective.dof_flat_idx,
        objective.segment_ids,
        objective.movable_nodes,
        objective.ligand_nodes,
    )


def _same_tensors(a: tuple, b: tuple) -> bool:
    return len(a) == len(b) and all(
        x.shape == y.shape and x.dtype == y.dtype and bool(torch.equal(x, y))
        for x, y in zip(a, b)
    )


class _PopulationMinimizer:
    """Batched L-BFGS over ``n_trials`` line-search candidates of every pose.

    One candidate objective holds ``n_trials`` copies of the population, so a
    single (graph) evaluation scores every trial step of every pose.
    """

    def __init__(
        self,
        backend: "GridBackend",
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        kin_dtype: torch.dtype,
    ) -> None:
        cuda = pose_stack.device.type == "cuda"
        n_trials, n_poses = (4 if cuda else 2), pose_stack.n_poses
        candidates = _tiled_candidates(pose_stack, n_trials)
        candidate_variants = _candidate_variants(
            backend,
            candidates,
            docking_dofs.ligand_block,
            docking_dofs.sidechain_blocks,
        )
        candidate_dofs = (
            candidate_variants.reference
            if candidate_variants is not None
            else build_ligand_docking_dofs(
                candidates,
                docking_dofs.ligand_block,
                sidechain_blocks=docking_dofs.sidechain_blocks,
            )
        )
        self.objective = _LightweightGridObjective(
            backend,
            candidates,
            candidate_dofs,
            kin_dtype,
            None,
            variants=candidate_variants,
        )
        self.n_trials = n_trials
        shape = (n_trials, n_poses, -1)

        def evaluate(x):
            energy, gradient = self.objective.energy_and_gradient(x.reshape(-1))
            return energy.reshape(n_trials, n_poses).to(x.dtype), gradient.reshape(
                shape
            )

        x0 = self.objective.initial_masked_dofs.reshape(n_trials, n_poses, -1)[0]
        self.lbfgs = BatchedLBFGS(
            evaluate,
            x0.clone(),
            n_trials=n_trials,
            graph=backend.cuda_graph and cuda,
        )

    def minimize(
        self, fixed_dofs: torch.Tensor, x0: torch.Tensor, max_iter: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Minimize ``x0`` (freed DOFs); ``fixed_dofs`` are the population's raw DOFs.

        Returns the final freed DOFs and the objective values there.
        """
        self.objective.full_dofs.copy_(
            torch.cat((fixed_dofs[:1], fixed_dofs[1:].repeat(self.n_trials, 1)))
        )
        x = self.lbfgs.minimize(x0, max_iter)
        return x, self.lbfgs.f.clone()


class GridBackend(ScoringBackend):
    """Rigid-receptor grid plus live hbond, intra, and LK-ball scoring.

    One :class:`GridSet` is retained by the backend and shared by every pose.
    Device/dtype copies are cached once and do not scale with population size.
    The grid contains fa_atr, fa_rep, fa_sol, and fa_elec, plus three weighted
    LK-ball grid families when ``lk_ball_grid`` is true. One receptor hbond
    structure, ligand-intra topology, and ligand-water topology are shared by
    every ligand pose; ``lk_ball_grid=False`` retains the exact live LK-ball
    implementation. Ligand-intra search weights match Rosetta GridScorer's
    softened ``oneb_wtd_`` objective.

    Minimization is a batched L-BFGS whose line search evaluates ``n_trials``
    step sizes of every pose at once; with ``cuda_graph`` each iteration is one
    CUDA-graph replay.
    """

    def __init__(
        self,
        grids: GridSet,
        ligand_block: int,
        ligand_atom_types: tuple[str, ...],
        ligand_charges: torch.Tensor,
        param_db: Optional[ParameterDatabase] = None,
        *,
        kin_dtype: Optional[torch.dtype] = None,
        live_hbond: Optional[LiveHBondData] = None,
        live_ligand_intra: Optional[LiveLigandIntraData] = None,
        live_lk_ball: Optional[LiveLKBallData] = None,
        lk_ball_grid: bool = False,
        score_function: Optional[ScoreFunction] = None,
        sidechain_blocks: tuple[int, ...] = (),
        context_blocks: tuple[int, ...] = (),
        metal_blocks: tuple[int, ...] = (),
        complete_block_types=None,
        grid_atom_indices: Optional[torch.Tensor] = None,
        movable_n_atoms: Optional[int] = None,
        cuda_graph: bool = True,
    ) -> None:
        self.grids = grids
        self.ligand_block = ligand_block
        self.ligand_atom_types = tuple(ligand_atom_types)
        self.ligand_charges = ligand_charges.detach().to("cpu", torch.float64)
        self.param_db = param_db
        self.kin_dtype = kin_dtype
        self.live_hbond = live_hbond
        self.live_ligand_intra = live_ligand_intra
        self.live_lk_ball = live_lk_ball
        self.lk_ball_grid = lk_ball_grid
        self.score_function = score_function
        self.sidechain_blocks = tuple(sidechain_blocks)
        self.context_blocks = tuple(context_blocks)
        #: Context blocks retained because the ligand may coordinate them
        self.metal_blocks = tuple(metal_blocks)
        #: Packed block types holding every bound form the search can reach,
        #: or None when no metal is in range; search_view scores on these.
        self.complete_block_types = complete_block_types
        self._complete_on: dict = {}
        #: Candidate-stack variant DOFs, by batch size and block types present.
        self._candidate_variant_cache: dict = {}
        self._single_variant_dofs: dict = {}
        self.grid_atom_indices = grid_atom_indices
        self.movable_n_atoms = movable_n_atoms or len(ligand_atom_types)
        self.cuda_graph = cuda_graph
        #: block type index -> its union-slot atom data, when a search can
        #: change the ligand's block type; empty keeps one fixed molecule
        self.ligand_variants: dict[int, LigandVariantAtoms] = {}
        #: Union gene columns shared by those block types, when a search can
        #: change the ligand's; None leaves every path on one topology
        self.ligand_variant_dofs: Optional[LigandVariantDOFs] = None
        #: (grid index, charges, presence) per pose, or None for one molecule
        self._pose_atom_state: Optional[tuple[torch.Tensor, ...]] = None
        self._pose_intra_state: Optional[tuple] = None
        #: Rendered metal coordination module, rebuilt whenever detection moves
        #: the ligand's bonds; None leaves coordination to finalization alone
        self._metal_module = None
        self._metal_n_poses = 0
        self._metal_index: Optional[torch.Tensor] = None
        self._metal_present: Optional[torch.Tensor] = None
        #: Bumped on every render. A captured minimizer graph holds the module's
        #: tensors by address, so it may not outlive one.
        self._metal_render = 0
        self._metal_stack: Optional[PoseStack] = None
        self._metal_shape = None
        self._rep_scale = 1.0
        self._minimizers: dict = {}  # shared with ``for_ligand_block`` views
        self._scorers: dict[tuple[torch.device, torch.dtype], GridScorer] = {}
        self._atom_index_cache: dict = {}

    def search_view(
        self, pose_stack: PoseStack
    ) -> tuple[PoseStack, "GridBackend", tuple[int, ...]]:
        """Compact ligand(+flexible sidechain) pose the GA searches over."""
        pose_stack = _on_complete_block_types(self, pose_stack)
        compact_sidechains: tuple[int, ...] = ()
        if self.sidechain_blocks:
            _, compact_sidechains, _ = movable_block_mapping(
                pose_stack, self.ligand_block, self.sidechain_blocks, self.metal_blocks
            )
        compact = movable_pose_stack(
            pose_stack, self.ligand_block, self.sidechain_blocks, self.metal_blocks
        )
        return compact, self.for_ligand_block(0), compact_sidechains

    def for_ligand_block(self, ligand_block: int) -> "GridBackend":
        """Return a lightweight view targeting another ligand block index.

        Grid construction fixes the ligand chemistry, not its block position in
        a scoring PoseStack. The ligand-only docking representation places the
        same block type at index zero. A shallow copy preserves the single grid
        and its device/dtype caches while allowing that representation to use
        the ordinary backend contract, whose ``score`` method has no block
        argument.
        """
        result = copy.copy(self)
        result.ligand_block = ligand_block
        return result

    @staticmethod
    def pocket_grids(
        pose_stack: PoseStack,
        ligand,
        param_db: ParameterDatabase,
        *,
        atom_types: Sequence[str] | None = None,
        flexible_residues: Sequence[int] | None = None,
        device: str | torch.device | None = None,
        voxel_spacing: float = 0.25,
        bbox_padding: float = 4.0,
        lk_ball_grid: bool = True,
        score_function: Optional[ScoreFunction] = None,
        extra_excluded_blocks: Sequence[int] = (),
    ) -> GridSet:
        """Receptor grids of the box around ``ligand``, to share across ligands.

        The grids depend on the receptor, the box and the probe atom types, not
        on the ligand: build them once for the union of the ligands' atom types
        and pass them to every :meth:`from_pose_stack` of a screen
        (``atom_types`` defaults to the ligand's own types).
        """
        block = resolve_ligand_block(pose_stack, ligand)
        flexible = validated_sidechain_blocks(
            pose_stack, block, () if flexible_residues is None else flexible_residues
        )
        if atom_types is None:
            block_type = pose_stack.block_type(0, block)
            atom_types = [
                atom.atom_type for atom in block_type.atoms[: block_type.n_atoms]
            ]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return build_grid(
            pose_stack,
            tuple(sorted(set(atom_types))),
            param_db,
            ligand_name3=pose_stack.block_type(0, block).name3,
            voxel_spacing=voxel_spacing,
            bbox_padding=bbox_padding,
            build_lk_ball=lk_ball_grid,
            score_function=score_function,
            # Anything the compacted pose retains is scored live, so it must
            # not also sit in the rigid receptor grid.
            excluded_receptor_blocks=tuple(
                sorted(
                    set(flexible)
                    | set(sidechain_context_blocks(pose_stack, block, flexible))
                    | set(int(b) for b in extra_excluded_blocks)
                )
            ),
            device=device,
        )

    @classmethod
    def from_pose_stack(
        cls,
        reference_pose_stack: PoseStack,
        ligand,
        param_db: ParameterDatabase,
        *,
        flexible_residues: Sequence[int] | None = None,
        grids: GridSet | None = None,
        grid_device: str | torch.device | None = None,
        voxel_spacing: float = 0.25,
        bbox_padding: float = 4.0,
        kin_dtype: Optional[torch.dtype] = None,
        lk_ball_grid: bool = True,
        include_ligand_intra: bool = True,
        cuda_graph: bool = True,
        ligand_intra_weight_profile: str = "search",
        score_function: Optional[ScoreFunction] = None,
        extra_atom_types: Sequence[str] = (),
        metal_context: bool = True,
    ) -> "GridBackend":
        """Build the receptor grid and live terms for one ligand of a complex.

        Args:
            reference_pose_stack: One pose holding the receptor and the ligand
                (in any placement; the search starts from random poses).
            ligand: The ligand: residue name (``"ATP"``), block index, or a
                boolean atom mask selecting exactly its atoms.
            param_db: Scoring database including the ligand's parameters.
            flexible_residues: Receptor residues (block indices, e.g. from
                :func:`select_pocket_residues`) whose sidechains move.
            grids: Receptor grids to reuse instead of building them, e.g. one
                set shared by every ligand of a screen (see
                :meth:`pocket_grids`); must cover this ligand's atom types and
                have been built for the same flexible residues.
            grid_device: Device the grids are built on (default: CUDA if
                available, else CPU).

        ``lk_ball_grid=True`` selects Rosetta's weighted gridded approximation;
        ``False`` retains the exact live LK-ball path. Ligand-intra scoring is
        enabled by default with Rosetta's softened search profile.
        ``cuda_graph`` captures each optimizer iteration as one CUDA graph
        (ignored on CPU).
        """

        site = _docking_site(
            reference_pose_stack,
            ligand,
            param_db,
            flexible_residues=flexible_residues,
            metal_context=metal_context,
            voxel_spacing=voxel_spacing,
            bbox_padding=bbox_padding,
        )
        reference_pose_stack = site.pose_stack
        ligand_block = site.ligand_block
        sidechain_blocks = site.sidechain_blocks
        metal_blocks = site.metal_blocks
        complete_block_types = site.complete_block_types
        pbt = reference_pose_stack.packed_block_types
        block_type = pbt.active_block_types[
            int(reference_pose_stack.block_type_ind64[0, ligand_block])
        ]
        context_blocks = tuple(
            sorted(
                (
                    set(
                        sidechain_context_blocks(
                            reference_pose_stack, ligand_block, sidechain_blocks
                        )
                    )
                    | set(metal_blocks)
                )
                - {ligand_block}
                - set(sidechain_blocks)
            )
        )
        moving = movable_atoms(
            reference_pose_stack, ligand_block, sidechain_blocks, metal_blocks
        )

        # Compact-pose atoms the GA moves: the ligand, plus flexible chi atoms.
        # Retained metals compact the pose too, even with no flexible sidechain.
        grid_atom_indices = None
        if sidechain_blocks or metal_blocks:
            movable_pose = movable_pose_stack(
                reference_pose_stack, ligand_block, sidechain_blocks, metal_blocks
            )
            _, compact_sidechains, _ = movable_block_mapping(
                reference_pose_stack, ligand_block, sidechain_blocks, metal_blocks
            )
            movable_dofs = build_ligand_docking_dofs(
                movable_pose, 0, sidechain_blocks=compact_sidechains
            )
            _, sidechain_atom_indices = _sidechain_coordinate_indices(
                movable_pose,
                movable_dofs,
                PoseStackKinematicsModule(
                    movable_pose, movable_dofs.fold_forest
                ).kmd.forest,
            )
            grid_atom_indices = torch.cat(
                (
                    torch.arange(block_type.n_atoms, dtype=torch.int64),
                    sidechain_atom_indices[0].to(device="cpu"),
                )
            ).contiguous()
        query = (
            torch.arange(block_type.n_atoms)
            if grid_atom_indices is None
            else grid_atom_indices
        )
        source_types = [
            int(reference_pose_stack.block_type_ind64[0, block])
            for block in moving.source_blocks
        ]
        atom_types = [
            atom.atom_type
            for index in source_types
            for atom in pbt.active_block_types[index].atoms[
                : pbt.active_block_types[index].n_atoms
            ]
        ]
        ligand_atom_types = tuple(atom_types[i] for i in query.tolist())

        if grids is None:
            grids = cls.pocket_grids(
                reference_pose_stack,
                ligand_block,
                param_db,
                # A search that can deprotonate the ligand retypes a donor, and
                # a grid without that type cannot score the bound form.
                atom_types=tuple(ligand_atom_types) + tuple(extra_atom_types),
                flexible_residues=sidechain_blocks,
                extra_excluded_blocks=metal_blocks,
                device=grid_device,
                voxel_spacing=voxel_spacing,
                bbox_padding=bbox_padding,
                lk_ball_grid=lk_ball_grid,
                score_function=score_function,
            )
        else:
            # Reused grids may have been built with a different palette.
            _setup_scoring_parameters(reference_pose_stack, param_db)
        missing = set(ligand_atom_types) - set(grids.atom_type_names)
        if missing:
            raise ValueError(f"grids lack the ligand atom types {sorted(missing)}")
        ligand_charges = torch.cat(
            [
                pbt.elec_partial_charge[index, : pbt.active_block_types[index].n_atoms]
                for index in source_types
            ]
        )[query]
        # Every bound form is already in the palette, so each one's sites can
        # be built now rather than when detection first reaches it.
        ligand_atom_maps = (
            ligand_variant_atom_maps(reference_pose_stack, ligand_block)
            if complete_block_types is not None
            else None
        )
        live_hbond = build_live_hbond_data(
            reference_pose_stack,
            ligand_block,
            param_db,
            sidechain_blocks=sidechain_blocks,
            movable_atoms=grid_atom_indices,
            extra_blocks=metal_blocks,
            ligand_atom_maps=ligand_atom_maps,
        )
        live_ligand_intra = (
            build_live_ligand_intra_data(
                reference_pose_stack,
                ligand_block,
                param_db,
                weight_profile=ligand_intra_weight_profile,
                sidechain_blocks=sidechain_blocks,
                extra_blocks=metal_blocks,
            )
            if include_ligand_intra
            else None
        )
        live_lk_ball = build_live_lk_ball_data(
            reference_pose_stack,
            ligand_block,
            param_db,
            sidechain_blocks=sidechain_blocks,
            movable_atoms=grid_atom_indices,
            extra_blocks=metal_blocks,
            ligand_atom_maps=ligand_atom_maps,
        )
        return cls(
            grids,
            ligand_block,
            ligand_atom_types,
            ligand_charges,
            param_db,
            kin_dtype=kin_dtype,
            live_hbond=live_hbond,
            live_ligand_intra=live_ligand_intra,
            live_lk_ball=live_lk_ball,
            lk_ball_grid=lk_ball_grid,
            score_function=score_function,
            sidechain_blocks=sidechain_blocks,
            context_blocks=context_blocks,
            metal_blocks=metal_blocks,
            complete_block_types=complete_block_types,
            grid_atom_indices=grid_atom_indices,
            movable_n_atoms=moving.n_atoms,
            cuda_graph=cuda_graph,
        )

    def _scorer(self, device: torch.device, dtype: torch.dtype) -> GridScorer:
        """The grid scorer for one device and precision, built on first use."""
        key = (torch.device(device), dtype)
        if key not in self._scorers:
            scorer = GridScorer(
                self.grids,
                self.ligand_atom_types,
                self.ligand_charges,
                device=device,
                dtype=dtype,
                score_function=self.score_function
                or beta2016_score_function(torch.device(device)),
                live_hbond=self.live_hbond,
                live_lk_ball=self.live_lk_ball,
                live_ligand_intra=self.live_ligand_intra,
                lk_ball_grid=self.lk_ball_grid,
                grid_atoms=self.grid_atom_indices,
            )
            scorer.set_rep_scale(self._rep_scale)
            self._scorers[key] = scorer
        scorer = self._scorers[key]
        scorer.set_pose_atom_state(*(self._pose_atom_state or (None,)))
        scorer.set_intra_topology(*(self._pose_intra_state or (None,)))
        return scorer

    def set_ligand_variants(
        self, variants: "LigandVariantDOFs", pose_stack: PoseStack
    ) -> None:
        """Register the block types a search can give this ligand.

        Each one's atom types and charges are read from the packed block types
        and laid out on the union's slots, so a pose that deprotonated to bind
        a metal is scored as the molecule it became.
        """
        # Detection rebuilds the stack, so its block types may not carry the
        # elec annotation these charges come from yet.
        _setup_scoring_parameters(pose_stack, self.param_db)
        self.ligand_variant_dofs = variants
        pbt = pose_stack.packed_block_types
        to_grid = self.grids.atom_type_to_grid
        n_ligand = len(variants.atom_names)
        if n_ligand > len(self.ligand_atom_types):
            raise ValueError("ligand variants are wider than the grid-scored atoms")
        n_other = len(self.ligand_atom_types) - n_ligand
        # Cleared in place, not rebound: ``for_ligand_block`` views share this
        # dict, and finalization reads it through the outer backend.
        self.ligand_variants.clear()
        for index in sorted(variants.dofs):
            block_type = pbt.active_block_types[index]
            local = variants.atom_map[index].to("cpu")
            types = list(self.ligand_atom_types)
            charges = self.ligand_charges.clone()
            for slot in range(n_ligand):
                atom = int(local[slot])
                if atom < 0:
                    continue
                types[slot] = block_type.atoms[atom].atom_type
                charges[slot] = pbt.elec_partial_charge[index, atom]
            missing = sorted(set(types) - set(to_grid))
            if missing:
                raise ValueError(f"grids lack the ligand atom types {missing}")
            present = torch.cat(
                (
                    (local >= 0),
                    torch.ones(n_other, dtype=torch.bool),
                )
            )
            # Read on the host above, held on the population's device below.
            device = pose_stack.device
            self.ligand_variants[index] = LigandVariantAtoms(
                local_atom=local.to(device),
                grid_index=torch.tensor(
                    [to_grid[t] for t in types], dtype=torch.int64, device=device
                ),
                charges=charges.to(device),
                grid_present=present.to(device),
            )

    def set_metal_term(self, pose_stack: PoseStack, weight: float) -> None:
        """Point the metal coordination term at this population.

        The module reads the ligand's block type and bonds, which detection
        rewrites every generation -- but it reads them from tensors, so they are
        written in place and the module (and any graph capturing it) is kept.
        It is rendered once per population shape, on a stack this backend owns.
        ``weight`` of zero, or a pose with no metal site, drops the term.
        """
        from tmol.score import ScoreFunction

        pbt = pose_stack.packed_block_types
        has_metal = any(
            pbt.active_block_types[int(index)].metal_sites
            for index in pose_stack.block_type_ind64[0].tolist()
            if index >= 0
        )
        if weight == 0.0 or not has_metal:
            if self._metal_module is not None:
                self._metal_render += 1
            self._metal_module = None
            self._metal_stack = None
            self._metal_n_poses = 0
            return

        shape = (pose_stack.n_poses, pose_stack.max_n_pose_atoms, pose_stack.device)
        if self._metal_stack is None or self._metal_shape != shape:
            score_function = ScoreFunction(self.param_db, pose_stack.device)
            score_function.set_weight(ScoreType.metal_coordination, weight)
            # Every block of the compacted pose may move except the metals, and
            # the term is a handful of sites, so it is not worth a mask whose
            # indices would have to be translated into compact numbering.
            owned = pose_stack.clone()
            self._metal_module = score_function.render_whole_pose_scoring_module(owned)
            self._metal_stack = owned
            self._metal_shape = shape
            self._metal_render += 1
        else:
            # Same population shape: only the ligand's type and bonds moved.
            owned = self._metal_stack
            owned.block_type_ind64.copy_(pose_stack.block_type_ind64)
            owned.block_type_ind.copy_(pose_stack.block_type_ind)
            owned.inter_residue_connections64.copy_(
                pose_stack.inter_residue_connections64
            )
            owned.inter_residue_connections.copy_(pose_stack.inter_residue_connections)
        self._metal_n_poses = pose_stack.n_poses
        self._metal_max_atoms = pose_stack.max_n_pose_atoms
        index, present = self._packed_from_union(pose_stack)
        # Held in stable buffers for the same reason as the per-pose atom state:
        # a captured graph reads them by address.
        if self._metal_index is None or self._metal_index.shape != index.shape:
            self._metal_index, self._metal_present = index, present
        else:
            self._metal_index.copy_(index)
            self._metal_present.copy_(present)

    def _packed_from_union(
        self, pose_stack: PoseStack
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where each movable-vector slot lives in the pose's packed coords.

        The optimizer works on fixed-width union slots with absent ones parked;
        the scoring module wants the pose's own tight layout, which differs for
        every pose whose ligand dropped a hydrogen.
        """
        ligand_map = self.ligand_local_atom_map(pose_stack)
        indices, present = [], []
        for block in range(pose_stack.max_n_blocks):
            block_type_index = int(pose_stack.block_type_ind64[0, block])
            if block_type_index < 0:
                continue
            if block == self.ligand_block and ligand_map is not None:
                local = ligand_map
            else:
                n_atoms = pose_stack.block_type(0, block).n_atoms
                local = torch.arange(
                    n_atoms, dtype=torch.int64, device=pose_stack.device
                ).expand(pose_stack.n_poses, -1)
            indices.append(
                pose_stack.block_coord_offset64[:, block, None] + local.clamp(min=0)
            )
            present.append(local >= 0)
        return torch.cat(indices, dim=1), torch.cat(present, dim=1)

    def _metal_energy_and_gradient(
        self, coords: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pose metal coordination energy and its coordinate gradient.

        The line search stacks several trial copies of the whole pool, so a
        batch is scored one pool at a time against the module rendered for it.
        """
        n_poses = self._metal_n_poses
        if coords.shape[0] != n_poses:
            if coords.shape[0] % n_poses:
                raise ValueError(
                    f"metal term rendered for {n_poses} poses cannot score "
                    f"{coords.shape[0]}"
                )
            scored = [
                self._metal_chunk(chunk) for chunk in coords.split(n_poses, dim=0)
            ]
            return (
                torch.cat([e for e, _ in scored]),
                torch.cat([g for _, g in scored]),
            )
        return self._metal_chunk(coords)

    def _metal_chunk(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One pool's metal coordination energy and coordinate gradient."""
        index, present = self._metal_index, self._metal_present
        rows = torch.arange(coords.shape[0], dtype=torch.int64, device=coords.device)[
            :, None
        ].expand_as(index)
        # Boolean-mask indexing would size its output on the device and so
        # cannot be captured in a CUDA graph. An absent slot instead scatters a
        # zero onto column 0, which accumulate leaves alone.
        safe_index = torch.where(present, index, index.new_zeros(()))
        # The callers score under no_grad; this term needs its own graph.
        with torch.enable_grad():
            tracked = coords.detach().clone().requires_grad_(True)
            packed = tracked.new_zeros((coords.shape[0], self._metal_max_atoms, 3))
            packed = packed.index_put(
                (rows, safe_index),
                torch.where(present[..., None], tracked, tracked.new_zeros(())),
                accumulate=True,
            )
            energies = self._metal_module(packed)
            if energies.dim() > 1:
                energies = energies.sum(dim=tuple(range(energies.dim() - 1)))
            # A generation whose poses declared no ligand-metal bond leaves the
            # movable coordinates out of the term's graph entirely.
            (gradient,) = torch.autograd.grad(
                energies.sum(), tracked, allow_unused=True
            )
        if gradient is None:
            gradient = torch.zeros_like(coords)
        return energies.detach(), gradient

    def apply_pose_variants(self, pose_stack: PoseStack) -> None:
        """Give the scorer this population's per-pose atom types and charges."""
        if not self.ligand_variants:
            self._pose_atom_state = None
            self._pose_intra_state = None
            return
        block_types = pose_stack.block_type_ind64[:, self.ligand_block].tolist()
        unknown = sorted(set(block_types) - set(self.ligand_variants))
        if unknown:
            raise ValueError(f"ligand block types {unknown} have no variant atoms")
        rows = [self.ligand_variants[int(b)] for b in block_types]
        # Scorers copy from these inside a captured evaluation, so a replay
        # reads them by address: they are rewritten, never rebound.
        self._pose_atom_state = _write_into(
            self._pose_atom_state,
            (
                torch.stack([r.grid_index for r in rows]),
                torch.stack([r.charges for r in rows]),
                torch.stack([r.grid_present for r in rows]),
            ),
        )
        self._pose_intra_state = _write_into(
            self._pose_intra_state, self._intra_topology(pose_stack)
        )
        # Each scorer holds its own device/dtype copy of the live data, so
        # every copy is pointed at this population's forms.
        if self.live_hbond is not None:
            set_hbond_pose_variants(self.live_hbond, block_types)
        if self.live_lk_ball is not None:
            set_lk_ball_pose_variants(self.live_lk_ball, block_types)
        for scorer in self._scorers.values():
            # Pushed here, not only on fetch: a replayed graph fetches nothing.
            scorer.set_pose_atom_state(*self._pose_atom_state)
            scorer.set_intra_topology(*(self._pose_intra_state or (None,)))
            if scorer.live_hbond is not None:
                set_hbond_pose_variants(scorer.live_hbond, block_types)
            if scorer.live_lk_ball is not None:
                set_lk_ball_pose_variants(
                    scorer.live_lk_ball, block_types, scorer._grid_slot_of
                )

    def _intra_topology(self, pose_stack: PoseStack):
        """Per-pose ligand block types, bonds and movable-slot order.

        The intra modules are rendered on the widest ligand form, so every
        block keeps a fixed region of the movable vector; only the ligand's own
        atom numbering moves, and a slot its block type lacks is absent.
        """
        local = self.ligand_local_atom_map(pose_stack)
        if local is None:
            return None
        width = local.shape[1]
        n_poses = pose_stack.n_poses
        index = (
            torch.arange(self.movable_n_atoms, device=pose_stack.device)
            .expand(n_poses, -1)
            .clone()
        )
        present = torch.ones(
            (n_poses, self.movable_n_atoms), dtype=torch.bool, device=pose_stack.device
        )
        index[:, :width] = local.clamp(min=0)
        present[:, :width] = local >= 0
        return (
            pose_stack.block_type_ind64.clone(),
            pose_stack.inter_residue_connections64.clone(),
            index,
            present,
        )

    @contextmanager
    def rep_scale(self, scale: float):
        """Scale fa_rep for the duration of the block, as Rosetta packing does.

        Rosetta packs each ramp step at a softened repulsive weight and then
        minimizes at the full one (``GridScorer::optimize``).
        """
        previous = self._rep_scale
        self._rep_scale = float(scale)
        for scorer in self._scorers.values():
            scorer.set_rep_scale(self._rep_scale)
        try:
            yield
        finally:
            self._rep_scale = previous
            for scorer in self._scorers.values():
                scorer.set_rep_scale(previous)

    def ligand_local_atom_map(self, pose_stack: PoseStack) -> Optional[torch.Tensor]:
        """Per-pose ``[pose, slot]`` ligand-block-local atom index, -1 absent.

        ``None`` when the backend holds one ligand block type, which keeps the
        homogeneous path on its own plain ranges.
        """
        if not self.ligand_variants:
            return None
        block_types = pose_stack.block_type_ind64[:, self.ligand_block]
        unknown = sorted(set(block_types.tolist()) - set(self.ligand_variants))
        if unknown:
            raise ValueError(f"ligand block types {unknown} have no variant atoms")
        return torch.stack(
            [self.ligand_variants[int(b)].local_atom for b in block_types], dim=0
        )

    def _ligand_atom_indices(
        self, pose_stack: PoseStack, ligand_block: int
    ) -> torch.Tensor:
        if ligand_block != self.ligand_block:
            raise ValueError(
                f"backend was built for ligand block {self.ligand_block}, "
                f"not {ligand_block}"
            )
        cache_key = (
            torch.device(pose_stack.device),
            pose_stack.n_poses,
            pose_stack.max_n_pose_atoms,
            pose_stack.max_n_blocks,
            ligand_block,
            id(pose_stack.packed_block_types),
        )
        layout = (
            pose_stack.packed_block_types,
            pose_stack.block_coord_offset64,
            pose_stack.block_type_ind64,
        )
        # Retain and check the actual layout owners, including in-place edits.
        # Inference tensors have no mutation counter and are not cached.
        versions = (
            None
            if any(t.is_inference() for t in layout[1:])
            else tuple(t._version for t in layout[1:])
        )
        cached = self._atom_index_cache.get(cache_key)
        if (
            versions is not None
            and cached is not None
            and all(a is b for a, b in zip(layout, cached[0]))
            and versions == cached[1]
        ):
            return cached[2]

        block_types = pose_stack.block_type_ind64[:, ligand_block]
        if bool(torch.any(block_types < 0)):
            raise ValueError("ligand block is absent from at least one pose")
        ligand_map = self.ligand_local_atom_map(pose_stack)
        if ligand_map is not None:
            # Poses may hold different metal variants; index each on its own
            # slots and leave absent ones clamped for the caller to mask.
            atom_indices = pose_stack.block_coord_offset64[
                :, ligand_block, None
            ] + ligand_map.clamp(min=0)
            if versions is not None:
                self._atom_index_cache[cache_key] = (layout, versions, atom_indices)
            return atom_indices
        first_type = block_types[0]
        if not bool(torch.all(block_types == first_type)):
            raise ValueError("GridBackend requires one homogeneous ligand type")
        block_type = pose_stack.packed_block_types.active_block_types[int(first_type)]
        if not self.sidechain_blocks and block_type.n_atoms != len(
            self.ligand_atom_types
        ):
            raise ValueError("pose ligand atom count does not match the grid backend")
        offsets = pose_stack.block_coord_offset64[:, ligand_block, None]
        local_atoms = torch.arange(
            block_type.n_atoms, dtype=torch.int64, device=pose_stack.device
        )
        atom_indices = offsets + local_atoms[None, :]
        if versions is not None:
            self._atom_index_cache[cache_key] = (layout, versions, atom_indices)
        return atom_indices

    def _movable_atom_indices(
        self, pose_stack: PoseStack, ligand_block: int
    ) -> torch.Tensor:
        """Return compact whole-movable coordinate indices for every pose."""

        if not self.sidechain_blocks and not self.metal_blocks:
            return self._ligand_atom_indices(pose_stack, ligand_block)
        expected_blocks = 1 + len(self.sidechain_blocks) + len(self.context_blocks)
        if ligand_block != 0 or pose_stack.max_n_blocks != expected_blocks:
            raise ValueError(
                "flexible GridBackend requires a compact ligand-plus-sidechains pose"
            )
        if pose_stack.max_n_pose_atoms != self.movable_n_atoms:
            raise ValueError(
                "movable pose atom count does not match the grid backend topology"
            )
        return torch.arange(
            self.movable_n_atoms,
            dtype=torch.int64,
            device=pose_stack.device,
        )[None, :].expand(pose_stack.n_poses, -1)

    def _score_ligand_coordinates(
        self, ligand_coordinates: torch.Tensor
    ) -> torch.Tensor:
        scorer = self._scorer(ligand_coordinates.device, ligand_coordinates.dtype)
        energies = scorer(ligand_coordinates)
        if self._metal_module is not None:
            energies = energies + self._metal_energy_and_gradient(ligand_coordinates)[0]
        return energies

    def _score_ligand_coordinates_with_gradients(
        self, ligand_coordinates: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scorer = self._scorer(ligand_coordinates.device, ligand_coordinates.dtype)
        energies, gradients = scorer.value_and_grad(ligand_coordinates)
        if self._metal_module is not None:
            metal, metal_gradient = self._metal_energy_and_gradient(ligand_coordinates)
            energies = energies + metal.to(energies.dtype)
            gradients = gradients + metal_gradient.to(gradients.dtype)
        return energies, gradients

    def score(self, pose_stack: PoseStack) -> torch.Tensor:
        """Return one grid GA objective energy per pose.

        This is the grid ligand-receptor terms (grid LJ/LK/elec + live hbond +
        live LK-ball) plus, by default, the live ligand-intramolecular energy --
        NOT a pure interface energy. Disable the intra contribution by building
        the backend with ``include_ligand_intra=False``.
        """

        pose_indices = torch.arange(
            pose_stack.n_poses, dtype=torch.int64, device=pose_stack.device
        )
        if self.ligand_variants:
            # Poses holding different variants pack their atoms differently,
            # so gather each onto the shared slots; an absent slot is marked
            # absent for every term rather than given a placeholder.
            atom_indices, _ = self._packed_from_union(pose_stack)
            ligand_coordinates = pose_stack.coords[pose_indices[:, None], atom_indices]
        else:
            atom_indices = self._movable_atom_indices(pose_stack, self.ligand_block)
            ligand_coordinates = pose_stack.coords[pose_indices[:, None], atom_indices]
        with torch.no_grad():
            energies = self._score_ligand_coordinates(ligand_coordinates)
        expected_shape = (pose_stack.n_poses,)
        if energies.shape != expected_shape:
            raise RuntimeError(
                f"grid scorer returned {tuple(energies.shape)}; "
                f"expected {expected_shape}"
            )
        return energies

    def minimize(
        self,
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        *,
        maxiter: Optional[int] = None,
        initial_raw_dofs: Optional[torch.Tensor] = None,
    ) -> MinimizationResult:
        """Minimize with direct analytic grid and kinematic derivatives.

        The optimizer holds only freed docking DOFs and a bounded L-BFGS
        history. Each closure uses the valid full-pose KinForest ordering,
        short-lived compiled kinematic/grid-gradient buffers, and explicitly
        assigned gradients. Ligand-intra kernels use a local autograd graph
        internally and return detached values; no graph reaches the optimizer.
        Selected receptor-sidechain DOFs receive grid, live-interface, and
        whole-movable gradients; all other receptor DOFs remain fixed.
        """

        if docking_dofs.ligand_block != self.ligand_block:
            raise ValueError("docking DOFs target a different ligand block")
        kin_dtype = self.kin_dtype or pose_stack.coords.dtype
        objective = _LightweightGridObjective(
            self,
            pose_stack,
            docking_dofs,
            kin_dtype,
            initial_raw_dofs,
            variants=self.ligand_variant_dofs,
        )
        minimizer = self._minimizer_for(pose_stack, docking_dofs, kin_dtype, objective)
        x0 = objective.initial_masked_dofs.reshape(pose_stack.n_poses, -1)
        max_iterations = 200 if maxiter is None else maxiter
        x, energies = minimizer.minimize(objective.full_dofs, x0, max_iterations)
        result = objective.final_result(x.reshape(-1))
        return replace(result, energies=energies.to(pose_stack.coords.dtype))

    def _captured_table_addresses(self) -> tuple:
        """Addresses of the per-pose tables a captured evaluation reads.

        They are rewritten in place while their shape holds; a reallocation
        (shared rows becoming per-pose rows, say) must not reuse a graph.
        """
        tables = [self._metal_index, self._metal_present]
        for scorer in self._scorers.values():
            tables += [
                scorer._type_table,
                scorer._charge_table,
                scorer._polar_type_table,
                *(scorer._lk_tables or ()),
                *(scorer._intra_packing or ()),
            ]
            for data, names in (
                (scorer.live_hbond, _HBOND_POSE_TABLES),
                (scorer.live_lk_ball, _LK_BALL_POSE_TABLES),
            ):
                if data is not None:
                    tables += [getattr(data, name) for name in names]
        return tuple(None if t is None else t.data_ptr() for t in tables)

    def _minimizer_for(
        self,
        pose_stack: PoseStack,
        docking_dofs: LigandDockingDOFs,
        kin_dtype: torch.dtype,
        objective: _LightweightGridObjective,
    ) -> "_PopulationMinimizer":
        """The (cached) candidate-batched minimizer of this population layout.

        A captured graph reads kinematics and per-pose scoring tables by address.
        The tables are rewritten in place each generation, so the graph stays
        valid as long as the kinematic layout it was captured on is unchanged;
        metal variants that differ only in their bonds leave it unchanged.
        """
        key = (
            docking_dofs.ligand_block,
            docking_dofs.sidechain_blocks,
            pose_stack.n_poses,
            kin_dtype,
            id(pose_stack.packed_block_types),
            # A rendered metal module's tensors are captured by address, so a
            # graph cannot be reused across a re-render.
            self._metal_render,
        )
        key += self._captured_table_addresses()
        layout = _kinematic_layout(objective)
        cached = self._minimizers.get(key)
        if cached is not None and _same_tensors(cached[0], layout):
            return cached[1]
        self._minimizers.clear()
        minimizer = _PopulationMinimizer(self, pose_stack, docking_dofs, kin_dtype)
        self._minimizers[key] = (layout, minimizer)
        return minimizer
