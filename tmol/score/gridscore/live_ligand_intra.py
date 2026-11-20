"""Live ligand-intramolecular energy for rigid-receptor grid scoring."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from tmol.database import ParameterDatabase
from tmol.pose._packed_block_types import PackedBlockTypes
from tmol.pose._pose_stack import PoseStack
from tmol.pose import repeat_pose_stack, select_blocks
from tmol.score import beta2016_score_function
from tmol.score import ScoreFunction
from tmol.score._score_types import ScoreType

from ._common import validate_ligand_coords
from ._movable import movable_block_mapping, movable_pose_stack

LigandIntraWeightProfile = Literal["search", "full"]

_LJLK_ELEC_SCORE_TYPES = (
    ScoreType.fa_ljatr,
    ScoreType.fa_ljrep,
    ScoreType.fa_lk,
    ScoreType.fa_elec,
)
_CART_SCORE_TYPES = (
    ScoreType.cart_lengths,
    ScoreType.cart_angles,
    ScoreType.cart_torsions,
    ScoreType.cart_impropers,
    ScoreType.cart_hxltorsions,
)


def _intra_score_function(
    param_db: ParameterDatabase,
    device: torch.device,
    weight_profile: LigandIntraWeightProfile,
    include_cart_bonded: bool,
    whole_movable: bool = False,
    score_function: ScoreFunction | None = None,
) -> ScoreFunction:
    """Build the tmol equivalent of Rosetta GridScorer's ``sf1b_``.

    Rosetta's ligand ``fa_intra_atr_xover4``, ``fa_intra_rep_xover4``,
    ``fa_intra_sol_xover4``, and ``fa_intra_elec`` map to tmol's ordinary
    ``fa_ljatr``, ``fa_ljrep``, ``fa_lk``, and ``fa_elec`` score types. Their
    intra-block branches apply the ligand CP_CROSSOVER_3FULL count-pair rule.
    Rosetta ``gen_bonded`` maps to ``gen_torsions``; ligand ``cart_bonded`` is
    represented by tmol's five ``cart_*`` channels for cyclic ligands.

    Rosetta ``GridScorer.cc:338-343`` defines the full one-body terms. Its
    search setup at lines 297-313 replaces intra-repulsion with 0.01 and both
    generic/cartesian bonded weights with 0.1 (the ligand cartesian assignment
    is also visible at line 370). The other weights come from the supplied
    score function, which defaults to beta2016.
    """

    if weight_profile not in ("search", "full"):
        raise ValueError("weight_profile must be 'search' or 'full'")

    source = score_function or beta2016_score_function(device, param_db=param_db)
    result = ScoreFunction(param_db, device)
    result.set_options(dict(source.term_options))
    if whole_movable:
        score_types = tuple(source.all_score_types())
    else:
        score_types = _LJLK_ELEC_SCORE_TYPES + (ScoreType.gen_torsions,)
        if include_cart_bonded:
            score_types += _CART_SCORE_TYPES
    weights = dict(zip(source.all_score_types(), source.weights_tensor().tolist()))
    for score_type in score_types:
        weight = weights.get(score_type, 0.0)
        if weight_profile == "search" and not whole_movable:
            weight = _SEARCH_DIAGONAL_WEIGHTS.get(score_type, weight)
        result.set_weight(score_type, weight)
    return result


def _ligand_pose_to(
    pose_stack: PoseStack,
    device: torch.device,
    dtype: torch.dtype,
) -> PoseStack:
    """Move a compact movable PoseStack and rebuild its device-local PBT."""

    # Resolve to a concrete device (e.g. cuda -> cuda:0) so the PoseStack's
    # declared device matches the device its tensors actually land on; an
    # indexless "cuda" would otherwise mismatch the tensors' "cuda:0".
    device = torch.empty(0, device=device).device

    source_pbt = pose_stack.packed_block_types
    source_type_indices = []
    for value in pose_stack.block_type_ind64[0].tolist():
        source_type_index = int(value)
        if source_type_index >= 0 and source_type_index not in source_type_indices:
            source_type_indices.append(source_type_index)
    block_types = [
        source_pbt.active_block_types[source_type_index]
        for source_type_index in source_type_indices
    ]
    pbt = PackedBlockTypes.from_restype_list(
        source_pbt.chem_db,
        source_pbt.restype_set,
        block_types,
        device=device,
    )

    def moved(value: torch.Tensor) -> torch.Tensor:
        return value.to(device=device).contiguous()

    def moved_connections(value: torch.Tensor) -> torch.Tensor:
        if value.shape[2] > pbt.max_n_conn:
            # Compacting the PBT may remove padding, but never topology.
            if bool((value[:, :, pbt.max_n_conn :, :] != -1).any()):
                raise ValueError(
                    "_ligand_pose_to cannot truncate inter_residue_connections "
                    f"to max_n_conn={pbt.max_n_conn}: dropped slots must all be -1"
                )
            value = value[:, :, : pbt.max_n_conn, :]
        return moved(value)

    type_mapping = {
        source_type_index: target_type_index
        for target_type_index, source_type_index in enumerate(source_type_indices)
    }
    source_block_type_ind64 = pose_stack.block_type_ind64.to(device=device)
    block_type_ind64 = torch.full_like(source_block_type_ind64, -1)
    for source_type_index, target_type_index in type_mapping.items():
        block_type_ind64[source_block_type_ind64 == source_type_index] = (
            target_type_index
        )
    return PoseStack(
        packed_block_types=pbt,
        coords=pose_stack.coords.to(device=device, dtype=dtype).contiguous(),
        block_coord_offset=moved(pose_stack.block_coord_offset),
        block_coord_offset64=moved(pose_stack.block_coord_offset64),
        inter_residue_connections=moved_connections(
            pose_stack.inter_residue_connections
        ),
        inter_residue_connections64=moved_connections(
            pose_stack.inter_residue_connections64
        ),
        inter_block_bondsep=pose_stack.inter_block_bondsep.to(device),
        block_type_ind=block_type_ind64.to(torch.int32),
        block_type_ind64=block_type_ind64,
        chain_id=moved(pose_stack.chain_id),
        chain_id64=moved(pose_stack.chain_id64),
        pdb_info=pose_stack.pdb_info,
        constraint_set=None,
        device=device,
    )


@dataclass(frozen=True)
class LiveLigandIntraData:
    """A compact movable/context topology and compiled-scorer cache."""

    ligand_pose_stack: PoseStack
    score_function: ScoreFunction
    param_db: ParameterDatabase
    weight_profile: LigandIntraWeightProfile
    include_cart_bonded: bool
    context_block_indices: tuple[int, ...] = ()
    _context_indices: torch.Tensor = field(init=False, compare=False, repr=False)
    _rep_scale: torch.Tensor = field(init=False, compare=False, repr=False)
    _scorer_cache: dict[object, object] = field(
        default_factory=dict, compare=False, repr=False
    )
    _weight_cache: dict[tuple, tuple] = field(
        default_factory=dict, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        pose_stack = self.ligand_pose_stack
        if pose_stack.n_poses != 1 or pose_stack.max_n_blocks < 1:
            raise ValueError("live ligand-intra data requires one movable pose")
        if pose_stack.max_n_pose_atoms < 1:
            raise ValueError("live ligand-intra data requires ligand atoms")
        if pose_stack.device != pose_stack.coords.device:
            raise ValueError("live ligand-intra PoseStack device is inconsistent")
        if pose_stack.packed_block_types.device != pose_stack.device:
            raise ValueError(
                "live ligand-intra packed block types are on another device"
            )
        if self.weight_profile not in ("search", "full"):
            raise ValueError("weight_profile must be 'search' or 'full'")
        if any(
            block < 0 or block >= pose_stack.max_n_blocks
            for block in self.context_block_indices
        ):
            raise ValueError("context block index is out of range")
        # Prepare host topology before scoring, including CUDA graph capture.
        object.__setattr__(
            self,
            "_context_indices",
            torch.tensor(
                self.context_block_indices, dtype=torch.int64, device=pose_stack.device
            ),
        )
        object.__setattr__(
            self,
            "_rep_scale",
            torch.ones((), dtype=torch.float64, device=pose_stack.device),
        )

    @property
    def n_ligand_atoms(self) -> int:
        return self.ligand_pose_stack.max_n_pose_atoms

    def set_rep_scale(self, scale: float) -> None:
        """Scale the repulsive weight of every movable interaction."""
        self._rep_scale.fill_(float(scale))

    def to(
        self,
        device: torch.device | str,
        dtype: torch.dtype,
        *,
        score_function: ScoreFunction | None = None,
    ) -> "LiveLigandIntraData":
        """Return ligand topology and scoring machinery on a device/dtype."""
        device = torch.device(device)
        ligand_pose_stack = _ligand_pose_to(self.ligand_pose_stack, device, dtype)
        score_function = _intra_score_function(
            self.param_db,
            device,
            self.weight_profile,
            self.include_cart_bonded,
            whole_movable=ligand_pose_stack.max_n_blocks > 1,
            score_function=score_function or self.score_function,
        )
        return LiveLigandIntraData(
            ligand_pose_stack=ligand_pose_stack,
            score_function=score_function,
            param_db=self.param_db,
            weight_profile=self.weight_profile,
            include_cart_bonded=self.include_cart_bonded,
            context_block_indices=self.context_block_indices,
        )


def build_live_ligand_intra_data(
    reference_pose_stack: PoseStack,
    ligand_block: int,
    param_db: ParameterDatabase,
    *,
    weight_profile: LigandIntraWeightProfile,
    sidechain_blocks: tuple[int, ...] = (),
    score_function: ScoreFunction | None = None,
) -> LiveLigandIntraData:
    """Build CPU live scoring data for movable blocks and peptide context.

    Context-only block diagonals are omitted because a context residue's own
    backbone-dependent one-body terms can require its next, intentionally
    omitted peptide neighbor. All off-diagonal context interactions remain.
    """

    if reference_pose_stack.n_poses != 1:
        raise ValueError("live ligand-intra construction requires one reference pose")

    ligand_pose_stack = movable_pose_stack(
        reference_pose_stack, ligand_block, sidechain_blocks
    )
    context_block_indices: tuple[int, ...] = ()
    if sidechain_blocks:
        _, _, context_block_indices = movable_block_mapping(
            reference_pose_stack, ligand_block, sidechain_blocks
        )
    ligand_pose_stack = _ligand_pose_to(
        ligand_pose_stack,
        reference_pose_stack.device,
        reference_pose_stack.coords.dtype,
    )
    include_cart_bonded = any(
        any(ligand_pose_stack.block_type(0, block).bond_to_ringness.values())
        for block in range(ligand_pose_stack.max_n_blocks)
    )
    score_function = _intra_score_function(
        param_db,
        reference_pose_stack.device,
        weight_profile,
        include_cart_bonded,
        whole_movable=ligand_pose_stack.max_n_blocks > 1,
        score_function=score_function,
    )
    return LiveLigandIntraData(
        ligand_pose_stack=ligand_pose_stack,
        score_function=score_function,
        param_db=param_db,
        weight_profile=weight_profile,
        include_cart_bonded=include_cart_bonded,
        context_block_indices=context_block_indices,
    )


def _validate_ligand_coords(
    data: LiveLigandIntraData, ligand_coords: torch.Tensor
) -> None:
    validate_ligand_coords(
        ligand_coords, data.n_ligand_atoms, data.ligand_pose_stack.coords, "intra"
    )


def _movable_block_mask(
    data: LiveLigandIntraData, pose_stack: PoseStack
) -> torch.Tensor | None:
    """Blocks whose coordinates move; context blocks are frozen."""
    if not data.context_block_indices:
        return None
    mask = torch.ones(
        (pose_stack.n_poses, pose_stack.max_n_blocks),
        dtype=torch.bool,
        device=pose_stack.device,
    )
    mask[:, data._context_indices] = False
    return mask


def _scorer_for(data: LiveLigandIntraData, n_poses: int):
    cached = data._scorer_cache.get(n_poses)
    if cached is None:
        pose_stack = repeat_pose_stack(data.ligand_pose_stack, n_poses)
        if data.ligand_pose_stack.max_n_blocks > 1:
            scorer = data.score_function.render_block_pair_scoring_module(
                pose_stack, movable_blocks=_movable_block_mask(data, pose_stack)
            )
        else:
            scorer = data.score_function.render_whole_pose_scoring_module(pose_stack)
        cached = (pose_stack, scorer)
        data._scorer_cache[n_poses] = cached
    return cached[1]


_SEARCH_DIAGONAL_WEIGHTS = {
    ScoreType.fa_ljrep: 0.01,
    ScoreType.gen_torsions: 0.1,
    **{score_type: 0.1 for score_type in _CART_SCORE_TYPES},
}


class _WholeMovableScorer:
    """The movable block-pair energy, evaluated as whole-pose scores.

    Equal to :func:`_movable_block_pair_energy` while the context blocks and
    the movable residues' backbones hold their reference coordinates, as they
    do throughout docking, but without block-pair autograd: the whole-pose
    scorer accumulates coordinate derivatives in its forward pass. The
    energy splits as

        masked whole-pose beta2016 (fa_ljrep scaled by the repulsive ramp)
        + sum over softened terms of (diagonal - full weight) times the
          energy of each movable block scored alone
        - a constant: the frozen context's own energy, which the block-pair
          form omits.

    A block scored alone has exactly its block-pair diagonal for the
    softened terms (pairwise and bonded terms within one block); the
    backbone-dependent one-body terms that differ are not softened.
    """

    def __init__(self, data: LiveLigandIntraData, n_poses: int) -> None:
        reference = data.ligand_pose_stack
        device, dtype = reference.device, reference.coords.dtype
        self._data = data
        # A rendered scorer's weights alias its score function's, and the
        # repulsive ramp rewrites them in place; data.score_function may be
        # the memoized beta2016 every other scorer shares, so score with a
        # private copy.
        sfxn = ScoreFunction(data.param_db, device)
        sfxn.set_options(dict(data.score_function.term_options))
        for score_type, weight in zip(
            data.score_function.all_score_types(),
            data.score_function.weights_tensor().tolist(),
        ):
            sfxn.set_weight(score_type, weight)
        pose_stack = repeat_pose_stack(reference, n_poses)
        self._whole = sfxn.render_whole_pose_scoring_module(
            pose_stack,
            movable_blocks=_movable_block_mask(data, pose_stack),
            include_frozen_energy=False,
        )
        order = tuple(sfxn.all_score_types())
        self._base_weights = (
            sfxn.weights_tensor().to(device=device, dtype=dtype).clone()
        )
        self._rep_index = (
            order.index(ScoreType.fa_ljrep) if ScoreType.fa_ljrep in order else None
        )

        # Only the search profile softens the diagonal; otherwise there is no
        # correction to add.
        self._singles = []
        softened = {
            score_type: weight
            for score_type, weight in _SEARCH_DIAGONAL_WEIGHTS.items()
            if data.weight_profile == "search" and score_type in order
        }
        if softened:
            correction = ScoreFunction(data.param_db, device)
            for score_type in softened:
                correction.set_weight(score_type, 1.0)
            correction_order = tuple(correction.all_score_types())
            self._correction_base = torch.tensor(
                [
                    (
                        softened[score_type] - float(sfxn.get_weight(score_type))
                        if score_type in softened
                        else 0.0
                    )
                    for score_type in correction_order
                ],
                dtype=dtype,
                device=device,
            )
            self._correction_rep_index = (
                correction_order.index(ScoreType.fa_ljrep)
                if ScoreType.fa_ljrep in softened
                else None
            )
            context = set(data.context_block_indices)
            for block in range(reference.max_n_blocks):
                if block in context:
                    continue
                single = repeat_pose_stack(select_blocks(reference, [block]), n_poses)
                begin = int(reference.block_coord_offset64[0, block])
                n_atoms = reference.block_type(0, block).n_atoms
                self._singles.append(
                    (
                        correction.render_whole_pose_scoring_module(single),
                        slice(begin, begin + n_atoms),
                    )
                )

        self._constant = torch.zeros((), dtype=dtype, device=device)
        coords = reference.coords.expand(n_poses, -1, -1).contiguous()
        with torch.no_grad():
            expected = _movable_block_pair_energy(data, reference.coords)[0]
            self._constant = (self(coords)[0] - expected).detach()

    def _set_weights(self) -> None:
        # Tensor ops and in-place copies only, so a captured CUDA graph keeps
        # following the repulsive ramp.
        scale = self._data._rep_scale.to(self._base_weights.dtype)
        weights = self._base_weights.clone()
        if self._rep_index is not None:
            weights[self._rep_index] = weights[self._rep_index] * scale
        with torch.no_grad():
            self._whole.weights.copy_(weights.unsqueeze(1))
        if not self._singles:
            return
        correction = self._correction_base.clone()
        if self._correction_rep_index is not None and self._rep_index is not None:
            full_rep = self._base_weights[self._rep_index] * scale
            correction[self._correction_rep_index] = (
                _SEARCH_DIAGONAL_WEIGHTS[ScoreType.fa_ljrep] - full_rep
            )
        for scorer, _ in self._singles:
            with torch.no_grad():
                scorer.weights.copy_(correction.unsqueeze(1))

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        self._set_weights()
        energies = self._whole(coords)
        for scorer, atoms in self._singles:
            energies = energies + scorer(coords[:, atoms].contiguous())
        return energies - self._constant


def _whole_movable_scorer_for(data: LiveLigandIntraData, n_poses: int):
    key = ("whole", n_poses)
    scorer = data._scorer_cache.get(key)
    if scorer is None:
        scorer = _WholeMovableScorer(data, n_poses)
        data._scorer_cache[key] = scorer
    return scorer


def _pair_weight_vectors(
    data: LiveLigandIntraData, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-term weights for off-diagonal and diagonal block pairs.

    Everything but the repulsive ramp is fixed for the life of the data, so it
    is built once per device and dtype.
    """

    key = (device, dtype)
    cached = data._weight_cache.get(key)
    if cached is None:
        base = data.score_function.weights_tensor().to(device=device, dtype=dtype)
        order = tuple(data.score_function.all_score_types())
        rep_index = (
            order.index(ScoreType.fa_ljrep) if ScoreType.fa_ljrep in order else None
        )
        diagonal_base = None
        if data.weight_profile == "search":
            diagonal_base = base.clone()
            for score_type, weight in _SEARCH_DIAGONAL_WEIGHTS.items():
                if score_type in order:
                    diagonal_base[order.index(score_type)] = weight
        cached = (base, diagonal_base, rep_index)
        data._weight_cache[key] = cached

    base, diagonal_base, rep_index = cached
    full_weights = base
    if rep_index is not None:
        full_weights = base.clone()
        full_weights[rep_index] = base[rep_index] * data._rep_scale.to(dtype)
    diagonal_weights = diagonal_base if diagonal_base is not None else full_weights
    return full_weights, diagonal_weights


def _movable_block_pair_energy(
    data: LiveLigandIntraData, ligand_coords: torch.Tensor
) -> torch.Tensor:
    """Weighted movable block-pair energy of every pose.

    A pair's weight depends only on its term and on whether it is a block with
    itself, so the block axes are reduced before weighting rather than after:
    the full ``[term, pose, block, block]`` product is never materialized.
    """

    scorer = _scorer_for(data, ligand_coords.shape[0])
    unweighted = scorer(
        ligand_coords.contiguous(), sum_terms=False, apply_weights=False
    )
    full_weights, diagonal_weights = _pair_weight_vectors(
        data, ligand_coords.device, ligand_coords.dtype
    )

    diagonal = torch.diagonal(unweighted, dim1=2, dim2=3)
    diagonal_sum = diagonal.sum(dim=2)
    off_diagonal_sum = unweighted.sum(dim=(2, 3)) - diagonal_sum
    # Context blocks are frozen, so the scorer skips them and every pair of
    # two of them is both constant and incomplete here.
    if data.context_block_indices:
        context = data._context_indices
        context_diagonal_sum = diagonal[:, :, context].sum(dim=2)
        context_pair_sum = unweighted[:, :, context][:, :, :, context].sum(dim=(2, 3))
        diagonal_sum = diagonal_sum - context_diagonal_sum
        off_diagonal_sum = off_diagonal_sum - (context_pair_sum - context_diagonal_sum)
    return (
        full_weights[:, None] * off_diagonal_sum
        + diagonal_weights[:, None] * diagonal_sum
    ).sum(dim=0)


def score_ligand_intra(
    data: LiveLigandIntraData, ligand_coords: torch.Tensor
) -> torch.Tensor:
    """Return weighted ligand-intramolecular energy for every ligand pose."""

    _validate_ligand_coords(data, ligand_coords)
    if not data.score_function.all_terms():
        return ligand_coords[:, :0].sum(dim=(1, 2))
    if data.ligand_pose_stack.max_n_blocks > 1:
        scorer = _whole_movable_scorer_for(data, ligand_coords.shape[0])
    else:
        scorer = _scorer_for(data, ligand_coords.shape[0])
    energies = scorer(ligand_coords.contiguous())
    expected_shape = (ligand_coords.shape[0],)
    if energies.shape != expected_shape:
        raise RuntimeError(
            f"ligand-intra scorer returned {tuple(energies.shape)}; "
            f"expected {expected_shape}"
        )
    return energies


def score_ligand_intra_with_gradients(
    data: LiveLigandIntraData, ligand_coords: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached weighted energies and compiled-autograd gradients."""

    _validate_ligand_coords(data, ligand_coords)
    with torch.enable_grad():
        coords_leaf = ligand_coords.detach().requires_grad_(True)
        energies = score_ligand_intra(data, coords_leaf)
        (gradients,) = torch.autograd.grad(energies.sum(), coords_leaf)
    return energies.detach(), gradients.detach()
