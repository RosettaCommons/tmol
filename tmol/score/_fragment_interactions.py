from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tmol.pose._split_block_mapping import SplitBlockMapping
from tmol.types._torch import Tensor
from tmol_core.scoring.fragment_interactions import (  # noqa: F401
    FragmentInteractionScores,
    FragmentRecord,
    _LegacyFragmentRecord,
    _LegacyFragmentMapping,
    _NormalizedFragmentRecord,
    fragment_mapping_plan,
    fragments_per_chunk,
)

if TYPE_CHECKING:
    from tmol.pose._pose_stack import PoseStack
    from tmol.score._score_function import ScoreFunction


_FRAGMENT_REDUCTION_WORKSPACE_BYTES = 256 * 1024 * 1024


@dataclass(frozen=True)
class _FragmentMappingData:
    """Validated mapping indices cached on one pose topology."""

    source_mapping: SplitBlockMapping | _LegacyFragmentMapping
    fragment_records: tuple[FragmentRecord, ...]
    record_linear_indices: Tensor[torch.int64][:]
    fragment_block_indices: Tensor[torch.int64][:]


def _fragment_mapping_data(
    pose_stack: PoseStack,
    mapping: SplitBlockMapping | _LegacyFragmentMapping,
) -> _FragmentMappingData:
    """Normalize and cache current or compatibility mapping records."""
    cached = getattr(pose_stack, "_fragment_interaction_mapping_data", None)
    tensor_device = pose_stack.block_type_ind.device
    if (
        isinstance(cached, _FragmentMappingData)
        and cached.source_mapping is mapping
        and cached.record_linear_indices.device == tensor_device
    ):
        return cached

    plan = fragment_mapping_plan(mapping, pose_stack.n_poses, pose_stack.max_n_blocks)

    data = _FragmentMappingData(
        source_mapping=mapping,
        fragment_records=plan.fragment_records,
        record_linear_indices=torch.tensor(
            plan.record_linear_indices,
            dtype=torch.int64,
            device=tensor_device,
        ),
        fragment_block_indices=torch.tensor(
            plan.fragment_block_indices,
            dtype=torch.int64,
            device=tensor_device,
        ),
    )
    pose_stack._fragment_interaction_mapping_data = data
    return data


def _sum_fragment_partner_scores(
    block_pair_scores: torch.Tensor,
    partner_mask: Tensor[torch.bool][:, :],
    fragment_block_indices: Tensor[torch.int64][:],
    *,
    max_workspace_bytes: int = _FRAGMENT_REDUCTION_WORKSPACE_BYTES,
) -> torch.Tensor:
    """Reduce both block-matrix orientations for many fragments at once.

    The gathered rows and columns are chunked so that their combined temporary
    storage stays near ``max_workspace_bytes``. This keeps the number of CUDA
    launches independent of the number of fragments within each chunk.

    Args:
        block_pair_scores: Weighted scores shaped
            ``[n_terms, n_poses, n_blocks, n_blocks]``.
        partner_mask: Selected partner blocks shaped ``[n_poses, n_blocks]``.
        fragment_block_indices: Fragment block columns shared by every pose,
            shaped ``[n_fragments]``.
        max_workspace_bytes: Target upper bound for the two gathered score
            tensors in one reduction chunk.

    Returns:
        Scores shaped ``[n_terms, n_poses, n_fragments]``.
    """
    n_terms, n_poses, n_blocks, _ = block_pair_scores.shape
    n_fragments = fragment_block_indices.numel()
    if n_fragments == 0 or n_terms == 0 or n_poses == 0 or n_blocks == 0:
        zero_scores = block_pair_scores.sum(dim=(-2, -1)).unsqueeze(-1)
        return zero_scores.expand(n_terms, n_poses, n_fragments)

    chunk_size = fragments_per_chunk(
        n_terms,
        n_poses,
        n_blocks,
        block_pair_scores.element_size(),
        max_workspace_bytes,
    )
    partner_weights = partner_mask.to(dtype=block_pair_scores.dtype)

    partials = []
    for fragment_indices in fragment_block_indices.split(chunk_size):
        selected_scores = torch.index_select(block_pair_scores, 2, fragment_indices)
        selected_scores.add_(
            torch.index_select(block_pair_scores, 3, fragment_indices).transpose(-1, -2)
        )
        partials.append(torch.einsum("tpfb,pb->tpf", selected_scores, partner_weights))
    return partials[0] if len(partials) == 1 else torch.cat(partials, dim=2)


def calculate_fragment_interactions(
    pose_stack: PoseStack,
    partner_mask: Tensor[torch.bool][:, :],
    *,
    sfxn: ScoreFunction,
    mapping: SplitBlockMapping | _LegacyFragmentMapping | None = None,
    sum_terms: bool = False,
) -> FragmentInteractionScores:
    """Return each ligand fragment's interaction with ``partner_mask``.

    The connected multi-block pose is scored once. Fragment-fragment entries
    remain in the block-pair matrix and are not silently assigned to either
    fragment. ``sfxn`` must use the same ligand-extended parameter database as
    ``pose_stack``.

    Args:
        pose_stack: Connected poses with identical fragment block layouts.
        partner_mask: Partner blocks shaped ``[n_poses, max_n_blocks]``. The
            mask must exclude every ligand fragment block.
        sfxn: Score function built from the pose's parameter database.
        mapping: Optional split-block mapping. By default the mapping attached
            to ``pose_stack`` is used. Legacy fragmented-ligand mappings remain
            accepted for compatibility.
        sum_terms: Sum the score-term dimension when true.

    Returns:
        Fragment scores shaped ``[n_terms, n_poses, n_fragments]``, or
        ``[n_poses, n_fragments]`` when ``sum_terms`` is true, plus the pose-zero
        mapping records that define the fragment columns.

    Raises:
        TypeError: If ``partner_mask`` is not Boolean.
        ValueError: If the mask, score function, or mapping is incompatible
            with the poses.
    """
    if mapping is None:
        mapping = getattr(pose_stack, "split_block_mapping", None)
    if mapping is None:
        raise ValueError("No split-block mapping was supplied or attached to the pose")
    if partner_mask.shape != pose_stack.block_type_ind.shape:
        raise ValueError(
            "partner_mask must have shape [n_poses, max_n_blocks]; got "
            f"{tuple(partner_mask.shape)}"
        )
    if partner_mask.dtype != torch.bool:
        raise TypeError("partner_mask must be a boolean tensor")
    if partner_mask.device != pose_stack.device:
        raise ValueError("partner_mask must be on the same device as pose_stack")
    if sfxn is None:
        raise ValueError(
            "sfxn is required and must use the ligand-extended parameter database"
        )

    mapping_data = _fragment_mapping_data(pose_stack, mapping)
    if bool(partner_mask.flatten()[mapping_data.record_linear_indices].any()):
        raise ValueError("partner_mask must not include ligand fragment blocks")

    scorer = sfxn.render_block_pair_scoring_module(pose_stack, interaction_only=True)
    block_pair_scores = scorer(pose_stack.coords, sum_terms=False)
    result = _sum_fragment_partner_scores(
        block_pair_scores,
        partner_mask,
        mapping_data.fragment_block_indices,
    )
    if sum_terms:
        result = result.sum(dim=0)
    return FragmentInteractionScores(
        scores=result,
        mapping=mapping_data.fragment_records,
    )
