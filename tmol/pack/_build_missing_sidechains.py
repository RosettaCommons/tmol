import copy

import torch

from tmol.types._torch import Tensor
from tmol.pose._pose_stack import PoseStack
from tmol.score._score_function import ScoreFunction
from tmol.pack._packer_task import PackerTask, PackerPalette
from tmol.pack._pack_rotamers import pack_rotamers
from tmol.pack.rotamer.dunbrack._dunbrack_chi_sampler import DunbrackChiSampler
from tmol.pack.rotamer._fixed_aa_chi_sampler import FixedAAChiSampler
from tmol.pack.rotamer._na_chi_sampler import NaChiRotamerSampler


def _build_missing_sidechains(
    pose_stack: PoseStack,
    sfxn: ScoreFunction,
    dunbrack_sampler: DunbrackChiSampler,
    block_has_missing_atoms: Tensor[torch.bool][:, :],
    no_optH: bool = False,
    na_sampler: NaChiRotamerSampler = None,
    has_missing_atoms: bool | None = None,
    seed: int | None = None,
) -> PoseStack:
    """Build missing sidechains and place hydrogens using per-block sampler assignment.

    Assigns samplers on a per-block basis in a single packing run:

    - Blocks with missing non-leaf (heavy) atoms: DunbrackChiSampler +
      FixedAAChiSampler for amino acids, NaChiRotamerSampler for nucleotides.
      The input conformation is not included as a rotamer because the sidechain
      is incomplete.
    - Complete blocks supported by OptHSampler: keep heavy atoms fixed while
      sampling proton chi angles and NHQ flips.
      The two sets are disjoint: NaChiRotamerSampler expands the RNA 2'-OH
      itself, exactly as DunbrackChiSampler expands protein proton chis, so a
      block must never be given both.
    - Complete blocks that OptHSampler cannot change (ALA, GLY, etc.) are
      frozen instead of becoming one-rotamer packing positions.

    When no_optH=True the old behavior is preserved: only Dunbrack runs for
    blocks with missing heavy atoms; all other blocks are frozen.

    Note: IncludeCurrentSampler is intentionally not used.  For Dunbrack
    blocks the native conformation is broken and must not appear as a rotamer.
    For OptH blocks, OptH includes native as rotamer-0 for NHQ residues.

    Args:
        pose_stack: The pose stack to process.
        sfxn: Score function used for packing.
        dunbrack_sampler: DunbrackChiSampler configured from the parameter DB.
        na_sampler: NaChiRotamerSampler configured from the parameter DB; when
            omitted, nucleotides with missing atoms get no rotamers.
        block_has_missing_atoms: Boolean tensor [n_poses, max_n_blocks]; True
            for blocks that have missing non-leaf (heavy) atoms.
        no_optH: When True, skip OptH and preserve old Dunbrack-only behavior.
        has_missing_atoms: Cached value of ``block_has_missing_atoms.any()``.
            Supplying it avoids a device synchronization in repeated builds.
        seed: Packer seed, as for :func:`~tmol.pack_rotamers`.

    Returns:
        PoseStack with missing sidechains built and (by default) hydrogens
        placed and optimized.
    """
    from tmol.pack.rotamer._opth_sampler import OptHSampler

    assert block_has_missing_atoms.device == pose_stack.device

    if has_missing_atoms is None:
        has_missing_atoms = bool(torch.any(block_has_missing_atoms))

    from tmol_core.pack.input_completion import prepare_input_completion_task

    task = prepare_input_completion_task(
        pose_stack,
        block_has_missing_atoms,
        task=PackerTask(pose_stack, PackerPalette()),
        dunbrack=dunbrack_sampler,
        fixed=FixedAAChiSampler(),
        opth=None if no_optH else OptHSampler(),
        nucleic_acid=na_sampler,
        has_missing_atoms=has_missing_atoms,
        supported_blocks=lambda sampler, pose: sampler.defines_rotamers_for_bts(
            pose.packed_block_types, pose.block_type_ind64.clamp_min(0)
        ),
        any_block=torch.any,
    )

    if not torch.is_grad_enabled() or not pose_stack.coords.requires_grad:
        return pack_rotamers(pose_stack, sfxn, task, verbose=False, seed=seed)

    # Input completion selects sampled coordinates as values. Preserve the
    # differentiable coordinates of frozen blocks, including their built leaves;
    # sampled chi construction has no inverse-kinematics coordinate derivative.
    with torch.no_grad():
        result = pack_rotamers(pose_stack, sfxn, task, verbose=False, seed=seed)
    frozen = task.is_real_block & ~task.per_block_is_block_type_allowed.any(dim=-1)
    pose, block = torch.nonzero(frozen, as_tuple=True)
    types = pose_stack.block_type_ind64[pose, block]
    source = pose_stack.block_coord_offset64[pose, block]
    target = result.block_coord_offset64[pose, block]
    from tmol.pack.rotamer._conformer_execution import copy_conformer_rows

    coords = result.coords.clone()
    copy_conformer_rows(
        pose_stack.packed_block_types.n_atoms[types].long(),
        source + pose * pose_stack.max_n_pose_atoms,
        target + pose * result.max_n_pose_atoms,
        pose_stack.coords.reshape(-1, 3),
        coords.reshape(-1, 3),
    )
    result = copy.copy(result)
    result.coords = coords
    return result
