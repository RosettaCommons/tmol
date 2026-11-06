import torch

from typing import Optional
from tmol.types._torch import Tensor
from tmol.types._functional import _validate_args
from tmol.io._canonical_ordering import CanonicalOrdering
from tmol_core.io.disulfides import (  # noqa: F401
    find_disulfide_pairs as _find_disulf_numba,
    prepare_disulfide_search,
    resolve_disulfides,
)


@_validate_args
def _find_disulfides(
    canonical_ordering: CanonicalOrdering,
    res_types: Tensor[torch.int32][:, :],
    coords: Tensor[torch.float32][:, :, :, 3],
    disulfides: Optional[Tensor[torch.int64][:, 3]] = None,
    find_additional_disulfides: Optional[bool] = True,
    cutoff_dis: float = 2.5,
):
    if not canonical_ordering.cys_inds.cys_co_aa_inds:
        return (
            torch.zeros((0, 3), dtype=torch.int64, device=res_types.device),
            torch.zeros_like(res_types),
        )

    plan = prepare_disulfide_search(
        canonical_ordering.cys_inds.cys_co_aa_inds,
        res_types.cpu().numpy(),
        None if disulfides is None else disulfides.cpu().numpy(),
        find_additional_disulfides,
        (
            canonical_ordering.cys_inds.sg_atoms_for_co_cys
            if canonical_ordering.cys_inds.sg_atom_for_co_cys < 0
            else None
        ),
    )
    if (
        not plan.search
        and disulfides is not None
        and canonical_ordering.cys_inds.cys_co_aa_inds
    ):
        return disulfides, torch.from_numpy(plan.variants).to(device=res_types.device)
    found, variants = resolve_disulfides(
        plan,
        coords.detach().cpu().numpy() if plan.search else None,
        canonical_ordering.cys_inds.sg_atom_for_co_cys,
        cutoff_dis,
    )
    return (
        torch.from_numpy(found).to(device=coords.device),
        torch.from_numpy(variants).to(device=coords.device),
    )
