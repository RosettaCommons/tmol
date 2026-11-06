"""Resolve explicit and geometrically inferred head-to-tail polymer closures."""

import torch


def _find_cyclic_closures(
    canonical_ordering,
    chain_id,
    res_types,
    coords,
    cyclic_bonds=None,
    find_additional_cyclic_closures=True,
    cutoff_dis=2.0,
):
    """Return [pose, up residue, down residue] rows on the input device.

    Infer only closures between the ends of the same contiguous polymer run.
    Missing connection atoms, padding, single residues and nonpolymers cannot
    create a closure. Explicit rows take precedence over geometric inference.
    """
    device = res_types.device
    explicit = (
        torch.empty((0, 3), dtype=torch.int64, device=device)
        if cyclic_bonds is None
        else cyclic_bonds
    )
    if not find_additional_cyclic_closures or res_types.shape[1] == 0:
        return explicit

    connection = canonical_ordering.polymer_conn_inds
    up = torch.as_tensor(connection.up_atom_for_co_restype, device=device)
    down = torch.as_tensor(connection.down_atom_for_co_restype, device=device)
    if up.numel() == 0:
        return explicit
    from tmol.io.details.compiled._compiled import _canonical_cycles

    inputs = [
        res_types.to(torch.int64),
        chain_id,
        up.to(torch.int64),
        down.to(torch.int64),
        coords,
        explicit,
    ]
    if device.type == "mps":
        inputs = [x.cpu() for x in inputs]
    partners = _canonical_cycles(inputs, cutoff_dis).to(device)
    pose, last = (partners >= 0).nonzero(as_tuple=True)
    inferred = torch.stack((pose, last, partners[pose, last]), dim=1)
    return torch.cat((explicit, inferred), dim=0)
