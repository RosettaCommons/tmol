"""Torch buffers and diagnostics around shared native connection policies."""

import torch


def _wire_canonical_connections(
    pbt, types, polymeric, first, last, broken, disulfides=None, cycles=None, extra=()
):
    from tmol._construction_ops import _canonical_wiring

    empty = torch.empty((0, 3), dtype=types.dtype, device=types.device)
    inputs = [
        types,
        polymeric,
        first,
        last,
        broken,
        pbt.up_conn_inds.to(types.dtype),
        pbt.down_conn_inds.to(types.dtype),
        pbt.canonical_dslf_conn_ind.to(types.dtype),
        empty if disulfides is None else disulfides,
        empty if cycles is None else cycles,
        torch.tensor(extra, dtype=types.dtype, device=types.device).reshape(-1, 5),
    ]
    if types.device.type == "mps":
        inputs = [x.cpu() for x in inputs]
    connections, status = _canonical_wiring(inputs, pbt.max_n_conn)
    connections = connections.to(types.device)
    if torch.any(status):
        _assert_connections_are_well_formed(pbt, types, connections)
        raise RuntimeError(
            f"malformed inter-residue connections: pose status {status.tolist()}"
        )
    return connections


def _assert_connections_are_well_formed(pbt, block_type_ind64, connections):
    """Validate endpoint bounds, reciprocity, and self-connections in the core."""
    from tmol._construction_ops import _canonical_connection_errors

    types = block_type_ind64
    if types.device.type == "mps":
        types, connections = types.cpu(), connections.cpu()
    bad = _canonical_connection_errors(types, connections)
    if bad.any():
        rows = torch.nonzero(bad)[:20]
        partners = connections[rows[:, 0], rows[:, 1], rows[:, 2]]
        rows = torch.cat((rows, partners), dim=1)
        raise RuntimeError(
            "malformed inter-residue connections (pose, block, port, partner, "
            f"partner port): {rows.tolist()}"
        )
