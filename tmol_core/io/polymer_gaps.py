"""Polymer gap and orientation policies using the database's connection atoms.

Rows describe a shared residue topology; coordinates may contain several poses.
Bond rows are (first residue, atom, second residue, atom), ordered by residue.
Gap flags are updated in place. A negative residue class is padding.
"""

import numpy


def _polymer_ports(co, restypes):
    ports = co.polymer_conn_inds
    valid = restypes >= 0
    if valid.all():
        return tuple(
            numpy.asarray(values)[restypes]
            for values in (ports.up_atom_for_co_restype, ports.down_atom_for_co_restype)
        )
    result = []
    for values in (ports.up_atom_for_co_restype, ports.down_atom_for_co_restype):
        out = numpy.full(restypes.shape, -1, dtype=numpy.int64)
        out[valid] = numpy.asarray(values)[restypes[valid]]
        result.append(out)
    return tuple(result)


def break_polymer_gaps(
    not_connected, bonds, inferred, coords, restypes, chain_id, co, threshold
):
    """Break inferred polymer links across gaps; reject contradictory supplied bonds.

    Use connection atoms rather than side-chain contacts. A connection within
    the cutoff in any pose (or unresolved) retains the shared topology.
    Unknown bond provenance is treated as supplied, never silently discarded.
    """
    if threshold <= 0 or len(restypes) < 2:
        return bonds
    up, down = _polymer_ports(co, restypes)
    polymer = (up >= 0) | (down >= 0)
    if not polymer.any():
        return bonds
    pair = numpy.arange(len(restypes) - 1)

    def linked(a, b):
        distance = coords[:, pair, a[:-1]] - coords[:, pair + 1, b[1:]]
        distance = numpy.linalg.norm(distance, axis=-1)
        return (a[:-1] >= 0) & (b[1:] >= 0) & ~(distance > threshold).all(axis=0)

    gap = (chain_id[:-1] == chain_id[1:]) & polymer[:-1] & polymer[1:]
    gap &= ~linked(up, down) & ~linked(down, up)
    first, a, second, b = bonds.T
    joined = ((a == up[first]) & (b == down[second])) | (
        (a == down[first]) & (b == up[second])
    )
    adjacent = (second == first + 1) & (chain_id[first] == chain_id[second]) & joined
    distance = numpy.linalg.norm(coords[:, first, a] - coords[:, second, b], axis=-1)
    across = adjacent & (distance > threshold).all(axis=0)
    # A graph-specified orientation takes precedence over incidental contacts
    # between the opposite pair of ports (e.g. N1--C2 instead of C1--N2).
    gap[first[adjacent]] = True
    gap[first[adjacent & ~across]] = False
    if numpy.any(across & ~inferred):
        row = numpy.flatnonzero(across & ~inferred)[0]
        raise ValueError(
            "Declared polymer bond crosses a geometry gap between residue indices "
            f"{first[row]} and {second[row]} (cutoff {threshold:g} Angstrom). "
            "Correct the supplied connectivity/coordinates, or set "
            "missing_density_distance_threshold=0 to retain the declared topology."
        )
    not_connected[:-1, 1] |= gap
    not_connected[1:, 0] |= gap
    return bonds[~across]


def orient_polymer_gap_flags(not_connected, chain_id, restypes, bonds, co):
    """Translate input-neighbor gap flags to chemical down/up on reversed chains.

    Declared adjacent polymer bonds establish direction. Non-polymer links and
    cyclic closures do not vote; mixed directions have no single chain ordering.
    """
    if not len(bonds) or not numpy.any(not_connected):
        return
    restypes = numpy.asarray(restypes)
    first, a, second, b = bonds.T
    up, down = _polymer_ports(co, restypes)
    adjacent = (second == first + 1) & (chain_id[first] == chain_id[second])
    forward = adjacent & (a == up[first]) & (b == down[second])
    reverse = adjacent & (a == down[first]) & (b == up[second])
    for chain in numpy.unique(chain_id[first[reverse]]):
        if not numpy.any(forward & (chain_id[first] == chain)):
            members = (chain_id == chain) & (restypes >= 0)
            not_connected[members] = not_connected[members, ::-1]
