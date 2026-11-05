"""Named backbone torsion ownership shared by native scoring frontends."""


def _omega_connection(block_type):
    """The connection whose bond this block type's omega spans, or -1.

    A residue that names no reference reads no tables and so scores no omega;
    another term that would otherwise constrain the same rotation reads this to
    stay off it.
    """
    if block_type.rama_reference is None:
        return -1
    torsion = block_type.torsion_to_uaids.get("omega")
    if torsion is None:
        return -1
    # omega's third atom sits across the connection whose bond it turns about
    atom_index, connection, _offset = torsion[2]
    return connection if atom_index < 0 else -1
