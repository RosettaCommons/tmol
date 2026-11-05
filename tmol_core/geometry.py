"""NumPy geometry used during chemical and topological preparation."""

import numpy


def numpy_coord_dihedrals(a, b, c, d):
    """Return vectorized coordinate dihedrals for NumPy topology setup."""
    ba = a - b
    bc = c - b
    cd = d - c
    unit_bc = bc / numpy.linalg.norm(bc, axis=-1, keepdims=True)
    v = ba - numpy.sum(ba * unit_bc, axis=-1, keepdims=True) * unit_bc
    w = cd - numpy.sum(cd * unit_bc, axis=-1, keepdims=True) * unit_bc
    x = numpy.sum(v * w, axis=-1)
    y = numpy.sum(numpy.cross(unit_bc, v) * w, axis=-1)
    return numpy.asarray(numpy.arctan2(y, x), dtype=numpy.float32)
