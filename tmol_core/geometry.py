"""NumPy geometry used during chemical and topological preparation."""

import numpy


def numpy_coord_dihedrals(a, b, c, d, *, output_dtype=numpy.float32):
    """Return dihedrals, retaining the legacy float32 default unless requested."""
    ba = a - b
    bc = c - b
    cd = d - c
    unit_bc = bc / numpy.linalg.norm(bc, axis=-1, keepdims=True)
    v = ba - numpy.sum(ba * unit_bc, axis=-1, keepdims=True) * unit_bc
    w = cd - numpy.sum(cd * unit_bc, axis=-1, keepdims=True) * unit_bc
    x = numpy.sum(v * w, axis=-1)
    y = numpy.sum(numpy.cross(unit_bc, v) * w, axis=-1)
    return numpy.asarray(numpy.arctan2(y, x), dtype=output_dtype)
