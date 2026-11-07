from typing import TYPE_CHECKING

import numpy

if TYPE_CHECKING:
    from tmol_core.chemical.restypes import RefinedResidueType


def _eye4() -> numpy.ndarray:
    """Return a single-precision homogeneous identity transform."""

    return numpy.eye(4, dtype=numpy.float32)


def _normalize(v: numpy.ndarray) -> numpy.ndarray:
    """Return ``v`` scaled to unit length."""
    return v / numpy.linalg.norm(v)


def _frame_from_coords(
    p1: numpy.ndarray,
    p2: numpy.ndarray,
    p3: numpy.ndarray,
) -> numpy.ndarray:
    """Construct a homogeneous frame from three Cartesian points."""
    ht = _eye4()
    z = _normalize(p3 - p2)
    v21 = _normalize(p1 - p2)
    y = _normalize(v21 - numpy.dot(z, v21) * z)
    x = _normalize(numpy.cross(y, z))

    ht[0:3, 0] = x
    ht[0:3, 1] = y
    ht[0:3, 2] = z
    ht[0:3, 3] = p3
    return ht


def _rot_x(rot: float) -> numpy.ndarray:
    """Return a homogeneous rotation about the x-axis."""
    ht = _eye4()
    crot = numpy.cos(rot)
    srot = numpy.sin(rot)
    ht[1, 1] = crot
    ht[2, 1] = srot
    ht[1, 2] = -srot
    ht[2, 2] = crot
    return ht


def _rot_z(rot: float) -> numpy.ndarray:
    """Return a homogeneous rotation about the z-axis."""
    ht = _eye4()
    crot = numpy.cos(rot)
    srot = numpy.sin(rot)
    ht[0, 0] = crot
    ht[1, 0] = srot
    ht[0, 1] = -srot
    ht[1, 1] = crot
    return ht


def _trans_z(trans: float) -> numpy.ndarray:
    """Return a homogeneous translation along the z-axis."""
    ht = _eye4()
    ht[2, 3] = trans
    return ht


def _build_coords_from_icoors(
    icoors_ancestors: numpy.ndarray,
    icoors_geom: numpy.ndarray,
) -> numpy.ndarray:
    """Build Cartesian coordinates from ancestor indices and internal geometry."""
    # start with atom 1 at the origin
    # place atom 2 along the x axis
    # place atom 3 in the x-y plane
    # place all other atoms

    n_atoms = icoors_ancestors.shape[0]
    coords = numpy.zeros((n_atoms, 3), dtype=numpy.float32)
    if n_atoms < 2:
        return coords
    coords[1, 0] = icoors_geom[1, 2]
    if n_atoms < 3:
        return coords

    # coord 2 in the x-y plane. Atom 2 is bonded to either atom 1 or atom 0
    # (its only two possible predecessors); its parent is whichever it is
    # bonded to and the geometry (d, theta) is measured against that parent.
    #
    # parent == atom 1 (chain 0->1->2, the canonical case): walk d from atom 1,
    #   angle taken against the atom1->atom0 direction.
    # parent == atom 0 (atom 2 is a sibling of atom 1 off the root): walk d from
    #   the origin, angle taken against the atom0->atom1 direction (rotate by
    #   pi - theta so atom 2 still lands in the +y half plane).
    if icoors_ancestors[2, 0] == 1:
        ht_base = _eye4()
        ht_base[:3, 3] = coords[1, :]
        rot_2 = _rot_z(icoors_geom[2, 1])
    else:
        ht_base = _eye4()
        ht_base[:3, 3] = coords[0, :]
        rot_2 = _rot_z(numpy.pi - icoors_geom[2, 1])
    trans_2 = _eye4()
    trans_2[0, 3] = icoors_geom[2, 2]
    ht_2 = ht_base @ rot_2 @ trans_2

    coords[2, :] = ht_2[:3, 3]

    for i in range(3, icoors_ancestors.shape[0]):
        ht_i = _frame_from_coords(
            coords[icoors_ancestors[i, 2], :],
            coords[icoors_ancestors[i, 1], :],
            coords[icoors_ancestors[i, 0], :],
        )

        ht_rot_z = _rot_z(icoors_geom[i, 0])
        # negative rotation about the x axis will point z
        # at the next atom
        ht_rot_x = _rot_x(-icoors_geom[i, 1])
        ht_trans_z = _trans_z(icoors_geom[i, 2])

        ht_i = ht_i @ ht_rot_z @ ht_rot_x @ ht_trans_z
        coords[i, :3] = ht_i[:3, 3]
    return coords


def _build_ideal_coords(
    restype: "RefinedResidueType",
) -> numpy.ndarray:
    """Build ideal Cartesian coordinates for a refined residue type."""
    # lets build a kinforest using not the prioritized bonds,
    # but the icoors; let's not even use the scan algorithm.
    # let's just build the coordinates directly from the
    # tree provided in the icoors.

    return _build_coords_from_icoors(restype.icoors_ancestors, restype.icoors_geom)


def full_ideal_coords(restype) -> dict[str, numpy.ndarray]:
    icoor_index = {icoor.name: i for i, icoor in enumerate(restype.icoors)}
    ancestors = numpy.empty((len(restype.icoors), 3), dtype=numpy.int32)
    geom = numpy.empty((len(restype.icoors), 3), dtype=numpy.float64)
    for i, icoor in enumerate(restype.icoors):
        ancestors[i] = (
            icoor_index[icoor.parent],
            icoor_index[icoor.grand_parent],
            icoor_index[icoor.great_grand_parent],
        )
        geom[i] = (icoor.phi, icoor.theta, icoor.d)
    coords = _build_coords_from_icoors(ancestors, geom)
    return {
        icoor.name: coords[i].astype(numpy.float64)
        for i, icoor in enumerate(restype.icoors)
    }
