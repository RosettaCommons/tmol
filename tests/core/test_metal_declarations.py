from types import SimpleNamespace as S

import numpy as np
import pytest

from tmol_core.chemical import GEOMETRY_NAMES
from tmol_core.io.metal_detection import declared_metal_sites, validate_required_donors


def test_explicit_geometry_and_required_donors_preserve_site_identity():
    geometry = GEOMETRY_NAMES.index("tetrahedral")
    sites = np.array([[0, 2, geometry]], np.int64)
    bonds = np.array([[0, 2, 1, 3, 7], [0, 4, -1, 5, 9]], np.int64)
    original = bonds.copy()
    declared, required = declared_metal_sites(sites, bonds)
    assert declared == {(0, 2): ("tetrahedral", ((1, 3, 7),))}
    assert required == {(0, 4): {(5, 9)}}
    np.testing.assert_array_equal(bonds, original)
    validate_required_donors([((0, 4), S(donor_atoms=((5, 9),)))], required)
    with pytest.raises(ValueError, match="Explicit metal coordination.*resolved"):
        validate_required_donors([((0, 4), S(donor_atoms=((5, 8),)))], required)
    assert declared_metal_sites(None, None) == ({}, {})
