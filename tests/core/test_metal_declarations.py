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


@pytest.mark.parametrize(
    "sites,bonds",
    [
        ([[0, 0, -1]], None),
        ([[0, 0, len(GEOMETRY_NAMES)]], None),
        ([[0, 0, 0], [0, 0, 1]], None),
        (None, [[0, 0, -1, 1, -1]]),
        (None, [[0, 0, -2, 1, 1]]),
        ([[0, 0, 0]], [[0, 0, 0, 1, 1], [0, 0, 0, 1, 2]]),
    ],
)
def test_invalid_or_conflicting_metal_rows_cannot_alias_valid_indices(sites, bonds):
    with pytest.raises(ValueError):
        declared_metal_sites(sites, bonds)


def test_canonical_donor_bounds_are_checked_before_coordinate_access():
    from tmol_core.chemical import ChemicalDatabase
    from tmol_core.chemical.patching import PatchedChemicalDatabase
    from tmol_core.io.canonical_ordering import CanonicalOrdering
    from tmol_core.io.metal_detection import find_metal_geometries

    chemical = PatchedChemicalDatabase.from_chem_db(ChemicalDatabase.get_default())
    co = CanonicalOrdering.from_chemdb(chemical)
    zinc = next(i for i, name in enumerate(co.restype_io_equiv_classes) if name == "ZN")
    cys = co.restype_io_equiv_classes.index("CYS")
    types = np.array([[zinc, cys, -1]], np.int32)

    def unexpected_coordinates():
        raise AssertionError("Invalid metadata must fail before coordinate access")

    for atom in (-1, len(co.restypes_ordered_atom_names["CYS"])):
        with pytest.raises(ValueError, match="Canonical donor atom"):
            find_metal_geometries(
                co,
                chemical,
                types,
                unexpected_coordinates,
                required_donors={(0, 0): {(1, atom)}},
            )
    with pytest.raises(ValueError, match="padded residue"):
        find_metal_geometries(
            co,
            chemical,
            types,
            unexpected_coordinates,
            required_donors={(0, 0): {(2, 0)}},
        )
    with pytest.raises(ValueError, match="site.*out of bounds"):
        find_metal_geometries(
            co,
            chemical,
            types,
            unexpected_coordinates,
            declared_sites={(0, 0): ("tetrahedral", ((4, 1, 0),))},
        )

    coords = np.full((1, 3, co.max_n_canonical_atoms, 3), np.nan, np.float32)
    coords[0, 0, co.restypes_atom_index_mapping["ZN"]["ZN"]] = [0, 0, 0]
    carbon = co.restypes_atom_index_mapping["CYS"]["CA"]
    coords[0, 1, carbon] = [2, 0, 0]
    requested = {(0, 0): {(1, carbon)}}
    with pytest.raises(ValueError, match="Explicit metal coordination"):
        find_metal_geometries(
            co,
            chemical,
            types,
            coords,
            required_donors=requested,
            require_all_donors=True,
        )
    _, assignments = find_metal_geometries(
        co, chemical, types, coords, required_donors=requested
    )
    assert all((1, carbon) not in got.donor_atoms for _, got in assignments)


def test_irregular_geometry_keeps_structure_defined_site_indices():
    from tmol_core.io.metal_detection import _validate_metal_inputs

    co = S(
        restype_io_equiv_classes=("ZN", "DON"),
        restypes_ordered_atom_names={"ZN": ("ZN",), "DON": ("O",)},
    )
    tables = S(ion_for_class={0: object()}, cluster_for_class={})
    _validate_metal_inputs(
        co,
        np.array([[0, 1]]),
        tables,
        {"irregular": ()},
        {(0, 0): ("irregular", ((7, 1, 0),))},
        None,
        None,
    )
