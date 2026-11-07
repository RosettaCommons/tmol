"""Zero-valued patch coordinates are explicit overrides, not missing values."""

from tmol_jax.database.chemical import Icoor, IcoorVariant
from tmol_jax.chemical.patched_chemdb import update_icoor


def test_patch_distinguishes_zero_from_absent():
    source = Icoor("A", 1.5, 0.8, 1.2, "B", "C", "D")
    mapping = {"<A>": "A"}
    inherited = update_icoor((source,), (IcoorVariant("A", source="<A>"),), (), mapping)
    zeroed = update_icoor(
        (source,), (IcoorVariant("A", source="<A>", phi=0.0),), (), mapping
    )
    assert inherited[0].phi == 1.5
    assert zeroed[0].phi == 0.0
    assert zeroed[0].theta == 0.8


def test_ideal_coordinates_follow_atom_names_when_atom_order_changes():
    import attrs
    import numpy as np
    from tmol_jax.database import ParameterDatabase
    from tmol_jax.chemical.restypes import _refined_from_raw
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    database = ParameterDatabase.get_default().chemical
    raw = next(residue for residue in database.residues if residue.name == "ALA")
    residue = _refined_from_raw(attrs.evolve(raw, atoms=raw.atoms[::-1]))
    packed = PackedBlockTypes.from_restypes(database, [residue])
    nitrogen = packed.ideal_coords[0, residue.atom_to_idx["N"]]
    alpha_carbon = packed.ideal_coords[0, residue.atom_to_idx["CA"]]
    np.testing.assert_allclose(nitrogen, 0, atol=1e-7)
    np.testing.assert_allclose(
        np.linalg.norm(alpha_carbon - nitrogen),
        residue.icoors[residue.icoors_index["CA"]].d,
        atol=1e-6,
    )


def test_legacy_patched_database_keeps_shared_extension_operations():
    import attrs
    from tmol_jax.database.chemical import ChemicalDatabase
    from tmol_jax.chemical.patched_chemdb import PatchedChemicalDatabase
    from tmol_jax.pose.packed_block_types import PackedBlockTypes

    raw = ChemicalDatabase.get_default()
    original = PatchedChemicalDatabase.from_chem_db(raw)
    alanine = next(residue for residue in raw.residues if residue.name == "ALA")
    custom = attrs.evolve(
        alanine, name="UXA", base_name="UXA", name3="UXA", io_equiv_class="UXA"
    )
    extended = original.with_added_residues((custom,))
    assert isinstance(extended, PatchedChemicalDatabase)
    assert not any(r.name == "UXA" for r in original.residues)
    palette = PackedBlockTypes.default(extended)
    names = {r.name for r in palette.active_block_types}
    assert {"UXA", "UXA:nterm", "UXA:cterm"} <= names
    base = next(r for r in palette.active_block_types if r.name == "UXA")
    assert tuple(a.name for a in base.atoms) == tuple(a.name for a in alanine.atoms)


def test_refinement_preserves_mutable_output_and_changed_raw_geometry():
    import attrs
    import numpy as np
    from tmol_core.chemical.restypes import refined_from_raw
    from tmol_jax.chemical.restypes import _refined_from_raw
    from tmol_jax.database import ParameterDatabase

    raw = attrs.evolve(
        next(
            r
            for r in ParameterDatabase.get_default().chemical.residues
            if r.name == "ALA"
        )
    )
    first = _refined_from_raw(raw)
    expected = refined_from_raw(raw)
    first.ideal_coords[:] = 99
    first.path_distance[:] = 0
    first.atom_to_idx = {}
    first.icoors_index.clear()
    first.atom_names_set.clear()
    second = _refined_from_raw(raw)
    np.testing.assert_array_equal(second.ideal_coords, expected.ideal_coords)
    np.testing.assert_array_equal(second.path_distance, expected.path_distance)
    assert second.atom_to_idx == expected.atom_to_idx
    assert second.icoors_index == expected.icoors_index
    assert second.atom_names_set == expected.atom_names_set
    assert second.atoms is raw.atoms
    raw.icoors = tuple(
        attrs.evolve(icoor, d=icoor.d + 0.25) if icoor.name == "CB" else icoor
        for icoor in raw.icoors
    )
    changed = _refined_from_raw(raw)
    cb, ca = (changed.icoors_index[name] for name in ("CB", "CA"))
    old_length = np.linalg.norm(second.ideal_coords[cb] - second.ideal_coords[ca])
    new_length = np.linalg.norm(changed.ideal_coords[cb] - changed.ideal_coords[ca])
    np.testing.assert_allclose(new_length - old_length, 0.25, atol=1e-6)
    np.testing.assert_array_equal(second.ideal_coords, expected.ideal_coords)


def test_refinement_does_not_retain_custom_raw_owner():
    import attrs
    import gc
    import weakref
    from tmol_jax.chemical.restypes import _refined_from_raw
    from tmol_jax.database import ParameterDatabase

    raw = attrs.evolve(
        next(
            r
            for r in ParameterDatabase.get_default().chemical.residues
            if r.name == "ALA"
        ),
        name="CUSTOM",
    )
    reference = weakref.ref(raw)
    residue = _refined_from_raw(raw)
    del raw
    gc.collect()
    assert reference() is None
    assert residue.name == "CUSTOM"
