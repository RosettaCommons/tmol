from types import SimpleNamespace as S

from tmol_core.io.canonical_ordering import only_coordinates_a_metal


def test_default_palette_keeps_polymer_disulfides_and_unconnected_ligands():
    def residue(polymer, *kinematic):
        return S(
            properties=S(polymer=S(is_polymer=polymer)),
            connections=[S(kinematic=k) for k in kinematic],
        )

    assert only_coordinates_a_metal(residue(False, False, False))
    assert not only_coordinates_a_metal(residue(True, False))
    assert not only_coordinates_a_metal(residue(False))
    assert not only_coordinates_a_metal(residue(False, False, True))
