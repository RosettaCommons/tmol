"""Chemical choice preparation is distinct from packing execution."""

import numpy as np
import pytest

from .molecules import CASES, MolecularCase


@pytest.mark.parametrize("noncanonical_design", [False, True])
def test_palette_preserves_termini_chirality_and_ligand_identity(
    backend, noncanonical_design
):
    names = (
        "ALA",
        "DALA",
        "AIB",
        "PRO",
        "ALA:nterm",
        "GLY:nterm",
        "PRO:nterm",
        "CYD",
        "CYS",
        "LG1",
    )
    case = MolecularCase(
        CASES["ligand"].parameter_files, (names, ("ALA",)), ((1,) * len(names), (1,))
    )
    backend.molecule(case, terms=("ref",))
    counts, choices, original, repack = backend.palette_choices(
        noncanonical_design=noncanonical_design
    )
    types = backend.pose.packed_block_types.active_block_types

    def offered(block, mask=None):
        row = choices[0, block]
        valid = row >= 0
        if mask is not None:
            valid &= mask[0, block]
        return {types[i].name for i in row[valid]}

    assert ({"AIB", "DALA"} <= offered(0)) == noncanonical_design
    assert ({"AIB", "ALA"} <= offered(1)) == noncanonical_design
    assert {"ALA:nterm", "GLY:nterm", "PRO:nterm"} <= offered(4)
    assert not {"ALA", "GLY", "PRO"} & offered(4)
    assert offered(9) == {"LG1"}
    assert "CYS" not in offered(7)
    for index, name in enumerate(names):
        assert offered(index, original) == {name}
        assert name in offered(index, repack)
    np.testing.assert_array_equal(counts[1, 1:], 0)
    np.testing.assert_array_equal(choices[1, 1:], -1)
    assert not original[1, 1:].any() and not repack[1, 1:].any()
