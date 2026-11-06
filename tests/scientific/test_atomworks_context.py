"""Canonical encoding preserves database variants and parameter context."""

import attrs

from tmol_core.io.atomworks_encoding import atomworks_parameter_database
from .molecules import BACKBONE_CASES


def test_atomworks_subset_preserves_tautomers_disulfides_aliases_and_parameters(
    backend,
):
    b = backend.molecule(BACKBONE_CASES["hydroxyproline"], terms=())
    # The subset is chemistry-only: even a caller-specific scoring context must
    # retain its identity rather than silently acquiring default parameters.
    marker = object()
    base = attrs.evolve(b.parameter_database, scoring=marker)
    subset = atomworks_parameter_database(base)
    assert type(subset) is type(base) and subset.scoring is marker
    names = {r.name for r in subset.chemical.residues}
    assert {"HIS", "HIS_D", "CYD", "ALA:nterm", "ALA:cterm"} <= names
    assert "HYP" not in names
    assert any(
        a.name3 == "MSE" and a.read_as == "MET" for a in subset.chemical.name3_aliases
    )
