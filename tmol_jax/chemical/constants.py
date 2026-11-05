"""Shared chemical topology limits."""

from tmol_core.chemical.constants import (  # noqa: F401
    MAX_PATHS_FROM_CONNECTION,
    MAX_SIG_BOND_SEPARATION,
)
from tmol_core.chemical.restypes import _THREE_LETTER_CODE

one2three = dict(sorted(_THREE_LETTER_CODE.items()))
three2one = {v: k for k, v in one2three.items()}
canonical_aa_3letter = tuple(one2three.values())
canonical_aa_1letter = tuple(one2three)
