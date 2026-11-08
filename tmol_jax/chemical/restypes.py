"""JAX residue-set defaults around shared framework-neutral refinement."""

import typing
from typing import Dict, List, Mapping, Optional, Sequence
import attr
from tmol_core.chemical.restypes import (  # noqa: F401
    AtomIndex,
    ConnectionIndex,
    BondCount,
    UnresolvedAtomID,
    uaid_t,
    ResName3,
    IcoorIndex,
    RefinedResidueType,
    refined_from_raw,
    cached_refined_from_raw as _refined_from_raw,
)


@attr.s(auto_attribs=True)
class ResidueTypeSet:
    """A set of residue types grouped by their ``name3`` code."""

    residue_types: Sequence[RefinedResidueType]
    restype_map: Mapping[ResName3, Sequence[RefinedResidueType]]
    chem_db: typing.Any  # PatchedChemicalDatabase (avoid cyclic import)

    _default: typing.ClassVar[Optional["ResidueTypeSet"]] = None

    @classmethod
    def get_default(cls) -> "ResidueTypeSet":
        """Return the residue-type set built from the default chemical DB."""
        if cls._default is None:
            # Import lazily to avoid a circular import at module load time.
            from tmol_jax.chemical.patched_chemdb import default_patched_chemdb

            cls._default = cls.from_database(default_patched_chemdb())
        return cls._default

    @classmethod
    def from_database(cls, chemical_db) -> "ResidueTypeSet":
        residue_types = [_refined_from_raw(r) for r in chemical_db.residues]
        restype_map: Dict[ResName3, List[RefinedResidueType]] = {}
        for rt in residue_types:
            restype_map.setdefault(rt.name3, []).append(rt)
        return cls(
            residue_types=residue_types,
            restype_map=restype_map,
            chem_db=chemical_db,
        )


__all__ = [
    "AtomIndex",
    "ConnectionIndex",
    "BondCount",
    "UnresolvedAtomID",
    "uaid_t",
    "ResName3",
    "IcoorIndex",
    "RefinedResidueType",
    "ResidueTypeSet",
]
