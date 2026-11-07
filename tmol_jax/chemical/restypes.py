"""JAX residue-set defaults around shared framework-neutral refinement."""

import copy
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
)


from tmol_core.chemical import RawResidueType
from tmol_core.weak_identity_cache import WeakIdentityLRU

_REFINED = WeakIdentityLRU(capacity=512)
_RAW_FIELDS = tuple(field.name for field in attr.fields(RawResidueType))
_DERIVED_FIELDS = tuple(
    field.name
    for field in attr.fields(RefinedResidueType)
    if field.name not in _RAW_FIELDS
)


def _refined_from_raw(raw):
    # Raw records are mutable; include every input field so replacing geometry,
    # identities or connectivity on the same owner cannot reuse stale topology.
    inputs = tuple(getattr(raw, name) for name in _RAW_FIELDS)
    try:
        hash(inputs)
    except TypeError:
        return refined_from_raw(raw)
    prototype = _REFINED.get_or_create(raw, inputs, lambda: refined_from_raw(raw))
    # Preserve the common constructor's raw-field sharing, while computed arrays
    # and mappings belong to each caller and can receive independent annotations.
    result = copy.copy(prototype)
    for name in _DERIVED_FIELDS:
        # Common refinement produces arrays, scalars and flat immutable-value
        # mappings/sets. Copy their storage without traversing chemistry again.
        setattr(result, name, copy.copy(getattr(prototype, name)))
    return result


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
