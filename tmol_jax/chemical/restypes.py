"""JAX residue-set defaults around shared framework-neutral refinement."""

import typing
from typing import Optional
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
    ResidueTypeSet as _ResidueTypeSet,
    refined_from_raw,
    cached_refined_from_raw as _refined_from_raw,
)


@attr.s(auto_attribs=True)
class ResidueTypeSet(_ResidueTypeSet):
    """Shared residue grouping with the JAX default parameter context."""

    _default: typing.ClassVar[Optional["ResidueTypeSet"]] = None

    @classmethod
    def get_default(cls) -> "ResidueTypeSet":
        """Return the residue-type set built from the default chemical DB."""
        if cls._default is None:
            # Import lazily to avoid a circular import at module load time.
            from tmol_jax.chemical.patched_chemdb import default_patched_chemdb

            cls._default = cls.from_database(default_patched_chemdb())
        return cls._default


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
