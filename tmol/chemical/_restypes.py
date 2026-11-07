"""Torch residue-set caching around the shared chemical refinement."""

import copy
from typing import Mapping, Sequence, List
import attr
import cattr
from toolz.curried import groupby
from tmol.database import ParameterDatabase
from tmol.database._patched_chemdb import PatchedChemicalDatabase
from tmol.database.chemical import RawResidueType
from tmol_core.chemical.restypes import (  # noqa: F401
    AtomIndex,
    ConnectionIndex,
    BondCount,
    BondType,
    BOND_TYPE_FROM_STR,
    UnresolvedAtomID,
    uaid_t,
    ResName3,
    IcoorIndex,
    RefinedResidueType,
    _l_base_name,
    _ONE_LETTER_CODE,
    _THREE_LETTER_CODE,
    _three2one,
    _one2three,
    _get_element_from_atom_name,
)


@attr.s(auto_attribs=True)
class ResidueTypeSet:
    """Collection of residue types refined from one chemical database."""

    __default = None
    __refined_cache = None

    @classmethod
    def get_default(cls) -> "ResidueTypeSet":
        """Load and return the residue type set constructed from the default param db"""
        if cls.__default is None:
            cls.__default = cls.from_database(ParameterDatabase.get_default().chemical)
        return cls.__default

    @classmethod
    def _refine(cls, r: RawResidueType) -> RefinedResidueType:
        return cattr.structure(cattr.unstructure(r), RefinedResidueType)

    @classmethod
    def _default_refined_cache(cls) -> Mapping[int, RefinedResidueType]:
        """``id(raw residue) -> RefinedResidueType`` prototype for the default DB.

        Refining a residue is expensive, and ligand-extended DBs reuse the
        default residue objects, so caching them here.

        These are prototypes only: callers get a shallow copy.
        """
        if cls.__refined_cache is None:
            default_residues = ParameterDatabase.get_default().chemical.residues
            cls.__refined_cache = {id(r): cls._refine(r) for r in default_residues}
        return cls.__refined_cache

    @classmethod
    def from_database(cls, chemical_db: PatchedChemicalDatabase):
        # Load and copy the default cache
        #   -> copy so later mutations do not modify cache
        cache = cls._default_refined_cache()
        residue_types = [
            copy.copy(cache[id(r)]) if id(r) in cache else cls._refine(r)
            for r in chemical_db.residues
        ]
        restype_map = groupby(lambda restype: restype.name3, residue_types)
        return cls(
            residue_types=residue_types,
            restype_map=restype_map,
            chem_db=chemical_db,
        )

    def extended(self, chemical_db: PatchedChemicalDatabase) -> "ResidueTypeSet":
        """This set grown to ``chemical_db``, whose residues begin with this set's; the
        types already here stay the same objects.

        Raises:
            ValueError: If this set does not hold one type per residue of its
                database, or ``chemical_db`` does not begin with those residues.
        """
        old = self.chem_db.residues
        if (
            len(self.residue_types) != len(old)
            or len(chemical_db.residues) < len(old)
            or any(a is not b for a, b in zip(chemical_db.residues, old))
        ):
            raise ValueError("chemical_db does not extend this residue type set")
        cache = self._default_refined_cache()
        added = [
            copy.copy(cache[id(r)]) if id(r) in cache else self._refine(r)
            for r in chemical_db.residues[len(old) :]
        ]
        return self.from_restype_list(chemical_db, [*self.residue_types, *added])

    @classmethod
    def from_restype_list(
        cls, chemical_db: PatchedChemicalDatabase, restypes: List[RefinedResidueType]
    ):
        restype_map = groupby(lambda restype: restype.name3, restypes)
        return cls(
            residue_types=restypes,
            restype_map=restype_map,
            chem_db=chemical_db,
        )

    residue_types: Sequence[RefinedResidueType]
    restype_map: Mapping[ResName3, Sequence[RefinedResidueType]]
    chem_db: PatchedChemicalDatabase
