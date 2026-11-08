"""NumPy statistical-library records and archive readers shared by frontends."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import attr
import numpy as np
from cattr import structure

from tmol_core._yaml import _load_yaml
from .content_hash import _content_hash
from .library_records import (
    RamaMappingParams,
    OmegaBBDepMappingParams,
    DunMappingParams,
)


def _load_backbone_tables(path, record_type, array_fields):
    with np.load(path, allow_pickle=False) as data:
        return tuple(
            record_type(
                table_id=str(data[f"t{i}__table_id"]),
                **{
                    field: np.asarray(data[f"t{i}__{field}"], dtype=np.float32)
                    for field in array_fields
                },
                bbstep=tuple(float(x) for x in data[f"t{i}__bbstep"]),
                bbstart=tuple(float(x) for x in data[f"t{i}__bbstart"]),
            )
            for i in range(int(data["__n_tables"]))
        )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaTables:
    table_id: str
    table: np.ndarray
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _RamaLookupRaw:
    rama_lookup: Tuple[RamaMappingParams, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class RamaDatabase:
    uniq_id: str
    rama_lookup: Tuple[RamaMappingParams, ...]
    rama_tables: Tuple[RamaTables, ...] = ()

    @staticmethod
    def load_tables_from_npz(npz_path: str) -> Tuple[RamaTables, ...]:
        """Load rama energy tables from a numpy ``.npz`` archive.

        Uses the flat per-table layout written by
        ``scripts/build_jax.py``.
        """
        return _load_backbone_tables(npz_path, RamaTables, ("table",))

    @classmethod
    def from_file(cls, path):
        rows = structure(_load_yaml(path), _RamaLookupRaw)
        stem = os.path.splitext(path)[0]
        tables = cls.load_tables_from_npz(stem + ".npz")
        database = cls("", rows.rama_lookup, tables)
        return attr.evolve(database, uniq_id=database.content_id())

    def content_id(self):
        return _content_hash(self.rama_lookup, self.rama_tables)


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepTables:
    table_id: str
    mu: np.ndarray
    sigma: np.ndarray
    bbstep: Tuple[float, float]
    bbstart: Tuple[float, float]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class _OmegaBBDepLookupRaw:
    omega_bbdep_lookup: Tuple[OmegaBBDepMappingParams, ...]


@attr.s(auto_attribs=True, frozen=True, slots=True)
class OmegaBBDepDatabase:
    uniq_id: str
    bbdep_omega_lookup: Tuple[OmegaBBDepMappingParams, ...]
    bbdep_omega_tables: Tuple[OmegaBBDepTables, ...] = ()

    @staticmethod
    def load_tables_from_npz(npz_path: str) -> Tuple[OmegaBBDepTables, ...]:
        """Load omega_bbdep tables from a numpy ``.npz`` archive."""
        return _load_backbone_tables(npz_path, OmegaBBDepTables, ("mu", "sigma"))

    @classmethod
    def from_file(cls, path):
        rows = structure(_load_yaml(path), _OmegaBBDepLookupRaw)
        stem = os.path.splitext(path)[0]
        tables = cls.load_tables_from_npz(stem + ".npz")
        database = cls("", rows.omega_bbdep_lookup, tables)
        return attr.evolve(database, uniq_id=database.content_id())

    def content_id(self):
        return _content_hash(self.bbdep_omega_lookup, self.bbdep_omega_tables)


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericDataForAA:
    rotamers: np.ndarray
    rotamer_probabilities: np.ndarray
    rotamer_means: np.ndarray
    rotamer_stdvs: np.ndarray
    prob_sorted_rot_inds: np.ndarray
    backbone_dihedral_start: np.ndarray
    backbone_dihedral_step: np.ndarray
    rotamer_alias: np.ndarray
    backbone_is_mirrored: bool = False
    backbone_source_start: Optional[np.ndarray] = None

    def nchi(self) -> int:
        return int(self.rotamers.shape[1])


@attr.s(auto_attribs=True, slots=True, frozen=True)
class RotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA


@attr.s(auto_attribs=True, slots=True, frozen=True)
class SemiRotamericAADunbrackLibrary:
    table_name: str
    rotameric_data: RotamericDataForAA
    non_rot_chi_start: float
    non_rot_chi_step: float
    non_rot_chi_period: float
    rotameric_chi_rotamers: np.ndarray
    nonrotameric_chi_probabilities: np.ndarray
    rotamer_boundaries: np.ndarray


@attr.s(auto_attribs=True, slots=True, frozen=True)
class _DunbrackLookupRaw:
    dunbrack_lookup: Tuple[DunMappingParams, ...]


@attr.s(auto_attribs=True, slots=True, frozen=True)
class DunbrackRotamerLibrary:
    dun_lookup: Tuple[DunMappingParams, ...]
    rotameric_libraries: Tuple[RotamericAADunbrackLibrary, ...] = ()
    semi_rotameric_libraries: Tuple[SemiRotamericAADunbrackLibrary, ...] = ()

    @staticmethod
    def load_libraries_from_npz(
        npz_path: str,
    ) -> Tuple[
        Tuple[RotamericAADunbrackLibrary, ...],
        Tuple[SemiRotamericAADunbrackLibrary, ...],
    ]:
        """Load the Dunbrack rotamer libraries from a numpy ``.npz``.

        The archive layout matches the one written by
        ``scripts/build_jax.py`` -- flat arrays keyed by
        ``rot{i}__...`` / ``semi{i}__...`` prefixes, plus ``__n_rot``
        and ``__n_semi`` counters.
        """
        rot_data_fields = (
            "rotamers",
            "rotamer_probabilities",
            "rotamer_means",
            "rotamer_stdvs",
            "prob_sorted_rot_inds",
            "backbone_dihedral_start",
            "backbone_dihedral_step",
            "rotamer_alias",
        )

        def _load_rot_data(data, prefix: str) -> RotamericDataForAA:
            mirror_key = prefix + "data__backbone_is_mirrored"
            source_key = prefix + "data__backbone_source_start"
            return RotamericDataForAA(
                **{
                    field: np.asarray(data[prefix + "data__" + field]).copy()
                    for field in rot_data_fields
                },
                backbone_is_mirrored=(
                    bool(data[mirror_key]) if mirror_key in data else False
                ),
                backbone_source_start=(
                    np.asarray(data[source_key]).copy() if source_key in data else None
                ),
            )

        with np.load(npz_path, allow_pickle=False) as data:
            n_rot = int(data["__n_rot"])
            n_semi = int(data["__n_semi"])
            rot_libs_out = []
            for i in range(n_rot):
                prefix = f"rot{i}__"
                rot_libs_out.append(
                    RotamericAADunbrackLibrary(
                        table_name=str(data[prefix + "table_name"]),
                        rotameric_data=_load_rot_data(data, prefix),
                    )
                )
            semi_libs_out = []
            for i in range(n_semi):
                prefix = f"semi{i}__"
                semi_libs_out.append(
                    SemiRotamericAADunbrackLibrary(
                        table_name=str(data[prefix + "table_name"]),
                        rotameric_data=_load_rot_data(data, prefix),
                        non_rot_chi_start=float(data[prefix + "non_rot_chi_start"]),
                        non_rot_chi_step=float(data[prefix + "non_rot_chi_step"]),
                        non_rot_chi_period=float(data[prefix + "non_rot_chi_period"]),
                        rotameric_chi_rotamers=np.asarray(
                            data[prefix + "rotameric_chi_rotamers"]
                        ).copy(),
                        nonrotameric_chi_probabilities=np.asarray(
                            data[prefix + "nonrotameric_chi_probabilities"]
                        ).copy(),
                        rotamer_boundaries=np.asarray(
                            data[prefix + "rotamer_boundaries"]
                        ).copy(),
                    )
                )
        return (tuple(rot_libs_out), tuple(semi_libs_out))

    @classmethod
    def from_file(cls, path):
        rows = structure(_load_yaml(path), _DunbrackLookupRaw)
        stem = os.path.splitext(path)[0]
        rot, semi = cls.load_libraries_from_npz(stem + ".npz")
        return cls(rows.dunbrack_lookup, rot, semi)
