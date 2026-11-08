"""Library metadata remains usable before fitting or inspecting coefficient data."""

from types import SimpleNamespace as Record

import numpy as np
import pytest

from tmol_core.pack.dunbrack import DunbrackSamplingMetadata
from tmol_core.scoring.dunbrack import prepare_dunbrack_lookup
from tmol_core.scoring.dunbrack_topology import (
    dunbrack_scoring_table_indices,
    prepare_dunbrack_block,
)


class ShapeOnly:
    def __init__(self, shape):
        self.shape = shape

    def __array__(self, *args, **kwargs):
        raise AssertionError("Lightweight metadata must not read table payloads")


def library_database():
    def library(name, nchi, nrot):
        return Record(
            table_name=name,
            rotameric_data=Record(
                rotamers=ShapeOnly((nrot, nchi)),
                rotamer_probabilities=ShapeOnly((nrot, 36, 36)),
            ),
        )

    return Record(
        rotameric_libraries=(library("unused", 1, 3), library("mirror", 2, 9)),
        semi_rotameric_libraries=(library("custom", 3, 27),),
        dun_lookup=[
            Record(residue_name="CUSTOM", dun_table_name="custom"),
            Record(residue_name="D_CUSTOM", dun_table_name="mirror"),
            Record(residue_name="ALIAS", dun_table_name="custom"),
        ],
    )


def test_lightweight_metadata_preserves_original_indices_and_source_shapes():
    database = library_database()
    lookup = prepare_dunbrack_lookup(database)
    assert lookup == (
        {"CUSTOM": 2, "D_CUSTOM": 1, "ALIAS": 2},
        {"CUSTOM": -1, "D_CUSTOM": 1, "ALIAS": -1},
        {"CUSTOM": 0, "D_CUSTOM": -1, "ALIAS": 0},
    )
    metadata = DunbrackSamplingMetadata.from_database(database)
    assert dict(metadata.library_indices) == lookup[0]
    assert metadata.nchi_for_table == (1, 2, 3)
    assert metadata.library_sizes == (3, 9, 27)
    # A source edit does not alter the already-created sampling snapshot.
    database.dun_lookup[0].residue_name = "REPLACED"
    assert metadata.library_index("CUSTOM") == 2
    with pytest.raises(TypeError):
        metadata.library_indices["CUSTOM"] = 0

    # Scoring eligibility requires no table shapes or payload at all.
    for library in (*database.rotameric_libraries, *database.semi_rotameric_libraries):
        del library.rotameric_data
    assert prepare_dunbrack_lookup(database)[0]["REPLACED"] == 2


def test_scoring_and_sampling_keep_their_distinct_chemical_domains():
    database = library_database()
    lookup = prepare_dunbrack_lookup(database)
    metadata = DunbrackSamplingMetadata.from_database(database)

    def residue(base_name, *, polymer=True, kind="amino_acid", backbone="alpha_aa"):
        return Record(
            base_name=base_name,
            dunbrack_reference="CUSTOM",
            chi_samples=(),
            properties=Record(
                polymer=Record(
                    is_polymer=polymer, polymer_type=kind, backbone_type=backbone
                )
            ),
        )

    # Original IDs, custom aliases and D libraries survive without narrowing.
    assert dunbrack_scoring_table_indices(residue("CUSTOM"), lookup) == (2, -1, 0)
    assert dunbrack_scoring_table_indices(residue("ALIAS"), lookup) == (2, -1, 0)
    mirrored = residue("D_CUSTOM")
    assert dunbrack_scoring_table_indices(mirrored, lookup) == (1, 1, -1)
    assert metadata.library_for_type(mirrored) == 1  # own library before reference
    borrowed = residue("BORROWED")
    assert metadata.defines_rotamers(borrowed)
    assert metadata.library_for_type(borrowed) == 2
    for unsupported in (
        borrowed,
        residue("CUSTOM", polymer=False),
        residue("CUSTOM", kind="nucleic_acid"),
        residue("CUSTOM", backbone="beta_aa"),
    ):
        assert dunbrack_scoring_table_indices(unsupported, lookup) == (-1, -1, -1)
        # No auxiliary table access is permitted for a native-zero block.
        block = prepare_dunbrack_block(unsupported, lookup, None)
        assert block.rotamer_table_set == -1 and block.n_dihedrals == 0
        assert block.dih_uaids.shape == (0, 4, 3)
    assert not metadata.defines_rotamers(residue("CUSTOM", polymer=False))

    explicit = residue("EXPLICIT")
    explicit.dunbrack_reference = None
    explicit.chi_samples = (Record(is_proton=False),)
    assert metadata.defines_rotamers(explicit)
    assert metadata.library_for_type(explicit) == -1
    explicit.chi_samples = (Record(is_proton=True),)
    assert not metadata.defines_rotamers(explicit)


@pytest.mark.parametrize("defect", ("missing", "duplicate_residue", "duplicate_table"))
def test_lookup_validation_precedes_sampling_shape_access(defect):
    database = library_database()
    for library in (*database.rotameric_libraries, *database.semi_rotameric_libraries):
        del library.rotameric_data
    if defect == "missing":
        database.dun_lookup[0].dun_table_name = "ABSENT"
        message = "missing tables"
    elif defect == "duplicate_residue":
        database.dun_lookup.append(database.dun_lookup[0])
        message = "lookup names must be unique"
    else:
        database.semi_rotameric_libraries[0].table_name = "mirror"
        message = "table names must be unique"
    for prepare in (prepare_dunbrack_lookup, DunbrackSamplingMetadata.from_database):
        with pytest.raises(ValueError, match=message):
            prepare(database)


def test_empty_library_metadata_does_not_invent_a_sampling_state():
    database = Record(
        rotameric_libraries=(), semi_rotameric_libraries=(), dun_lookup=()
    )
    assert prepare_dunbrack_lookup(database) == ({}, {}, {})
    metadata = DunbrackSamplingMetadata.from_database(database)
    assert metadata.nchi_for_table == metadata.library_sizes == ()
    assert metadata.library_index("CUSTOM") == -1
    assert not np.asarray(metadata.library_sizes).size
