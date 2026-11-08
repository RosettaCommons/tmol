"""Dunbrack residue selection, unresolved torsions and padded block metadata."""

from itertools import count
from dataclasses import dataclass, fields
import numpy


@dataclass
class DunbrackBlockAttrs:
    n_dihedrals: int
    dih_uaids: numpy.ndarray
    rotamer_table_set: int
    rotameric_index: int
    semirotameric_index: int
    n_chi: int
    n_rotameric_chi: int
    probability_table_offset: int
    mean_table_offset: int
    rotamer_index_to_table_index_offset: int
    semirotameric_tableset_offset: int
    # a d-amino acid reads its l form's library at negated torsions; only the
    #    default used for a terminus needs the sign applied here, since a
    #    torsion measured from coordinates is already mirrored
    is_mirrored: int


def _empty_dunbrack_attrs() -> "DunbrackBlockAttrs":
    return DunbrackBlockAttrs(
        n_dihedrals=0,
        dih_uaids=numpy.full((0, 4, 3), -1, dtype=numpy.int32),
        rotamer_table_set=-1,
        rotameric_index=-1,
        semirotameric_index=-1,
        n_chi=0,
        n_rotameric_chi=0,
        probability_table_offset=-1,
        mean_table_offset=-1,
        rotamer_index_to_table_index_offset=-1,
        semirotameric_tableset_offset=numpy.array(-1),
        is_mirrored=0,
    )


def dunbrack_scoring_table_indices(block_type, table_indices):
    """Resolve scoring libraries from chemistry, without fitting tables.

    Only alpha amino acids with their own declared lookup are scored. A borrowed
    ``dunbrack_reference`` controls sampling, not scoring. Preserve original
    library indices; variants use their base name and D forms their own lookup.
    """
    polymer = block_type.properties.polymer
    if (
        not polymer.is_polymer
        or polymer.polymer_type != "amino_acid"
        or polymer.backbone_type != "alpha_aa"
    ):
        return (-1, -1, -1)
    all_index, rotameric_index, semirotameric_index = (
        table.get(block_type.base_name, -1) for table in table_indices
    )
    return (
        (all_index, rotameric_index, semirotameric_index)
        if all_index >= 0
        else (-1, -1, -1)
    )


def prepare_dunbrack_block(block_type, table_indices, aux):
    rotamer_table_set, rotameric_index, semirotameric_index = (
        dunbrack_scoring_table_indices(block_type, table_indices)
    )
    if rotamer_table_set < 0:
        return _empty_dunbrack_attrs()
    polymer = block_type.properties.polymer
    semirotameric = semirotameric_index != -1

    semirotameric_tableset_offset = (
        aux["semirotameric_tableset_offsets"][semirotameric_index]
        if semirotameric
        else -1
    )

    empty_tor = numpy.full((4, 3), -1, dtype=numpy.int32)

    phi_uaids = dunbrack_torsion("phi", block_type)
    if phi_uaids is None:
        phi_uaids = empty_tor

    psi_uaids = dunbrack_torsion("psi", block_type)
    if psi_uaids is None:
        psi_uaids = empty_tor

    chis = []
    n = count(1)
    while (t := dunbrack_torsion("chi" + str(next(n)), block_type)) is not None:
        chis += [t]

    dih_uaids = numpy.array([phi_uaids] + [psi_uaids] + chis)

    n_chi = aux["nchi_for_table_set"][rotamer_table_set]
    n_rotameric_chi = n_chi - (1 if semirotameric else 0)
    n_dihedrals = n_chi + 2
    probability_table_offset = aux["rotameric_prob_tableset_offsets"][rotameric_index]
    mean_table_offset = aux["rotameric_meansdev_tableset_offsets"][rotamer_table_set]
    rotamer_index_to_table_index_offset = aux["rotameric_chi_ri2ti_offsets"][
        rotamer_table_set
    ]

    dunbrack_attrs = DunbrackBlockAttrs(
        n_dihedrals=n_dihedrals,
        dih_uaids=dih_uaids,
        rotamer_table_set=rotamer_table_set,
        rotameric_index=rotameric_index,
        semirotameric_index=semirotameric_index,
        n_chi=n_chi,
        n_rotameric_chi=n_rotameric_chi,
        probability_table_offset=probability_table_offset,
        mean_table_offset=mean_table_offset,
        rotamer_index_to_table_index_offset=rotamer_index_to_table_index_offset,
        semirotameric_tableset_offset=semirotameric_tableset_offset,
        is_mirrored=int(polymer.sidechain_chirality == "d"),
    )

    return dunbrack_attrs


def dunbrack_torsion(name, block_type):
    if name in block_type.torsion_to_uaids:
        return numpy.array(block_type.torsion_to_uaids[name], dtype=numpy.int32)
    return None


def pack_dunbrack_field(values, default_fill=-1):
    arrays = [
        None if value is None else numpy.asarray(value, dtype=numpy.int32)
        for value in values
    ]
    shapes = [array.shape for array in arrays if array is not None]
    maximum = tuple(max(dim) for dim in zip(*shapes))
    packed = numpy.full((len(arrays), *maximum), default_fill, dtype=numpy.int32)
    for index, array in enumerate(arrays):
        if array is not None:
            packed[(index, *(slice(0, size) for size in array.shape))] = array
    return packed


def pack_dunbrack_blocks(records):
    """Return native int32 arrays in the canonical DunbrackBlockAttrs order."""
    packed = {
        field.name: pack_dunbrack_field([getattr(r, field.name) for r in records])
        for field in fields(DunbrackBlockAttrs)
    }
    if not records:
        packed["dih_uaids"] = numpy.empty((0, 0, 4, 3), dtype=numpy.int32)
    return packed
