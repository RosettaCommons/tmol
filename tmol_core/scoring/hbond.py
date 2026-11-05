"""Complete validated hydrogen-bond family tables, without framework tensors.

Polynomial layout: xmin, xmax, min_val, max_val, then c10 through c0.
The canonical coefficients stay float64 through native table preparation.
"""

from dataclasses import dataclass

import attr
import numpy

from .atom_types import atom_type_parameters, block_type_indices
from .tiling import tile_subset_indices

from .atom_types import HYBRIDIZATION_NAMES


@dataclass
class HBondPairData:
    donor_names: tuple
    acceptor_names: tuple
    donor_weight: numpy.ndarray
    acceptor_weight: numpy.ndarray
    acceptor_hybridization: numpy.ndarray
    polynomials: dict


def hbond_pair_parameters(chemical_database, hbond_database):
    donors = {g.name: g for g in hbond_database.donor_type_params}
    donor_type_to_index = {name: i for i, name in enumerate(donors)}
    donor_type_index = tuple(donor_type_to_index)

    acceptors = {g.name: g for g in hbond_database.acceptor_type_params}
    acceptor_type_to_index = {name: i for i, name in enumerate(acceptors)}
    acceptor_type_index = tuple(acceptor_type_to_index)

    atom_type_hybridization = {
        a.name: a.acceptor_hybridization for a in chemical_database.atom_types
    }
    acceptor_type_hybridization = {
        g.acceptor_type: atom_type_hybridization[g.a]
        for g in hbond_database.acceptor_atom_types
    }

    # Preserve the historical last-definition-wins rules while ordering the
    # dense table once. Repeated advanced-index assignment is unnecessary.
    poly_values = {
        p.name: [p.xmin, p.xmax, p.min_val, p.max_val]
        + [getattr(p, f"c_{n}") for n in range(10, -1, -1)]
        for p in hbond_database.polynomial_parameters
    }
    kinds = ("AHdist", "cosBAH", "cosAHD")
    pairs = {}
    for row in hbond_database.pair_parameters:
        if row.donor_type not in donors or row.acceptor_type not in acceptors:
            raise ValueError(
                f"HBond pair references unknown family: "
                f"{row.donor_type}, {row.acceptor_type}"
            )
        for kind in kinds:
            name = getattr(row, kind)
            if name not in poly_values:
                raise ValueError(f"HBond pair references unknown polynomial {name!r}")
        pairs[row.donor_type, row.acceptor_type] = row
    rows = []
    for donor in donors:
        for acceptor in acceptors:
            row = pairs.get((donor, acceptor))
            if row is None:
                raise ValueError(f"Missing HBond pair parameters: {donor}, {acceptor}")
            rows.append(row)
    shape = (len(donors), len(acceptors))

    def weights(records, kind):
        with numpy.errstate(over="ignore"):
            values = numpy.array(
                [g.weight for g in records.values()], dtype=numpy.float32
            )
        invalid = ~numpy.isfinite(values)
        if invalid.any():
            names = [name for name, bad in zip(records, invalid) if bad]
            raise ValueError(f"HBond {kind} weights {names} are non-finite in float32")
        return values

    def pack_polynomial(kind):
        names = [getattr(row, kind) for row in rows]
        values = numpy.array(
            [poly_values[name] for name in names], dtype=numpy.float64
        ).reshape(shape + (15,))
        invalid = ~numpy.isfinite(values).all(axis=-1).ravel()
        if invalid.any():
            names = list(
                dict.fromkeys(name for name, bad in zip(names, invalid) if bad)
            )
            raise ValueError(f"HBond polynomials {names} contain non-finite parameters")
        return values

    hybrids = [acceptor_type_hybridization[name] for name in acceptors]
    if any(value not in HYBRIDIZATION_NAMES for value in hybrids):
        raise ValueError("HBond acceptor family has unknown hybridization")
    hybridization = numpy.asarray(
        [HYBRIDIZATION_NAMES.index(value) for value in hybrids], dtype=numpy.int32
    )
    return HBondPairData(
        donor_type_index,
        acceptor_type_index,
        numpy.broadcast_to(weights(donors, "donor")[:, None], shape),
        numpy.broadcast_to(weights(acceptors, "acceptor")[None, :], shape),
        numpy.broadcast_to(hybridization[None, :], shape),
        {kind: pack_polynomial(kind) for kind in kinds},
    )


def attached_H_for_don(atom_is_hydrogen, D_idx, bonds, bond_spans):
    donH = numpy.full(atom_is_hydrogen.shape, -1, dtype=numpy.int32)
    heavy_at_don_for_H = numpy.full(atom_is_hydrogen.shape, -1, dtype=numpy.int32)
    which_H_for_hvy = numpy.full(atom_is_hydrogen.shape, -1, dtype=numpy.int32)
    n_attached_H_for_D = numpy.full(D_idx.shape, 0, dtype=numpy.int32)

    n_donH = 0
    for i, d in enumerate(D_idx):
        count_H_for_d = 0
        for b_ind in range(bond_spans[d, 0], bond_spans[d, 1]):
            neighb = bonds[b_ind, 1]
            if atom_is_hydrogen[neighb]:
                donH[n_donH] = neighb
                heavy_at_don_for_H[n_donH] = d
                which_H_for_hvy[n_donH] = count_H_for_d
                count_H_for_d += 1
                n_donH += 1
        n_attached_H_for_D[i] = count_H_for_d
    donH = donH[:n_donH]
    heavy_at_don_for_H = heavy_at_don_for_H[:n_donH]
    which_H_for_hvy = which_H_for_hvy[:n_donH]

    sort_inds = numpy.argsort(donH)
    return (
        donH[sort_inds],
        heavy_at_don_for_H[sort_inds],
        which_H_for_hvy[sort_inds],
        n_attached_H_for_D,
    )


@attr.s(auto_attribs=True, frozen=True, slots=True)
class HBondBlockTypeParams:
    donH_inds: numpy.ndarray
    don_hvy_inds: numpy.ndarray
    acc_inds: numpy.ndarray
    tile_n_donH: numpy.ndarray
    tile_n_don_hvy: numpy.ndarray
    tile_n_acc: numpy.ndarray
    tile_donH_inds: numpy.ndarray
    tile_donH_hvy_inds: numpy.ndarray
    tile_don_hvy_inds: numpy.ndarray
    tile_which_donH_of_donH_hvy: numpy.ndarray
    tile_acc_inds: numpy.ndarray
    tile_donorH_type: numpy.ndarray
    tile_acceptor_type: numpy.ndarray
    tile_acceptor_hybridization: numpy.ndarray
    tile_acceptor_n_attached_H: numpy.ndarray
    is_hydrogen: numpy.ndarray

    def __attrs_post_init__(self):
        vectors = {
            "donH_inds",
            "don_hvy_inds",
            "acc_inds",
            "tile_n_donH",
            "tile_n_don_hvy",
            "tile_n_acc",
            "is_hydrogen",
        }
        for field in attr.fields(type(self)):
            value = getattr(self, field.name)
            rank = 1 if field.name in vectors else 2
            if (
                not isinstance(value, numpy.ndarray)
                or value.dtype != numpy.int32
                or value.ndim != rank
            ):
                raise TypeError(f"{field.name} must be an int32 array of rank {rank}")


@dataclass(frozen=True)
class HBondChemicalData:
    atom_names: tuple
    hybridization: numpy.ndarray
    is_hydrogen: numpy.ndarray
    acceptor_types: numpy.ndarray
    donor_types: numpy.ndarray

    @classmethod
    def from_database(cls, chemical_database, hbond_database):
        names, flags = atom_type_parameters(chemical_database.atom_types)

        def families(rows, atom_field, family_field, parameters):
            indices = {
                name: i
                for i, name in enumerate(dict.fromkeys(row.name for row in parameters))
            }
            mapping = {
                getattr(row, atom_field): indices.get(getattr(row, family_field), -1)
                for row in rows
            }
            return numpy.asarray(
                [mapping.get(name, -1) for name in names], dtype=numpy.int64
            )

        return cls(
            names,
            flags["acceptor_hybridization"],
            flags["is_hydrogen"].astype(numpy.int32),
            families(
                hbond_database.acceptor_atom_types,
                "a",
                "acceptor_type",
                hbond_database.acceptor_type_params,
            ),
            families(
                hbond_database.donor_atom_types,
                "d",
                "donor_type",
                hbond_database.donor_type_params,
            ),
        )


def prepare_hbond_block(block_type, chemistry, indexed_bonds, bond_spans, tile_size=32):
    """Prepare arbitrary residue atoms and attached hydrogens in canonical order.

    Bond inputs are directed intra-residue pairs and per-atom half-open ranges;
    donor-heavy indices remain full residue indices across tile boundaries.
    """
    atom_type_idx = block_type_indices(chemistry.atom_names, block_type)
    atom_acceptor_hybridization = chemistry.hybridization[atom_type_idx][None, :]
    acc_type = chemistry.acceptor_types[atom_type_idx]
    don_type = chemistry.donor_types[atom_type_idx]
    is_acc, is_don = acc_type != -1, don_type != -1

    A_idx = numpy.nonzero(is_acc)[0].astype(dtype=numpy.int32)
    is_hydrogen = chemistry.is_hydrogen[atom_type_idx]

    tiled_acc_orig_inds, tile_n_acc = tile_subset_indices(
        A_idx, tile_size, True, block_type.n_atoms - 1
    )

    n_tiles = tile_n_acc.shape[0]
    tiled_acc_orig_inds = tiled_acc_orig_inds.reshape(n_tiles, tile_size)
    is_tiled_acc = tiled_acc_orig_inds != -1
    tile_acc_inds = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
    tile_acceptor_type = numpy.copy(tile_acc_inds)
    tile_acceptor_hybridization = numpy.copy(tile_acc_inds)
    tile_acc_inds[is_tiled_acc] = A_idx % tile_size
    tile_acceptor_type[is_tiled_acc] = acc_type[A_idx]
    tile_acceptor_hybridization[is_tiled_acc] = atom_acceptor_hybridization[0, A_idx]

    # now lets get the list of attached hydrogen atoms:
    D_idx = numpy.nonzero(is_don)[0].astype(dtype=numpy.int32)
    H_idx, D_for_H, which_H_for_Hs_D, n_H_for_D = attached_H_for_don(
        is_hydrogen,
        D_idx,
        indexed_bonds,
        bond_spans,
    )
    donH_type = don_type[D_for_H]
    n_H_for_at = numpy.zeros(block_type.n_atoms, dtype=numpy.int32)
    n_H_for_at[D_idx] = n_H_for_D
    tile_acceptor_n_attached_H = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
    tile_acceptor_n_attached_H[is_tiled_acc] = n_H_for_at[A_idx]

    tiled_donH_orig_inds, tile_n_donH = tile_subset_indices(
        H_idx, tile_size, True, block_type.n_atoms - 1
    )
    tiled_don_hvy_orig_inds, tile_n_don_hvy = tile_subset_indices(
        D_idx, tile_size, True, block_type.n_atoms - 1
    )

    assert tile_n_donH.shape[0] == n_tiles
    assert tile_n_don_hvy.shape[0] == n_tiles
    tiled_donH_orig_inds = tiled_donH_orig_inds.reshape((n_tiles, tile_size))
    is_tiled_donH = tiled_donH_orig_inds != -1
    tiled_don_hvy_orig_inds = tiled_don_hvy_orig_inds.reshape((n_tiles, tile_size))
    is_tiled_don_hvy = tiled_don_hvy_orig_inds != -1

    tile_donH_inds = numpy.full((n_tiles, tile_size), -1, dtype=numpy.int32)
    tile_don_hvy_inds = numpy.copy(tile_donH_inds)
    tile_donH_hvy_inds = numpy.copy(tile_donH_inds)
    tile_donorH_type = numpy.copy(tile_donH_inds)
    tile_which_donH_of_donH_hvy = numpy.copy(tile_donH_inds)

    tile_donH_inds[is_tiled_donH] = H_idx % tile_size
    tile_don_hvy_inds[is_tiled_don_hvy] = D_idx % tile_size
    # NB: store the donor heavy atom's *block-local* (global-within-block)
    # index, NOT a tile-local one. This array is tiled by the donor
    # HYDROGEN (lk_ball water generation processes each donH in its H's
    # tile), but the heavy atom may live in a different tile -- so a
    # %tile_size value would be unrecoverable. lk_ball's load_coord/write
    # paths consume this as a block-centric atom index (it handles the
    # out-of-tile fetch). hbond scoring does not use this array.
    tile_donH_hvy_inds[is_tiled_donH] = D_for_H
    tile_donorH_type[is_tiled_donH] = donH_type
    tile_which_donH_of_donH_hvy[is_tiled_donH] = which_H_for_Hs_D
    hbbt_params = HBondBlockTypeParams(
        donH_inds=H_idx,
        don_hvy_inds=D_idx,
        acc_inds=A_idx,
        tile_n_donH=tile_n_donH,
        tile_n_don_hvy=tile_n_don_hvy,
        tile_n_acc=tile_n_acc,
        tile_donH_inds=tile_donH_inds,
        tile_donH_hvy_inds=tile_donH_hvy_inds,
        tile_don_hvy_inds=tile_don_hvy_inds,
        tile_which_donH_of_donH_hvy=tile_which_donH_of_donH_hvy,
        tile_acc_inds=tile_acc_inds,
        tile_donorH_type=tile_donorH_type,
        tile_acceptor_type=tile_acceptor_type,
        tile_acceptor_hybridization=tile_acceptor_hybridization,
        tile_acceptor_n_attached_H=tile_acceptor_n_attached_H,
        is_hydrogen=is_hydrogen,
    )
    return hbbt_params


def pack_hbond_blocks(blocks, max_n_atoms=None, tile_size=32):
    """Pack all residue annotations; zero counts/flags and -1 index padding."""
    blocks = tuple(blocks)
    if max_n_atoms is None:
        max_n_atoms = max((b.is_hydrogen.size for b in blocks), default=0)
    max_n_tiles = (max_n_atoms + tile_size - 1) // tile_size
    counts = ("tile_n_donH", "tile_n_don_hvy", "tile_n_acc")
    indices = (
        "tile_donH_inds",
        "tile_donH_hvy_inds",
        "tile_don_hvy_inds",
        "tile_which_donH_of_donH_hvy",
        "tile_acc_inds",
        "tile_donorH_type",
        "tile_acceptor_type",
        "tile_acceptor_hybridization",
        "tile_acceptor_n_attached_H",
    )
    arrays = {
        name: numpy.zeros((len(blocks), max_n_tiles), dtype=numpy.int32)
        for name in counts
    }
    arrays.update(
        {
            name: numpy.full(
                (len(blocks), max_n_tiles, tile_size), -1, dtype=numpy.int32
            )
            for name in indices
        }
    )
    arrays["is_hydrogen"] = numpy.zeros((len(blocks), max_n_atoms), dtype=numpy.int32)
    for i, block in enumerate(blocks):
        for name, target in arrays.items():
            source = getattr(block, name)
            target[i, : source.shape[0]] = source
    return arrays


def native_hbond_parameters(chemical_database, hbond_database):
    """Return canonical pair, polynomial and global native ABI tables.

    Polynomial rows contain c10..c0, one alignment zero, then range and bound;
    all coefficients retain float64 precision independently of coordinates.
    """
    pair = hbond_pair_parameters(chemical_database, hbond_database)
    distances = [
        p.xmax
        for p in hbond_database.polynomial_parameters
        if p.dimension == "hbgd_AHdist"
    ]
    if not distances and pair.donor_weight.size:
        raise ValueError("HBond pair table has no AHdist polynomial range")
    fields = (
        "hb_sp2_range_span",
        "hb_sp2_BAH180_rise",
        "hb_sp2_outer_width",
        "hb_sp3_softmax_fade",
        "threshold_distance",
    )
    globals = numpy.asarray(
        [
            [getattr(hbond_database.global_parameters, name) for name in fields]
            + [max(distances, default=0.0)]
        ],
        dtype=numpy.float32,
    )
    parameters = numpy.stack(
        (
            pair.acceptor_hybridization.astype(numpy.float32),
            pair.acceptor_weight,
            pair.donor_weight,
        ),
        axis=-1,
    )
    polynomials = numpy.concatenate(
        [
            numpy.concatenate(
                (
                    values[..., 4:],
                    numpy.zeros(values.shape[:-1] + (1,), dtype=numpy.float64),
                    values[..., :4],
                ),
                axis=-1,
            )
            for values in pair.polynomials.values()
        ],
        axis=-1,
    )
    return dict(
        global_param_table=globals,
        pair_param_table=parameters,
        pair_poly_table=polynomials,
    )
