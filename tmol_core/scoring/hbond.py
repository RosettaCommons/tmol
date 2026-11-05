"""Complete validated hydrogen-bond family tables, without framework tensors.

Polynomial layout: xmin, xmax, min_val, max_val, then c10 through c0.
The canonical coefficients stay float64 through native table preparation.
"""

from dataclasses import dataclass

import numpy

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
