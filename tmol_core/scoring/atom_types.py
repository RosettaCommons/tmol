"""Chemical atom flags and real-atom index validation, independent of execution."""

from enum import IntEnum

import numpy as np

HYBRIDIZATION_NAMES = (None, "sp2", "sp3", "ring")


class AcceptorHybridization(IntEnum):
    none = 0
    sp2 = 1
    sp3 = 2
    ring = 3


def atom_type_parameters(atom_types):
    """Return ordered names and flags, with one explicit inert padding row."""
    names = tuple(atom.name for atom in atom_types) + (None,)
    if len(set(names)) != len(names):
        raise ValueError("Chemical atom type names must be unique and non-null")
    fields = ("is_acceptor", "is_donor", "is_hydroxyl", "is_polarh")
    values = {
        field: np.asarray(
            [
                (
                    bool(value)
                    if (value := getattr(atom, field)) is not None and value == value
                    else False
                )
                for atom in atom_types
            ]
            + [False]
        )
        for field in fields
    }
    values["is_hydrogen"] = np.asarray(
        [atom.element == "H" for atom in atom_types] + [False]
    )
    values["acceptor_hybridization"] = np.asarray(
        [HYBRIDIZATION_NAMES.index(atom.acceptor_hybridization) for atom in atom_types]
        + [0],
        dtype=np.int32,
    )
    return names, values


def block_type_indices(names, block_type):
    """Require every physical atom to resolve; None is reserved for padding."""
    lookup = {name: i for i, name in enumerate(names) if name is not None}
    indices = np.asarray(
        [lookup.get(atom.atom_type, -1) for atom in block_type.atoms], dtype=np.int64
    )
    invalid = np.flatnonzero(indices < 0)
    if invalid.size:
        atoms = ", ".join(
            f"{block_type.atoms[i].name} ({block_type.atoms[i].atom_type!r})"
            for i in invalid
        )
        raise ValueError(
            f"Residue {block_type.name} has unregistered chemical atom types: {atoms}"
        )
    return indices
