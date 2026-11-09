"""Residue-specific, wildcard and cross-connection atom naming for native terms."""

from dataclasses import dataclass

import numpy as np


class AtomNames:
    @staticmethod
    def get_atom_unique_id_name(block_name, atom_name):
        return "UNIQUE_ID:" + block_name + ":" + atom_name

    @staticmethod
    def get_atom_wildcard_id_name(atom_name):
        return "WILDCARD_ID:" + atom_name

    @staticmethod
    def get_atom_cross_id_name(atom_name):
        return "CROSS_ID:" + atom_name

    def _create_uniq_and_wildcard_names_for_bt(self, block_type):
        names = [atom.name for atom in block_type.atoms]
        return (
            [
                self.get_atom_unique_id_name(block_type.base_name, name)
                for name in names
            ],
            [self.get_atom_wildcard_id_name(name) for name in names],
            [self.get_atom_cross_id_name(name) for name in names],
        )


@dataclass
class AtomIdData:
    atom_unique_ids: np.ndarray
    atom_wildcard_ids: np.ndarray
    atom_cross_ids: np.ndarray
    atom_unique_id_index: dict


def packed_atom_ids(block_types, max_n_atoms=None):
    block_types = tuple(block_types)
    if max_n_atoms is None:
        max_n_atoms = max((bt.n_atoms for bt in block_types), default=0)
    tables = [
        np.full((len(block_types), max_n_atoms), -1, dtype=np.int32) for _ in range(3)
    ]
    index = {}
    naming = AtomNames()
    for i, bt in enumerate(block_types):
        for table, names in zip(
            tables, naming._create_uniq_and_wildcard_names_for_bt(bt)
        ):
            for j, name in enumerate(names):
                table[i, j] = index.setdefault(name, len(index))
    return AtomIdData(*tables, index)
