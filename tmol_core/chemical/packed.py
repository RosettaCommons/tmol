"""Framework-neutral packed topology for any ordered refined residue palette.

Arrays retain physical atom order and mature torsion/connection conventions.
Backends choose device transfer and connection capacity. Additional atom typing,
ideal geometry and path distances are lazy host data, so consumers pay only for
metadata they need. No parameter-dependent score annotations are cached here.
"""

from dataclasses import dataclass
from functools import cached_property

import numpy

from .constants import MAX_PATHS_FROM_CONNECTION


@dataclass
class PackedBlockTypesData:
    chem_db: object
    active_block_types: object
    max_n_atoms: int
    n_atoms: numpy.ndarray
    atom_is_real: numpy.ndarray
    atom_is_hydrogen: numpy.ndarray
    atom_downstream_of_conn: numpy.ndarray
    atom_paths_from_conn: numpy.ndarray
    max_n_torsions: int
    n_torsions: numpy.ndarray
    torsion_is_real: numpy.ndarray
    torsion_uaids: numpy.ndarray
    is_torsion_mc: numpy.ndarray
    n_mc_torsions: numpy.ndarray
    mc_torsion_is_real: numpy.ndarray
    mc_torsions: numpy.ndarray
    n_sc_torsions: numpy.ndarray
    sc_torsion_is_real: numpy.ndarray
    sc_torsions: numpy.ndarray
    which_mcsc_torsions: numpy.ndarray
    max_n_bonds: int
    n_bonds: numpy.ndarray
    bond_is_real: numpy.ndarray
    bond_indices: numpy.ndarray
    max_n_conn: int
    n_conn: numpy.ndarray
    conn_is_real: numpy.ndarray
    conn_atom: numpy.ndarray
    down_conn_inds: numpy.ndarray
    up_conn_inds: numpy.ndarray
    polymeric_conn_inds: numpy.ndarray
    default_jump_connection_atom_inds: numpy.ndarray

    @property
    def n_types(self):
        return len(self.active_block_types)

    @classmethod
    def from_restypes(cls, chem_db, active_types, *, min_n_conn=0):
        active_block_types = tuple(active_types)
        max_n_atoms = cls.count_max_n_atoms(active_block_types)
        n_atoms = cls.count_n_atoms(active_block_types)
        atom_is_real = cls.determine_real_atoms(max_n_atoms, n_atoms)
        atom_is_hydrogen = cls.determine_h_atoms(
            chem_db, max_n_atoms, n_atoms, active_block_types
        )
        atom_downstream_of_conn = cls.join_atom_downstream_of_conn(active_block_types)
        atom_paths_from_conn = cls.join_atom_paths_from_conn(active_block_types)
        n_torsions, torsion_is_real, torsion_uaids = cls.join_torsion_uaids(
            active_block_types
        )
        is_torsion_mc = cls.join_is_torsion_mcs(active_block_types)
        n_mc_torsions, mc_torsion_is_real, mc_torsions = cls.join_mc_torsion_inds(
            active_block_types
        )
        n_sc_torsions, sc_torsion_is_real, sc_torsions = cls.join_sc_torsion_inds(
            active_block_types
        )
        which_mcsc_torsions = cls.join_mcsc_torsion_inds(active_block_types)
        n_bonds, bond_is_real, bond_indices = cls.join_bond_indices(active_block_types)
        n_conn, conn_is_real, conn_atom = cls.join_conn_indices(active_block_types)
        down_conn_inds, up_conn_inds = cls.join_polymeric_connections(
            active_block_types
        )
        def_jumpconn_inds = cls.join_default_jump_connection_atom_inds(
            active_block_types
        )

        if min_n_conn > conn_atom.shape[1]:
            extra = min_n_conn - conn_atom.shape[1]
            conn_atom = numpy.pad(conn_atom, ((0, 0), (0, extra)), constant_values=-1)
            conn_is_real = numpy.pad(conn_is_real, ((0, 0), (0, extra)))
            atom_downstream_of_conn = numpy.pad(
                atom_downstream_of_conn,
                ((0, 0), (0, extra), (0, 0)),
                constant_values=-1,
            )
            atom_paths_from_conn = numpy.pad(
                atom_paths_from_conn,
                ((0, 0), (0, extra), (0, 0), (0, 0)),
                constant_values=-1,
            )
        return cls(
            chem_db=chem_db,
            active_block_types=active_block_types,
            max_n_atoms=max_n_atoms,
            n_atoms=n_atoms,
            atom_is_real=atom_is_real,
            atom_is_hydrogen=atom_is_hydrogen,
            atom_downstream_of_conn=atom_downstream_of_conn,
            atom_paths_from_conn=atom_paths_from_conn,
            max_n_torsions=torsion_is_real.shape[1],
            n_torsions=n_torsions,
            torsion_is_real=torsion_is_real,
            torsion_uaids=torsion_uaids,
            is_torsion_mc=is_torsion_mc,
            n_mc_torsions=n_mc_torsions,
            mc_torsion_is_real=mc_torsion_is_real,
            mc_torsions=mc_torsions,
            n_sc_torsions=n_sc_torsions,
            sc_torsion_is_real=sc_torsion_is_real,
            sc_torsions=sc_torsions,
            which_mcsc_torsions=which_mcsc_torsions,
            max_n_bonds=bond_is_real.shape[1],
            n_bonds=n_bonds,
            bond_is_real=bond_is_real,
            bond_indices=bond_indices,
            max_n_conn=conn_is_real.shape[1],
            n_conn=n_conn,
            conn_is_real=conn_is_real,
            conn_atom=conn_atom,
            down_conn_inds=down_conn_inds,
            up_conn_inds=up_conn_inds,
            polymeric_conn_inds=numpy.stack([down_conn_inds, up_conn_inds], axis=1),
            default_jump_connection_atom_inds=def_jumpconn_inds,
        )

    @classmethod
    def count_max_n_atoms(cls, active_block_types):
        return max((len(restype.atoms) for restype in active_block_types), default=0)

    @classmethod
    def count_n_atoms(cls, active_block_types):
        return numpy.array(
            [len(restype.atoms) for restype in active_block_types],
            dtype=numpy.int32,
        )

    @classmethod
    def determine_real_atoms(cls, max_n_atoms, n_atoms):
        return numpy.arange(max_n_atoms)[None, :] < n_atoms[:, None]

    @classmethod
    def determine_h_atoms(
        cls,
        chem_db,
        max_n_atoms,
        n_atoms,
        active_block_types,
    ):
        atype_element = {atype.name: atype.element for atype in chem_db.atom_types}
        atom_is_hydrogen = [
            [int(atype_element[atom.atom_type] == "H") for atom in bt.atoms]
            + [0] * (max_n_atoms - len(bt.atoms))
            for bt in active_block_types
        ]
        return numpy.array(atom_is_hydrogen, dtype=numpy.int32).reshape(
            len(active_block_types), max_n_atoms
        )

    @classmethod
    def join_atom_downstream_of_conn(cls, active_block_types):
        n_restypes = len(active_block_types)
        max_n_conn = max((len(rt.connections) for rt in active_block_types), default=0)
        max_n_atoms = max((len(rt.atoms) for rt in active_block_types), default=0)
        atom_downstream_of_conn = numpy.full(
            (n_restypes, max_n_conn, max_n_atoms), -1, dtype=numpy.int32
        )
        for i, rt in enumerate(active_block_types):
            rt_adoc = rt.atom_downstream_of_conn
            atom_downstream_of_conn[i, : rt_adoc.shape[0], : rt_adoc.shape[1]] = rt_adoc
        return atom_downstream_of_conn

    @classmethod
    def join_atom_paths_from_conn(clas, active_block_types):
        n_restypes = len(active_block_types)
        max_n_conn = max((len(bt.connections) for bt in active_block_types), default=0)

        atom_paths_from_conn = numpy.full(
            (
                n_restypes,
                max_n_conn,
                MAX_PATHS_FROM_CONNECTION,
                3,
            ),
            -1,
            dtype=numpy.int32,
        )

        for i, bt in enumerate(active_block_types):
            paths = bt.atom_paths_from_conn
            atom_paths_from_conn[i, : len(bt.connections)] = paths

        return atom_paths_from_conn

    @classmethod
    def _join_block_type_attribute(
        cls,
        active_block_types,
        attribute_name,
        dtype=None,
        sentinel=-1,
    ):
        arrays = [
            numpy.asarray(getattr(bt, attribute_name), dtype=dtype)
            for bt in active_block_types
        ]
        if not arrays:
            tails = {"ordered_torsions": (4, 3), "bond_indices": (2,)}
            return (
                numpy.zeros(0, dtype=numpy.int32),
                numpy.zeros((0, 0), dtype=bool),
                numpy.empty(
                    (0, 0, *tails.get(attribute_name, ())), dtype=dtype or numpy.int32
                ),
            )
        n_elements = numpy.asarray(
            [array.shape[0] for array in arrays], dtype=numpy.int32
        )
        joined = numpy.full(
            (len(arrays), int(n_elements.max()), *arrays[0].shape[1:]),
            sentinel,
            dtype=arrays[0].dtype,
        )
        for i, array in enumerate(arrays):
            joined[i, : array.shape[0]] = array

        real = numpy.arange(joined.shape[1])[None, :] < n_elements[:, None]
        return n_elements, real, joined

    @classmethod
    def join_torsion_uaids(cls, active_block_types):
        # unresolved atom ids are three integers:
        #  1st integer: an atom index, -1 if the atom is unresolved
        #  2nd integer: the connection index for this block that the unresolved atom
        #               is on ther other side of, -1 if
        #  3rd integer: the number of chemical bonds into the other block that the
        #               unresolved atom is found.
        return cls._join_block_type_attribute(active_block_types, "ordered_torsions")

    @classmethod
    def join_is_torsion_mcs(cls, active_block_types):
        return cls._join_block_type_attribute(
            active_block_types,
            "is_torsion_mc",
            dtype=bool,
            sentinel=False,
        )[2]

    @classmethod
    def join_mc_torsion_inds(cls, active_block_types):
        return cls._join_block_type_attribute(active_block_types, "mc_torsions")

    @classmethod
    def join_sc_torsion_inds(cls, active_block_types):
        return cls._join_block_type_attribute(active_block_types, "sc_torsions")

    @classmethod
    def join_mcsc_torsion_inds(cls, active_block_types):
        # only return the joined tensor of mcsc indices, index 2 of the tuple
        # returned by join_tensors_and_report_real_entries
        return cls._join_block_type_attribute(active_block_types, "which_mcsc_torsion")[
            2
        ]

    @classmethod
    def join_bond_indices(cls, active_block_types):
        return cls._join_block_type_attribute(
            active_block_types, "bond_indices", dtype=numpy.int32
        )

    @classmethod
    def join_conn_indices(cls, active_block_types):
        return cls._join_block_type_attribute(
            active_block_types,
            "ordered_connection_atoms",
            dtype=numpy.int32,
        )

    @classmethod
    def join_polymeric_connections(cls, active_block_types):
        down_conn_inds = numpy.array(
            [bt.down_connection_ind for bt in active_block_types],
            dtype=numpy.int32,
        )
        up_conn_inds = numpy.array(
            [bt.up_connection_ind for bt in active_block_types],
            dtype=numpy.int32,
        )
        return down_conn_inds, up_conn_inds

    @classmethod
    def join_default_jump_connection_atom_inds(cls, active_block_types):
        return numpy.array(
            [bt.default_jump_connection_atom_index for bt in active_block_types],
            dtype=numpy.int32,
        )

    @cached_property
    def atom_names(self):
        return tuple(
            tuple(a.name for a in bt.atoms) + ("",) * (self.max_n_atoms - bt.n_atoms)
            for bt in self.active_block_types
        )

    @cached_property
    def atom_types(self):
        indices = {at.name: i for i, at in enumerate(self.chem_db.atom_types)}
        result = numpy.zeros((self.n_types, self.max_n_atoms), dtype=numpy.int32)
        for i, bt in enumerate(self.active_block_types):
            result[i, : bt.n_atoms] = [indices[a.atom_type] for a in bt.atoms]
        return result

    @cached_property
    def atom_elements(self):
        elements = {el.name: i for i, el in enumerate(self.chem_db.element_types)}
        types = {at.name: at.element for at in self.chem_db.atom_types}
        result = numpy.zeros((self.n_types, self.max_n_atoms), dtype=numpy.int32)
        for i, bt in enumerate(self.active_block_types):
            result[i, : bt.n_atoms] = [
                elements.get(types[a.atom_type], -1) for a in bt.atoms
            ]
        return result

    @cached_property
    def atom_is_backbone(self):
        result = numpy.zeros((self.n_types, self.max_n_atoms), dtype=bool)
        for i, bt in enumerate(self.active_block_types):
            mainchain = set(bt.properties.polymer.mainchain_atoms or ())
            result[i, : bt.n_atoms] = [a.name in mainchain for a in bt.atoms]
        return result

    @cached_property
    def ideal_coords(self):
        result = numpy.zeros((self.n_types, self.max_n_atoms, 3), dtype=numpy.float32)
        for i, bt in enumerate(self.active_block_types):
            result[i, : bt.n_atoms] = bt.ideal_coords[bt.at_to_icoor_ind]
        return result

    @cached_property
    def intrares_path_distance(self):
        from .bonds import pack_bond_separations

        return pack_bond_separations(
            self.active_block_types, self.max_n_atoms, padding=0
        )

    @cached_property
    def bond_topology(self):
        from .bonds import pack_bond_topology

        return pack_bond_topology(self.active_block_types, self.max_n_atoms)
