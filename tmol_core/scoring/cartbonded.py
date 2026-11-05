"""Complete Cartesian-bonded chemistry and parameter preparation for native kernels.

Host arrays preserve mature atom namespaces, exact variant overrides, scoped
reference aliases, explicit connection-owned parameters and improper ownership.
Only a frontend owns annotation lifetime and device transfer.
"""

import logging
from itertools import permutations, product
from types import SimpleNamespace

import attrs
import numpy

from .atom_ids import AtomNames, packed_atom_ids
from .hash_util import _make_hashtable_keys_values, _add_to_hashtable
from .cartbonded_connections import _compile_connection_parameters

logger = logging.getLogger(__name__)
CROSS_RES_PREFIX = "+"


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class CartBondedBlockAnnotations:
    cartbonded_subgraphs: numpy.ndarray
    cartbonded_subgraph_type_counts: numpy.ndarray
    cartbonded_subgraph_type_offsets: numpy.ndarray
    cartbonded_params: dict


@attrs.define(auto_attribs=True, slots=True, frozen=True)
class CartBondedPackedData:
    cartbonded_subgraphs: numpy.ndarray
    cartbonded_subgraph_offsets: numpy.ndarray
    cartbonded_subgraph_type_counts: numpy.ndarray
    cartbonded_subgraph_type_offsets: numpy.ndarray
    cartbonded_subgraph_param_indices: numpy.ndarray
    cartbonded_max_subgraphs_per_block: int
    cartbonded_atom_unique_id_index: dict
    atom_unique_ids: numpy.ndarray
    atom_wildcard_ids: numpy.ndarray
    atom_cross_ids: numpy.ndarray
    cartbonded_params_hash_keys: numpy.ndarray
    cartbonded_params_hash_values: numpy.ndarray
    connection_hash_keys: numpy.ndarray
    connection_spans: numpy.ndarray
    connection_paths: numpy.ndarray
    atom_is_rosetta: numpy.ndarray
    rosetta_typed: frozenset


class CartBondedPreparation(AtomNames):
    def __init__(self, cart_database, rosetta_typed):
        self.cart_database = cart_database
        self.rosetta_typed = frozenset(rosetta_typed)
        self.improper_roots = {
            imp.atm3.lstrip(CROSS_RES_PREFIX)
            for params in cart_database.residue_params.values()
            for imp in params.improper_parameters
        }
        self.hash = cart_database.hash
        self._params_for_res_cache = {}

    def find_subgraphs(self, bonds, block_type):  # noqa: C901
        lengths = []
        angles = []
        torsions = []
        improper = []

        # create a convenient datastructure for following connections
        bondmap = {}
        for bond in bonds:
            if bond[0] not in bondmap:
                bondmap[bond[0]] = set()
            bondmap[bond[0]].add(bond[1])

        # get lengths
        for atom1 in bondmap:
            for atom2 in bondmap[atom1]:
                if atom1 < atom2:
                    lengths.append((atom1, atom2, -1, -1))

        # get angles
        for atom1 in bondmap:
            for atom2 in bondmap[atom1]:
                for atom3 in bondmap[atom2]:
                    if atom1 >= atom3:
                        continue
                    angles.append((atom1, atom2, atom3, -1))

        # get torsions
        for atom1 in bondmap:
            for atom2 in bondmap[atom1]:
                for atom3 in bondmap[atom2]:
                    if atom1 == atom3:
                        continue
                    for atom4 in bondmap[atom3]:
                        if atom2 == atom4:
                            continue
                        if atom1 >= atom4:
                            continue
                        torsions.append((atom1, atom2, atom3, atom4))

        # get improper torsions
        for improper_root in self.improper_roots:
            if improper_root in block_type.atom_to_idx:
                atom3 = block_type.atom_to_idx[improper_root]
                for atom1, atom2, atom4 in permutations(bondmap.get(atom3, ()), 3):
                    improper.append((atom1, atom2, atom3, atom4))

        return (
            lengths,
            angles,
            torsions,
            improper,
        )

    def get_raw_params_for_res(self, res: str):
        if res in self.cart_database.residue_params:
            return (
                self.cart_database.residue_params[res].length_parameters
                + self.cart_database.residue_params[res].angle_parameters
                + self.cart_database.residue_params[res].torsion_parameters
                + self.cart_database.residue_params[res].improper_parameters
                + self.cart_database.residue_params[res].hxltorsion_parameters
            )
        return []

    def get_formatted_atoms_and_params(self, raw_params):
        fields = ["atm1", "atm2", "atm3", "atm4"]
        atoms = [
            getattr(raw_params, field) for field in fields if hasattr(raw_params, field)
        ]
        fields = ["type", "x0", "K", "k1", "k2", "k3", "phi1", "phi2", "phi3"]
        params = [
            getattr(raw_params, field) for field in fields if hasattr(raw_params, field)
        ]
        return atoms, params

    def get_params_for_res(self, res: str):
        cached = self._params_for_res_cache.get(res)
        if cached is not None:
            return cached

        params_by_atom_unique_id = {}

        # Fetch the raw params from the DB
        all_params = self.get_raw_params_for_res(res)

        # res "wildcard" matches by atom name in any residue type; a leading '+'
        # marks an atom across a residue connection. The two axes are independent.
        is_wildcard = res == "wildcard"

        for param in all_params:
            # Format the raw param
            atoms, params = self.get_formatted_atoms_and_params(param)

            for i, atom in enumerate(atoms):
                if atom.startswith(CROSS_RES_PREFIX):
                    atoms[i] = self.get_atom_cross_id_name(
                        atom[len(CROSS_RES_PREFIX) :]
                    )
                elif is_wildcard:
                    atoms[i] = self.get_atom_wildcard_id_name(atom)
                else:
                    atoms[i] = self.get_atom_unique_id_name(res, atom)

            key = tuple(atoms)

            params_by_atom_unique_id[key] = params

        self._params_for_res_cache[res] = params_by_atom_unique_id
        return params_by_atom_unique_id

    _warned_unparameterized = set()

    @classmethod
    def _warn_unparameterized(cls, block_type, lengths):
        """Warn once per type whose own bonds have no rows, as HIS_POS had; HOH (the
        readers drop water) and bonds to virtual atoms are exempt."""
        virtual = {
            i
            for i, atom in enumerate(block_type.atoms)
            if atom.atom_type == "Vrt" or atom.name in block_type.properties.virtual
        }
        if block_type.base_name in cls._warned_unparameterized | {"HOH"} or all(
            {a, b} & virtual for a, b, _, _ in lengths
        ):
            return
        cls._warned_unparameterized.add(block_type.base_name)
        logger.warning(
            "cart_bonded has no parameters for %s or its variants (first seen: %s): "
            "their bond lengths and angles are unrestrained",
            block_type.base_name,
            block_type.name,
        )

    def _parameter_name(self, block_type):
        if block_type.is_ligand_fragment:
            records = self.cart_database.residue_params
            # Fragment preparations carry the source ligand's complete
            # CartRes. Equal records must share its namespace so paths across
            # a cut still resolve to the original ligand parameters. This
            # also covers separately deserialized copies of those records.
            if records.get(block_type.name) == records.get(block_type.base_name):
                return block_type.base_name
        return (
            block_type.name
            if block_type.name in self.cart_database.residue_params
            else block_type.base_name
        )

    def _cartbonded_atom_ids(self, packed_block_types, id_index):
        """Reuse ordinary IDs unless an exact variant needs its own namespace."""
        ids = packed_block_types.atom_unique_ids
        overrides = [
            (i, bt)
            for i, bt in enumerate(packed_block_types.active_block_types)
            if self._parameter_name(bt) != bt.base_name
        ]
        if not overrides:
            return ids
        array = ids.copy()
        for i, bt in overrides:
            for j, atom in enumerate(bt.atoms):
                name = self.get_atom_unique_id_name(bt.name, atom.name)
                array[i, j] = id_index.setdefault(name, len(id_index))
        return array

    @staticmethod
    def _padded_param_key(key):
        padded = [-1, -1, -1, -1]
        padded[: len(key)] = key
        return tuple(padded)

    @staticmethod
    def _lookup_param_index(key, param_key_to_index):
        return param_key_to_index.get(CartBondedPreparation._padded_param_key(key), -1)

    def _precompute_subgraph_param_indices(
        self,
        packed_block_types,
        subgraph_offsets,
        total_subgraphs,
        param_key_to_index,
        atom_unique_ids,
        atom_wildcard_ids,
        block_annotations,
    ):
        """Resolve invariant intra-block parameter searches once during setup."""
        # Scoped reference IDs, not the block types' own: param_key_to_index is
        # keyed on the scoped IDs and the kernel is handed the same table.
        param_indices = numpy.full(total_subgraphs, -1, dtype=numpy.int32)

        for block_type_index, block_params in enumerate(block_annotations):
            block_offset = subgraph_offsets[block_type_index]
            unique_ids = atom_unique_ids[block_type_index]
            wildcard_ids = atom_wildcard_ids[block_type_index]
            subgraphs = numpy.asarray(
                block_params.cartbonded_subgraphs, dtype=numpy.int32
            )
            for local_index, subgraph in enumerate(subgraphs):
                n_atoms = 4
                while n_atoms > 0 and subgraph[n_atoms - 1] == -1:
                    n_atoms -= 1
                if n_atoms == 0:
                    continue
                found = -1
                for atom_ids in (unique_ids, wildcard_ids):
                    found = self._lookup_param_index(
                        [int(atom_ids[subgraph[i]]) for i in range(n_atoms)],
                        param_key_to_index,
                    )
                    if found != -1:
                        break
                    found = self._lookup_param_index(
                        [
                            int(atom_ids[subgraph[n_atoms - 1 - i]])
                            for i in range(n_atoms)
                        ],
                        param_key_to_index,
                    )
                    if found != -1:
                        break
                param_indices[block_offset + local_index] = found
        return param_indices

    def _reference_params(self, pbt, atom_ids, params):
        """Compile name fallbacks once, scoped to the actual block type.

        Keep native lookup unchanged: renamed atoms use private IDs with
        exact-name rows first and borrowed chemical-role rows second.
        Unrelated blocks never acquire that alias; ordinary types reuse their
        existing IDs and allocate no additional table.
        """
        changed = [
            (i, bt)
            for i, bt in enumerate(pbt.active_block_types)
            if any(atom.cartbonded_reference for atom in bt.atoms)
        ]
        originals = (pbt.atom_wildcard_ids, pbt.atom_cross_ids)
        if not changed:
            return originals
        tables = []
        for prefix, original in zip(("WILDCARD_ID:", "CROSS_ID:"), originals):
            rows = [
                (key, value)
                for key, value in params.items()
                if any(atom.startswith(prefix) for atom in key)
            ]
            ids = original.copy()
            for i, bt in changed:
                exact, borrowed = {}, {}
                for j, atom in enumerate(bt.atoms):
                    name = prefix + atom.name
                    if atom.cartbonded_reference:
                        scoped = prefix + f"{bt.name}:{atom.name}"
                        ids[i, j] = atom_ids.setdefault(scoped, len(atom_ids))
                    else:
                        scoped = name
                    exact[name] = (scoped,)
                    reference = prefix + (atom.cartbonded_reference or atom.name)
                    borrowed.setdefault(reference, []).append(scoped)
                # Exact supplied rows win regardless of parameter ordering.
                for names in (exact, borrowed):
                    for key, value in rows:
                        choices = [
                            names.get(atom, ()) if atom.startswith(prefix) else (atom,)
                            for atom in key
                        ]
                        for mapped in product(*choices):
                            params.setdefault(mapped, value)
            tables.append(ids)
        return tuple(tables)

    def prepare_block(self, block_type):
        # Get the subgraphs for this block type
        lengths, angles, torsions, improper = self.find_subgraphs(
            block_type.bond_indices, block_type
        )
        cart_subgraphs = numpy.asarray(
            lengths + angles + torsions + improper, dtype=numpy.int32
        ).reshape(-1, 4)
        cart_subgraph_type_counts = numpy.array(
            [len(lengths), len(angles), len(torsions) + len(improper)]
        )
        cart_subgraph_type_offsets = numpy.array(
            [
                0,
                cart_subgraph_type_counts[0],
                cart_subgraph_type_counts[0] + cart_subgraph_type_counts[1],
            ]
        )

        # Fetch the params from the database, updating the atom id store if necessary
        # An exact patched name replaces this residue's complete CartRes.
        # Keep its atom namespace separate so the unpatched type, other
        # variants and other score terms can retain their existing parameters.
        parameter_name = self._parameter_name(block_type)
        cartbonded_params = self.get_params_for_res(parameter_name)
        if parameter_name not in self.cart_database.residue_params:
            self._warn_unparameterized(block_type, lengths)
        cb_block_ann = CartBondedBlockAnnotations(
            cartbonded_subgraphs=cart_subgraphs,
            cartbonded_subgraph_type_counts=cart_subgraph_type_counts,
            cartbonded_subgraph_type_offsets=cart_subgraph_type_offsets,
            cartbonded_params=cartbonded_params,
        )
        return cb_block_ann

    def prepare_packed(self, block_types, block_annotations=None, *, atom_ids=None):
        block_types = tuple(block_types)
        ids = atom_ids if atom_ids is not None else packed_atom_ids(block_types)
        packed_block_types = SimpleNamespace(
            active_block_types=block_types, **vars(ids)
        )
        if block_annotations is None:
            block_annotations = [self.prepare_block(bt) for bt in block_types]
        # Aggregate the subgraphs and collect metadata
        total_subgraphs = sum(
            annotation.cartbonded_subgraphs.shape[0] for annotation in block_annotations
        )
        subgraphs = numpy.full((total_subgraphs, 4), -1, dtype=numpy.int32)
        subgraph_offsets = []
        subgraph_type_counts = []
        subgraph_type_offsets = []
        offset = 0
        max_subgraphs_per_block = 0
        for bt_params in block_annotations:
            subgraph_offsets.append(offset)
            n_subgraphs = bt_params.cartbonded_subgraphs.shape[0]
            subgraph_type_counts.append(bt_params.cartbonded_subgraph_type_counts)
            subgraph_type_offsets.append(bt_params.cartbonded_subgraph_type_offsets)
            subgraphs[offset : offset + n_subgraphs] = bt_params.cartbonded_subgraphs
            offset += n_subgraphs

            max_subgraphs_per_block = max(
                max_subgraphs_per_block, offset - subgraph_offsets[-1]
            )
        subgraph_offsets = numpy.asarray(subgraph_offsets, dtype=numpy.int32)
        subgraph_type_counts = numpy.asarray(subgraph_type_counts, dtype=numpy.int32)
        subgraph_type_offsets = numpy.asarray(subgraph_type_offsets, dtype=numpy.int32)

        # Aggregate the params
        # we will be adding new "wildcard" atom ids, so we will copy the
        # atom_unique_id_index annotation
        cbet_atom_unique_id_index = packed_block_types.atom_unique_id_index.copy()
        atom_unique_ids = self._cartbonded_atom_ids(
            packed_block_types, cbet_atom_unique_id_index
        )

        # get the params not associated with any specific residue
        wildcard_params = self.get_params_for_res("wildcard").items()

        # we have perhaps created new atom ids for this "wildcard" residue name,
        # so we must expand our set of unique atom ids.
        for key, _ in wildcard_params:
            for at in key:
                if at not in cbet_atom_unique_id_index:
                    cbet_atom_unique_id_index[at] = len(cbet_atom_unique_id_index)

        for bt_params in block_annotations:
            for key in bt_params.cartbonded_params:
                for at in key:
                    if at not in cbet_atom_unique_id_index:
                        cbet_atom_unique_id_index[at] = len(cbet_atom_unique_id_index)

        # Collect each parameter key once. Residue variants with the same base
        # name share parameter dictionaries, and hash lookup uses the first
        # inserted value for duplicate keys.
        named_params = {}
        for annotation in block_annotations:
            for key_w_str, value in annotation.cartbonded_params.items():
                named_params.setdefault(key_w_str, value)

        for key_w_str, value in wildcard_params:
            named_params.setdefault(key_w_str, value)
        atom_wildcard_ids, atom_cross_ids = self._reference_params(
            packed_block_types, cbet_atom_unique_id_index, named_params
        )
        padded_key = self._padded_param_key
        hash_keys, hash_values = _make_hashtable_keys_values(
            max(len(named_params), 1), 2, 5, 7
        )
        param_key_to_index = {}
        for cur_val, (key_w_str, value) in enumerate(named_params.items()):
            key = tuple(cbet_atom_unique_id_index[at] for at in key_w_str)
            _add_to_hashtable(hash_keys, hash_values, cur_val, key, value)
            param_key_to_index[padded_key(key)] = cur_val

        # Intra-block topology and atom naming are fixed for a packed block
        # type. Resolve the exact/reversed/wildcard parameter search once here
        # instead of repeating four hash probes in every scoring invocation.
        subgraph_param_indices = self._precompute_subgraph_param_indices(
            packed_block_types,
            subgraph_offsets,
            total_subgraphs,
            param_key_to_index,
            atom_unique_ids,
            atom_wildcard_ids,
            block_annotations,
        )

        connection_keys, connection_spans, connection_paths, connection_values = (
            _compile_connection_parameters(
                self.cart_database.connection_params,
                packed_block_types,
                len(hash_values),
            )
        )
        if len(connection_values):
            hash_values = numpy.concatenate((hash_values, connection_values), axis=0)
        return CartBondedPackedData(
            cartbonded_subgraphs=subgraphs,
            cartbonded_subgraph_offsets=subgraph_offsets,
            cartbonded_subgraph_type_counts=subgraph_type_counts,
            cartbonded_subgraph_type_offsets=subgraph_type_offsets,
            cartbonded_subgraph_param_indices=subgraph_param_indices,
            cartbonded_max_subgraphs_per_block=max_subgraphs_per_block,
            cartbonded_atom_unique_id_index=cbet_atom_unique_id_index,
            atom_unique_ids=atom_unique_ids,
            atom_wildcard_ids=atom_wildcard_ids,
            atom_cross_ids=atom_cross_ids,
            cartbonded_params_hash_keys=hash_keys,
            cartbonded_params_hash_values=hash_values,
            connection_hash_keys=connection_keys,
            connection_spans=connection_spans,
            connection_paths=connection_paths,
            atom_is_rosetta=self.ownership_mask(block_types),
            rosetta_typed=self.rosetta_typed,
        )

    def ownership_mask(self, block_types):
        max_atoms = max((bt.n_atoms for bt in block_types), default=1)
        mask = numpy.zeros((len(block_types), max(max_atoms, 1)), dtype=numpy.int32)
        for i, bt in enumerate(block_types):
            for j, atom in enumerate(bt.atoms):
                mask[i, j] = atom.atom_type in self.rosetta_typed
        return mask
