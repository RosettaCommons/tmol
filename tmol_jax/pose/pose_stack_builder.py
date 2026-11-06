"""Pack protein coordinates and construct exact connection-graph distances."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.chemical.constants import MAX_SIG_BOND_SEPARATION
from .pdb_info import PDBInfo, DEFAULT_ATOM_OCCUPANCY, DEFAULT_ATOM_B_FACTOR
from .pose_stack import PoseStack


def _connect(connections, pose, left, right, left_conn, right_conn):
    if left_conn < 0 or right_conn < 0:
        raise ValueError("requested bond has no matching residue connection")
    if np.any(connections[pose, left, left_conn] >= 0) or np.any(
        connections[pose, right, right_conn] >= 0
    ):
        raise ValueError("a residue connection cannot carry two bonds")
    connections[pose, left, left_conn] = right, right_conn
    connections[pose, right, right_conn] = left, left_conn


def _connections(pbt, block_types, chains, disulfides, explicit=None):
    poses, blocks = block_types.shape
    connections = np.full((poses, blocks, pbt.max_n_conn, 2), -1, np.int32)
    for pose, row in enumerate(block_types):
        for left in range(len(chains[pose]) - 1) if explicit is None else ():
            right = left + 1
            if (
                row[left] < 0
                or row[right] < 0
                or chains[pose][left] != chains[pose][right]
            ):
                continue
            left_type, right_type = (
                pbt.active_block_types[row[i]] for i in (left, right)
            )
            up, down = left_type.up_connection_ind, right_type.down_connection_ind
            if up >= 0 and down >= 0:
                _connect(connections, pose, left, right, up, down)
        for left, right in disulfides[pose]:
            if not (0 <= left < blocks and 0 <= right < blocks) or left == right:
                raise ValueError("invalid disulfide residue pair")
            if row[left] < 0 or row[right] < 0:
                raise ValueError("disulfide endpoints cannot be padding")
            endpoints = [
                pbt.active_block_types[row[i]].connection_to_cidx.get("dslf", -1)
                for i in (left, right)
            ]
            _connect(connections, pose, left, right, *endpoints)
        for left, left_name, right, right_name in (
            () if explicit is None else explicit[pose]
        ):
            if (
                not isinstance(left, (int, np.integer))
                or not isinstance(right, (int, np.integer))
                or not (0 <= left < blocks and 0 <= right < blocks)
                or row[left] < 0
                or row[right] < 0
                or left == right
            ):
                raise ValueError("invalid inter-residue bond endpoints")
            slots = [
                pbt.active_block_types[row[b]].connection_to_cidx.get(name, -1)
                for b, name in ((left, left_name), (right, right_name))
            ]
            _connect(connections, pose, left, right, *slots)
    return connections


def _bond_separation(pbt, block_types, connections):
    from tmol_core.pose.bond_separation import bondseparation_from_graph

    poses, blocks, slots, _ = connections.shape
    cutoff = MAX_SIG_BOND_SEPARATION
    paths = np.asarray(pbt.intrares_path_distance)
    atoms = np.asarray(pbt.conn_atoms)
    type_counts = np.asarray(pbt.n_conn)
    counts = np.where(block_types >= 0, type_counts[np.maximum(block_types, 0)], 0)
    intra = np.full((poses, blocks, slots, slots), cutoff, np.int32)
    for pose, block in zip(*np.nonzero(block_types >= 0)):
        kind = block_types[pose, block]
        n = counts[pose, block]
        endpoints = atoms[kind, :n]
        distances = paths[kind][np.ix_(endpoints, endpoints)]
        intra[pose, block, :n, :n] = np.where(
            distances < 0, cutoff, np.minimum(distances, cutoff)
        )
    return bondseparation_from_graph(counts, intra, connections)


def _validate_rows(block_types, coordinates, chains, numbers, names, disulfides):
    poses = len(block_types)
    if not poses or any(
        len(rows) != poses for rows in (coordinates, chains, numbers, names, disulfides)
    ):
        raise ValueError("pose inputs must have the same nonzero batch length")
    for types, labels, nums, residue_names in zip(block_types, chains, numbers, names):
        if any(len(row) != len(types) for row in (labels, nums, residue_names)):
            raise ValueError("residue metadata must match each pose's block count")
    if not any(len(row) for row in block_types):
        raise ValueError("pose batch contains no residues")


class PoseStackBuilder:
    @classmethod
    def from_block_types_and_coords(
        cls,
        pbt,
        block_type_inds_per_pose,
        coords_per_pose,
        chain_id_per_pose,
        res_num_per_pose,
        res_name_per_pose=None,
        disulfide_pairs_per_pose=None,
        *,
        connections_per_pose=None,
        _connection_table=None,
    ):
        """Pack aligned arrays; negative block types denote padding.

        Explicit connections are ``(block, connection_name, block,
        connection_name)`` tuples per pose. When supplied, they replace inferred
        polymer links, allowing branches, cycles and disconnected components.
        """
        rows = [np.asarray(row, dtype=np.int32) for row in block_type_inds_per_pose]
        if connections_per_pose is not None and len(connections_per_pose) != len(rows):
            raise ValueError("connections must match the pose batch length")
        if connections_per_pose is not None and disulfide_pairs_per_pose is not None:
            raise ValueError("use explicit connections or disulfide pairs, not both")
        if any(np.any(row >= pbt.n_types) for row in rows):
            raise ValueError("block type index is outside the residue palette")
        if res_name_per_pose is None:
            res_name_per_pose = [
                [
                    pbt.active_block_types[kind].name3 if kind >= 0 else "UNK"
                    for kind in row
                ]
                for row in rows
            ]
        disulfides = (
            [()] * len(rows)
            if disulfide_pairs_per_pose is None
            else disulfide_pairs_per_pose
        )
        _validate_rows(
            rows,
            coords_per_pose,
            chain_id_per_pose,
            res_num_per_pose,
            res_name_per_pose,
            disulfides,
        )
        block_types = np.full((len(rows), max(map(len, rows))), -1, np.int32)
        for pose, row in enumerate(rows):
            block_types[pose, : len(row)] = row
        counts = np.asarray(pbt.n_atoms)[np.maximum(block_types, 0)]
        counts = np.where(block_types >= 0, counts, 0)
        cumulative = np.cumsum(counts, axis=1, dtype=np.int32)
        offsets = cumulative - counts
        totals = cumulative[:, -1]
        dtype = np.result_type(*[np.asarray(x).dtype for x in coords_per_pose])
        if dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise TypeError("coordinates must have float32 or float64 dtype")
        if dtype == np.dtype("float64") and not jax.config.x64_enabled:
            raise ValueError("float64 coordinates require jax_enable_x64=True")
        coords = np.zeros((len(rows), int(totals.max()), 3), dtype)
        for pose, source in enumerate(coords_per_pose):
            source = np.asarray(source, dtype=dtype)
            if source.shape != (totals[pose], 3):
                raise ValueError(
                    "coordinate shape does not match the residue atom counts"
                )
            coords[pose, : totals[pose]] = source
        if _connection_table is None:
            connections = _connections(
                pbt, block_types, chain_id_per_pose, disulfides, connections_per_pose
            )
        else:
            if connections_per_pose is not None or disulfide_pairs_per_pose is not None:
                raise ValueError(
                    "prevalidated connections cannot be combined with bond declarations"
                )
            connections = np.asarray(_connection_table, dtype=np.int32)
            if connections.shape != (*block_types.shape, pbt.max_n_conn, 2):
                raise ValueError(
                    "prevalidated connection table must match the packed layout"
                )
        distances = _bond_separation(pbt, block_types, connections)
        numbers = np.full(block_types.shape, -1, np.int32)
        for pose, row in enumerate(res_num_per_pose):
            numbers[pose, : len(row)] = row
        records = [
            [["ATOM"] * int(counts[p, b]) for b in range(len(rows[p]))]
            for p in range(len(rows))
        ]
        metadata = PDBInfo(
            chain_id=[[str(label) for label in row] for row in chain_id_per_pose],
            res_num=numbers,
            res_name=[list(row) for row in res_name_per_pose],
            atom_record_type=records,
            residue_insertion_codes=np.full(block_types.shape, "", object),
            atom_occupancy=np.full(
                coords.shape[:2], DEFAULT_ATOM_OCCUPANCY, np.float32
            ),
            atom_b_factor=np.full(coords.shape[:2], DEFAULT_ATOM_B_FACTOR, np.float32),
        )
        return PoseStack(
            packed_block_types=pbt,
            coords=jnp.asarray(coords),
            block_coord_offset=jnp.asarray(offsets),
            inter_residue_connections=jnp.asarray(connections),
            inter_block_bondsep=None,
            near_blocks=jnp.asarray(distances.near_blocks),
            bondsep=jnp.asarray(distances.bondsep),
            block_type_ind=jnp.asarray(block_types),
            pdb_info=metadata,
        )
