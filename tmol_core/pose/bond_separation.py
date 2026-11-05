"""Sparse, capped shortest paths between molecular residue connections.

The frontier expansion and compact sorted-block representation are shared by
all frontends. Inputs and outputs are host arrays; execution adapters own any
device transfer. Storage follows actual short paths, rather than a dense
all-connection-pairs matrix, including metal clusters and arbitrary crosslinks.
"""

import attr
import numpy
from tmol_core.chemical.constants import MAX_SIG_BOND_SEPARATION


@attr.s(auto_attribs=True, frozen=True, slots=True)
class InterBlockBondsepData:
    near_blocks: numpy.ndarray
    bondsep: numpy.ndarray

    @classmethod
    def empty(cls, n_poses, max_n_blocks, max_n_conn, n_slots):
        near = numpy.full(
            (n_poses, max_n_blocks, n_slots, 2),
            MAX_SIG_BOND_SEPARATION,
            dtype=numpy.int32,
        )
        near[..., 0] = -1
        separations = numpy.full(
            (n_poses, max_n_blocks, n_slots, max_n_conn, max_n_conn),
            MAX_SIG_BOND_SEPARATION,
            dtype=numpy.int8,
        )
        return cls(near, separations)


def bondseparation_from_graph(counts, intra_separation, connections):
    """Resolve capped distances for counts[P,B], intra[P,B,C,C], links[P,B,C,2].

    Each connection endpoint is (block, connection) in the same pose; block -1
    means unbound. Padded blocks have zero counts. Result near_blocks[P,B,K,2]
    stores (partner block, minimum distance), in sorted partner order, with a
    final (-1, cap) sentinel slot; bondsep[P,B,K,C,C] stores int8 separations.
    """
    counts = numpy.asarray(counts)
    intra_separation = numpy.asarray(intra_separation)
    connections = numpy.asarray(connections)
    n_poses, max_n_blocks, max_n_conn = connections.shape[:3]
    block_counts = counts.flatten().astype(numpy.int64)
    node_start = numpy.cumsum(block_counts, 0) - block_counts
    n_nodes = int(block_counts.sum())
    conn = numpy.arange(max_n_conn, dtype=numpy.int64)
    real = conn < counts[..., None]
    pose, block, conn1, conn2 = numpy.nonzero(real[..., :, None] & real[..., None, :])
    first = node_start[pose * max_n_blocks + block]
    intra_key = (first + conn1) * n_nodes + first + conn2
    intra_weight = intra_separation[pose, block, conn1, conn2].astype(numpy.int32)
    pose, block, conn1 = numpy.nonzero(connections[..., 0] != -1)
    partner = connections[pose, block, conn1]
    inter_key = (node_start[pose * max_n_blocks + block] + conn1) * n_nodes + (
        node_start[pose * max_n_blocks + partner[:, 0]] + partner[:, 1]
    )

    edge_key, edge_of_key = numpy.unique(
        numpy.concatenate((intra_key, inter_key)), return_inverse=True
    )
    edge_weight = numpy.empty(edge_key.shape, dtype=numpy.int32)
    edge_weight[edge_of_key[: intra_key.shape[0]]] = intra_weight
    edge_weight[edge_of_key[intra_key.shape[0] :]] = 1
    edge_src = edge_key // n_nodes
    edge_dst = edge_key % n_nodes
    out_degree = numpy.bincount(edge_src, minlength=n_nodes)
    out_start = numpy.cumsum(out_degree, 0) - out_degree

    # extend every path improved in the last round by one edge, until none improves
    best_key, best_sep = edge_key, edge_weight
    front_src, front_node, front_sep = edge_src, edge_dst, edge_weight
    while front_src.shape[0] > 0:
        n_out = out_degree[front_node]
        path = numpy.repeat(numpy.arange(front_src.shape[0]), n_out)
        path_start = numpy.cumsum(n_out, 0) - n_out
        edge = out_start[front_node][path] + (
            numpy.arange(path.shape[0]) - path_start[path]
        )
        sep = front_sep[path] + edge_weight[edge]
        short = sep < MAX_SIG_BOND_SEPARATION
        cand_key = front_src[path][short] * n_nodes + edge_dst[edge][short]

        merged_key, merged_of = numpy.unique(
            numpy.concatenate((best_key, cand_key)), return_inverse=True
        )
        merged_sep = numpy.full(
            merged_key.shape,
            MAX_SIG_BOND_SEPARATION,
            dtype=numpy.int32,
        )
        numpy.minimum.at(
            merged_sep, merged_of, numpy.concatenate((best_sep, sep[short]))
        )
        previous = numpy.full_like(merged_sep, MAX_SIG_BOND_SEPARATION)
        previous[merged_of[: best_key.shape[0]]] = best_sep
        improved = merged_sep < previous
        best_key, best_sep = merged_key, merged_sep
        front_src = merged_key[improved] // n_nodes
        front_node = merged_key[improved] % n_nodes
        front_sep = merged_sep[improved]

    node_row = numpy.repeat(numpy.arange(block_counts.shape[0]), block_counts)
    node_conn = numpy.arange(n_nodes) - node_start[node_row]
    src = best_key // n_nodes
    dst = best_key % n_nodes
    # node_row is pose * max_n_blocks + block; one slot per (row, block2)
    n_rows = n_poses * max_n_blocks
    pair_key, pair_of_entry = numpy.unique(
        node_row[src] * max_n_blocks + node_row[dst] % max_n_blocks,
        return_inverse=True,
    )
    pair_row = pair_key // max_n_blocks
    row_len = numpy.bincount(pair_row, minlength=n_rows)
    n_slots = int(row_len.max()) + 1 if n_rows else 1
    row_start = numpy.cumsum(row_len, 0) - row_len
    pair_slot = numpy.arange(pair_key.shape[0], dtype=numpy.int64) - row_start[pair_row]

    result = InterBlockBondsepData.empty(n_poses, max_n_blocks, max_n_conn, n_slots)
    near_blocks = result.near_blocks.reshape(n_rows, n_slots, 2)
    bondsep = result.bondsep.reshape(n_rows, n_slots, max_n_conn, max_n_conn)
    near_blocks[pair_row, pair_slot, 0] = (pair_key % max_n_blocks).astype(numpy.int32)
    entry_row, entry_slot = pair_row[pair_of_entry], pair_slot[pair_of_entry]
    bondsep[entry_row, entry_slot, node_conn[src], node_conn[dst]] = best_sep.astype(
        numpy.int8
    )
    if max_n_conn > 0:
        near_blocks[..., 1] = numpy.amin(bondsep, axis=(2, 3)).astype(numpy.int32)
    return result
