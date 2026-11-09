"""Physical chain membership from the mature polymer-port graph policy."""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


def connection_masks(block_types, max_connections):
    """Polymer chains use backbone ports; nonpolymers use all chemical ports."""
    masks = np.zeros((len(block_types), max_connections), bool)
    for index, block in enumerate(block_types):
        if block.properties.polymer.is_polymer:
            for port in (block.up_connection_ind, block.down_connection_ind):
                if port >= 0:
                    masks[index, port] = True
        else:
            masks[index, : len(block.connections)] = True
    return masks


def chain_indices(types, connections, masks):
    """Return local connected-component labels, keeping gap residues at -1."""
    poses, blocks = types.shape
    if blocks == 0:
        return np.empty((poses, 0), np.int64)
    real = types >= 0
    valid = real[..., None] & masks[np.maximum(types, 0)]
    p, b, port = np.nonzero(valid & (connections[..., 0] >= 0))
    partners = connections[p, b, port, 0]
    first = p * blocks + b
    second = p * blocks + partners
    pairs = np.concatenate(
        (np.stack((first, second), axis=1), np.stack((second, first), axis=1))
    )
    graph = csr_matrix(
        (np.ones(len(pairs), np.int32), (pairs[:, 0], pairs[:, 1])),
        shape=(poses * blocks, poses * blocks),
    )
    _, labels = connected_components(graph, directed=False, return_labels=True)
    labels = labels.reshape(poses, blocks)
    counts = np.amax(labels, axis=1) - np.amin(labels, axis=1) + 1
    offsets = np.cumsum(counts) - counts
    labels = labels - offsets[:, None]
    labels[~real] = -1
    return labels
