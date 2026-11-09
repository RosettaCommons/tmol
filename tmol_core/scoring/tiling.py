"""NumPy preparation of stable, padded tile-local indices."""

import numpy


def tile_subset_indices(indices, tile_size, return_args=False, max_entry=None):
    indices = numpy.asarray(indices)
    if max_entry is None:
        max_entry = numpy.amax(indices)
    n_tiles = max_entry // tile_size + 1
    tiled_indices = numpy.full_like(indices, -1, shape=(n_tiles * tile_size,))
    n_in_tile = numpy.full_like(indices, 0, shape=(n_tiles,))
    if return_args:
        ind_arange = numpy.arange(indices.shape[0], dtype=indices.dtype)
    for i in range(n_tiles):
        subset = (indices >= i * tile_size) & (indices < (i + 1) * tile_size)
        subset_size = numpy.sum(subset)
        target = slice(i * tile_size, i * tile_size + subset_size)
        tiled_indices[target] = (
            ind_arange[subset] if return_args else indices[subset] - i * tile_size
        )
        n_in_tile[i] = subset_size
    return tiled_indices, n_in_tile
