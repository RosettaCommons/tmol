"""Logical chain labels for ragged, optionally multichain sequences."""

import numpy


def chain_ids_from_lengths(residue_counts, max_n_residues, chain_lengths=None):
    if chain_lengths is not None and len(chain_lengths) == 0:
        chain_lengths = None
    counts = numpy.asarray(residue_counts)
    result = numpy.full((len(counts), max_n_residues), -1, dtype=numpy.int32)
    if numpy.any(counts < 0) or numpy.any(counts > max_n_residues):
        raise ValueError("Sequence residue counts exceed the padded extent")
    if chain_lengths is not None and len(chain_lengths) != len(counts):
        raise ValueError("Provide chain lengths for every pose")
    for pose, count in enumerate(counts):
        lengths = [int(count)] if chain_lengths is None else chain_lengths[pose]
        if any(length < 0 for length in lengths) or sum(lengths) != count:
            raise ValueError("Chain lengths must partition each pose's residues")
        start = 0
        for chain, length in enumerate(lengths):
            result[pose, start : start + length] = chain
            start += length
    return result
