"""Shared host assembly of conformer forests and generational scan plans."""

from dataclasses import dataclass
import numpy

KINFOREST_FIELDS = ("id", "doftype", "parent", "frame_x", "frame_y", "frame_z")


@dataclass(frozen=True)
class ConformerLayout:
    n_atoms: numpy.ndarray
    offsets: numpy.ndarray
    total: int
    forest: numpy.ndarray
    nodes: numpy.ndarray
    scans: numpy.ndarray
    gens: numpy.ndarray


def prepare_conformer_layout(
    metadata, max_n_atoms, n_atoms_by_type, conformer_type, coordinate_offsets=None
):
    """Plan arbitrary residue conformers behind one shared virtual root.

    Optional coordinate offsets address an existing pose layout; the forest's
    internal atom numbering remains compact. No coordinate physics is executed.
    """
    types = numpy.asarray(conformer_type, dtype=numpy.int32)
    n_atoms = numpy.asarray(n_atoms_by_type, dtype=numpy.int32)[types]
    ends = numpy.cumsum(n_atoms, dtype=numpy.int64)
    total = int(ends[-1]) if len(ends) else 0
    if total >= numpy.iinfo(numpy.int32).max:
        raise ValueError(
            "Conformer atoms exceed 32-bit index capacity; reduce the sampling budget"
        )
    offsets = ends - n_atoms
    if coordinate_offsets is None:
        coordinate_offsets = offsets
    forest = compact_conformer_forest(
        metadata, max_n_atoms, types, total, n_atoms, coordinate_offsets
    )
    nodes, scans, gens = construct_conformer_scans(metadata, types, offsets)
    return ConformerLayout(n_atoms, offsets, total, forest, nodes, scans, gens)


def _update_nodes(
    nodes_orig, genStartsStack, n_nodes_offset_for_rot, n_atoms_offset_for_rot
):
    """Merge residue-local nodes into one generational scan layout.

    Args:
        nodes_orig: Concatenated residue-local node indices.
        genStartsStack: Per-conformer generation and scan boundaries.
        n_nodes_offset_for_rot: Start of each conformer's local node array.
        n_atoms_offset_for_rot: Start of each conformer in the merged forest.

    Returns:
        Merged node indices ordered by generation, then conformer.
    """
    n_rotamers = genStartsStack.shape[0]
    rotamer_indices = numpy.arange(n_rotamers)
    nodes = numpy.empty(nodes_orig.shape[0], dtype=numpy.int32)
    output_offset = 0
    for generation in range(genStartsStack.shape[1] - 1):
        starts = genStartsStack[:, generation, 0]
        lengths = numpy.maximum(genStartsStack[:, generation + 1, 0] - starts, 0)
        count = int(lengths.sum())
        if count == 0:
            continue
        rotamer = numpy.repeat(rotamer_indices, lengths)
        within_rotamer = numpy.arange(count) - numpy.repeat(
            numpy.cumsum(lengths) - lengths, lengths
        )
        source = n_nodes_offset_for_rot[rotamer] + starts[rotamer] + within_rotamer
        values = nodes_orig[source]
        numpy.add(
            values, n_atoms_offset_for_rot[rotamer], out=values, where=values != 0
        )
        nodes[output_offset : output_offset + count] = values
        output_offset += count
    assert output_offset == nodes.shape[0]
    return nodes


def _update_scan_starts(
    n_scans, atomStartsOffsets, scanStartsStack, genStartsStack, ngenStack
):
    """Merge residue-local scan starts into generation-major order."""
    n_rotamers = genStartsStack.shape[0]
    rotamer_indices = numpy.arange(n_rotamers)
    scan_starts = numpy.empty(n_scans, dtype=numpy.int32)
    output_offset = 0
    for generation in range(genStartsStack.shape[1] - 1):
        lengths = ngenStack[generation]
        count = int(lengths.sum())
        if count == 0:
            continue
        rotamer = numpy.repeat(rotamer_indices, lengths)
        within_rotamer = numpy.arange(count) - numpy.repeat(
            numpy.cumsum(lengths) - lengths, lengths
        )
        source = genStartsStack[rotamer, generation, 1] + within_rotamer
        scan_starts[output_offset : output_offset + count] = (
            atomStartsOffsets[generation, rotamer] + scanStartsStack[rotamer, source]
        )
        output_offset += count
    assert output_offset == n_scans
    return scan_starts


def construct_conformer_scans(
    metadata, block_type_ind_for_conf, n_atoms_offset_for_conf
):
    scanStartsStack = metadata.scans[block_type_ind_for_conf]
    genStartsStack = metadata.gens[block_type_ind_for_conf]
    atomStartsStack = numpy.swapaxes(genStartsStack[:, :, 0], 0, 1)
    natomsPerGen = atomStartsStack[1:, :] - atomStartsStack[:-1, :]
    natomsPerGen[natomsPerGen < 0] = 0
    atomStartsOffsets = numpy.cumsum(natomsPerGen, axis=1)
    atomStartsOffsets -= natomsPerGen
    ngenStack = numpy.swapaxes(metadata.n_scans_per_gen[block_type_ind_for_conf], 0, 1)
    ngenStack[ngenStack < 0] = 0
    scanStarts = _update_scan_starts(
        ngenStack.sum(), atomStartsOffsets, scanStartsStack, genStartsStack, ngenStack
    )
    nodes_orig = metadata.nodes[block_type_ind_for_conf].ravel()
    nodes_orig = nodes_orig[nodes_orig >= 0]
    n_nodes_for_conf = metadata.n_nodes[block_type_ind_for_conf]
    first_node_for_conf = numpy.cumsum(n_nodes_for_conf)
    n_nodes_offset_for_conf = first_node_for_conf - n_nodes_for_conf
    nodes = _update_nodes(
        nodes_orig, genStartsStack, n_nodes_offset_for_conf, n_atoms_offset_for_conf
    )
    gen_starts = numpy.sum(genStartsStack, axis=0)
    return (nodes, scanStarts, gen_starts)


def _load_from_rotamers(arr, n_atoms_total, n_atoms_for_rot):
    """Compact real per-conformer entries behind one virtual root."""
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    atom_is_real = numpy.arange(arr.shape[1]) < n_atoms_for_rot[:, None]
    compact_arr[1:] = arr[atom_is_real]
    return compact_arr


def _load_from_rotamers_w_offsets(
    arr, n_atoms_total, n_atoms_for_rot, n_atoms_offset_for_rot
):
    """Compact real entries and translate their conformer-local indices."""
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    atom_is_real = numpy.arange(arr.shape[1]) < n_atoms_for_rot[:, None]
    offsets = numpy.repeat(
        n_atoms_offset_for_rot[: n_atoms_for_rot.shape[0]], n_atoms_for_rot
    )
    compact_arr[1:] = arr[atom_is_real] + offsets
    return compact_arr


def _load_rotamer_parents(
    parents, n_atoms_total, n_atoms_for_rot, n_atoms_offset_for_rot
):
    """Compact parent indices while joining conformers at a virtual root."""
    compact_arr = numpy.zeros((n_atoms_total + 1,), dtype=numpy.int32)
    atom_is_real = numpy.arange(parents.shape[1]) < n_atoms_for_rot[:, None]
    offsets = numpy.repeat(
        n_atoms_offset_for_rot[: n_atoms_for_rot.shape[0]], n_atoms_for_rot
    )
    rotamer_starts = numpy.cumsum(n_atoms_for_rot) - n_atoms_for_rot
    rotamer_starts = rotamer_starts[n_atoms_for_rot > 0]
    offsets[rotamer_starts] = 1
    compact_arr[1:] = parents[atom_is_real] + offsets
    return compact_arr


def compact_conformer_forest(
    metadata,
    max_n_atoms,
    conf_block_type_ind,
    n_atoms_total,
    n_atoms_for_conf,
    block_offset_for_conf,
):
    """Construct a KinForest for a set of conformers by stringing
    together the kinforest data for individual conformers.
    The "block_ofset_for_conf" array is used to construct
    the "id" tensor in the KinForest, which maps to the atom
    indices; thus it should contain the atom-index offsets
    for the first atom in each rotamer in the coords tensor
    that will be used to construct the kinforest_coords tensor.
    """
    starts = numpy.cumsum(n_atoms_for_conf, dtype=numpy.int64) - n_atoms_for_conf
    source = numpy.repeat(
        conf_block_type_ind.astype(numpy.int64) * max_n_atoms - starts, n_atoms_for_conf
    )
    source += numpy.arange(n_atoms_total)
    offsets = numpy.repeat(starts + 1, n_atoms_for_conf)
    fields = KINFOREST_FIELDS
    compact = numpy.zeros((len(fields), n_atoms_total + 1), dtype=numpy.int32)
    for i, name in enumerate(fields):
        compact[i, 1:] = getattr(metadata, name).reshape(-1)[source]
        if name == "id":
            compact[i, 1:] += numpy.repeat(
                block_offset_for_conf[: n_atoms_for_conf.shape[0]], n_atoms_for_conf
            )
        elif name != "doftype":
            compact[i, 1:] += offsets
    compact[0, 0] = -1
    compact[2, starts[n_atoms_for_conf > 0] + 1] = 0
    return compact
