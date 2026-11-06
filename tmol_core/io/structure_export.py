"""Host molecular export shared by all execution frontends."""

import warnings
import numpy
import biotite.structure

from tmol_core.chemical.restypes import BondType as ChemBondType
from tmol_core.chemical.bond_orders import _chemical_bond_orders
from tmol_core.chemical import _site_connections
from tmol_core.io.atom_array import template_array as _template_array
from tmol_core.io.atomworks_reader import _renumbered_decreasing_chains


def _with_metals_rejoined(structure, block_for_atom, origins):
    """Put each split-out metal back in the component it came from."""
    block_for_atom = numpy.asarray(block_for_atom)
    after = {}
    for block in numpy.flatnonzero([o is not None for o in origins]):
        res_id, ins_code, res_name, atom_name = origins[block]
        metal = numpy.flatnonzero(block_for_atom == block)
        if len(metal) != 1:
            continue
        metal = int(metal[0])
        component = numpy.flatnonzero(
            (structure.chain_id == structure.chain_id[metal])
            & (structure.res_id == res_id)
            & (structure.ins_code == ins_code)
            & (structure.res_name == res_name)
        )
        if not len(component):
            continue
        structure.res_id[metal] = res_id
        structure.res_name[metal] = res_name
        structure.atom_name[metal] = atom_name
        structure.ins_code[metal] = ins_code
        structure.hetero[metal] = structure.hetero[component[0]]
        after[metal] = int(component[-1])
        # within a residue CIF has no coordination order; the split retypes them
        bonds = structure.bonds.as_array()
        inside = ((bonds[:, 0] == metal) & numpy.isin(bonds[:, 1], component)) | (
            (bonds[:, 1] == metal) & numpy.isin(bonds[:, 0], component)
        )
        bonds[inside, 2] = biotite.structure.BondType.SINGLE
        structure.bonds = biotite.structure.BondList(structure.array_length(), bonds)
    if not after:
        return structure
    order = []
    for i in range(structure.array_length()):
        if i in after:
            continue
        order.append(i)
        order += [m for m, last in after.items() if last == i]
    if isinstance(structure, biotite.structure.AtomArrayStack):
        return structure[:, order]
    return structure[order]


def _bonds_from_pose_stack(pose_stack, structure, block_for_atom):
    """The bond table of an exported structure, from the pose's residue types, which
    every pose of the stack must share; bonds to atoms left out are dropped."""
    block_types = pose_stack.block_type_ind64
    connections = pose_stack.inter_residue_connections64
    if not (
        numpy.all(block_types == block_types[:1])
        and numpy.all(connections == connections[:1])
    ):
        raise ValueError(
            "poses with different residue types or connections cannot share "
            "one bond table"
        )
    pbt = pose_stack.packed_block_types
    bt_for_block = block_types[0].tolist()
    irc = connections[0].tolist()
    index = {
        (int(block), str(name)): i
        for i, (block, name) in enumerate(zip(block_for_atom, structure.atom_name))
    }

    sites_for_type = {}

    def metal_bond(bt, bt_ind, conn):
        if bt_ind not in sites_for_type:
            sites_for_type[bt_ind] = frozenset(_site_connections(bt))
        return bt.connections[conn].name in sites_for_type[bt_ind]

    orders_for_type = {}
    bonds = []
    for block, bt_ind in enumerate(bt_for_block):
        if bt_ind < 0:
            continue
        bt = pbt.active_block_types[bt_ind]
        if bt_ind not in orders_for_type:
            # Only the first block of each type needs this; there are two dozen
            # types behind hundreds of blocks.
            orders_for_type[bt_ind] = _chemical_bond_orders(bt, pbt.chem_db)
        for key, order in orders_for_type[bt_ind].items():
            a, b = tuple(key)
            i, j = index.get((block, a)), index.get((block, b))
            if i is not None and j is not None:
                bonds.append((i, j, order))
        for conn, (partner, partner_conn) in enumerate(irc[block]):
            if partner < 0 or (partner, partner_conn) <= (block, conn):
                continue
            other = pbt.active_block_types[bt_for_block[partner]]
            i = index.get((block, bt.connections[conn].atom))
            j = index.get((partner, other.connections[partner_conn].atom))
            if i is None or j is None:
                continue
            if metal_bond(bt, bt_ind, conn) or metal_bond(
                other, bt_for_block[partner], partner_conn
            ):
                bonds.append((i, j, "COORDINATION"))
            else:
                order = ChemBondType(bt.connection_bond_types[conn]).name
                bonds.append((i, j, "SINGLE" if order == "AROMATIC" else order))

    bond_array = numpy.array(
        [
            (i, j, int(getattr(biotite.structure.BondType, order)))
            for i, j, order in bonds
        ],
        dtype=numpy.int64,
    ).reshape(-1, 3)
    return biotite.structure.BondList(structure.array_length(), bond_array)


def _renumbered_for_cif(structure):
    """Renumber, with a warning, the chains biotite's CIF round trip cannot carry.

    biotite reads a label_seq_id of -1 as missing, and atomworks rejects numbering
    that decreases within a chain: the first is shifted to start at 1, the second 1..N.
    """
    template = _template_array(structure)
    res_id, ins_code, decreasing = _renumbered_decreasing_chains(template)
    reasons = dict.fromkeys(decreasing, "numbering that decreases within a chain")
    for chain in dict.fromkeys(template.chain_id.tolist()):
        in_chain = template.chain_id == chain
        if chain not in reasons and (res_id[in_chain] == -1).any():
            reasons[chain] = "a residue id of -1"
            res_id[in_chain] += 1 - res_id[in_chain].min()
    for chain, reason in reasons.items():
        warnings.warn(
            f"Renumbering chain {chain} of the output AtomArray: biotite's CIF "
            f"writer does not support {reason}"
        )
    if not reasons:
        return structure
    structure = structure.copy()
    structure.res_id = res_id
    structure.ins_code = ins_code
    return structure


def _biotite_from_canonical_form_impl(cf, co, include_virtual_atoms):
    """The exported structure, and the canonical residue of each of its atoms."""
    import biotite.structure as struc

    n_poses, n_residues, max_atoms = cf.coords.shape[:3]
    if n_poses > 1 and not _poses_have_identical_metadata(cf):
        raise ValueError(
            "Cannot convert CanonicalForm with multiple poses to biotite structure: "
            "poses have different metadata. Only coordinate differences are allowed "
            "for multi-pose conversion."
        )

    coords = cf.coords
    res_types = cf.res_types[0]
    present = ~numpy.isnan(coords).any(axis=-1)
    atom_mask = present.any(axis=0)
    names, elements, rows, columns = [], [], [], []
    for res_id in numpy.flatnonzero(res_types >= 0):
        res_name = co.restype_io_equiv_classes[res_types[res_id]]
        atom_names = co.restypes_ordered_atom_names[res_name][:max_atoms]
        virtual = (
            frozenset()
            if include_virtual_atoms
            else co.restypes_virtual_atoms.get(res_name, frozenset())
        )
        indices = [
            i
            for i in numpy.flatnonzero(atom_mask[res_id, : len(atom_names)])
            if atom_names[i] not in virtual
        ]
        names.extend(atom_names[i] for i in indices)
        elements.extend(
            co.restypes_atom_elements[res_name][atom_names[i]] for i in indices
        )
        rows.extend([res_id] * len(indices))
        columns.extend(indices)
    rows, columns = numpy.asarray(rows, dtype=int), numpy.asarray(columns, dtype=int)
    result = (
        struc.AtomArray(len(rows))
        if n_poses == 1
        else struc.AtomArrayStack(n_poses, len(rows))
    )
    selected = coords[:, rows, columns]
    selected[~present[:, rows, columns]] = numpy.nan
    result.coord = selected[0] if n_poses == 1 else selected
    result.set_annotation("atom_name", numpy.asarray(names, dtype=str))
    result.set_annotation("element", numpy.asarray(elements, dtype=str))
    result.set_annotation(
        "res_name", numpy.asarray(co.restype_io_equiv_classes)[res_types[rows]]
    )
    chain_labels = (
        cf.chain_labels if cf.chain_labels is not None else cf.chain_id.astype(str)
    )
    res_labels = (
        cf.res_labels
        if cf.res_labels is not None
        else numpy.arange(1, n_residues + 1)[None, :]
    )
    result.set_annotation("chain_id", numpy.asarray(chain_labels[0, rows], dtype=str))
    result.set_annotation("res_id", res_labels[0, rows])
    if cf.residue_insertion_codes is not None:
        result.set_annotation(
            "ins_code", numpy.asarray(cf.residue_insertion_codes[0, rows], dtype=str)
        )
    for name, values in (
        ("b_factor", cf.atom_b_factor),
        ("occupancy", cf.atom_occupancy),
    ):
        if values is not None:
            result.set_annotation(name, values[0, rows, columns].copy())
    if cf.residue_annotations is not None:
        annotations = cf.residue_annotations[0, rows]
        for name in annotations.dtype.names:
            values = annotations[name]
            result.set_annotation(
                name, values.astype(str) if values.dtype == object else values
            )
    return result, rows


def _poses_have_identical_metadata(cf) -> bool:
    """Check if all poses in the CanonicalForm have identical metadata.

    Returns True if all poses have the same:
    - chain_id
    - res_types
    - res_labels
    - residue_insertion_codes
    - chain_labels
    - atom_occupancy and atom_b_factor

    Only coordinates are allowed to differ between poses.
    """
    n_poses = cf.coords.shape[0]

    if n_poses <= 1:
        return True

    if not numpy.all(cf.chain_id[0] == cf.chain_id[1:]):
        return False

    if not numpy.all(cf.res_types[0] == cf.res_types[1:]):
        return False

    for values in (
        cf.res_labels,
        cf.residue_insertion_codes,
        cf.chain_labels,
        cf.atom_b_factor,
        cf.atom_occupancy,
        cf.metal_origins,
        cf.residue_annotations,
    ):
        if values is not None and not all(
            numpy.array_equal(values[0], row, equal_nan=values.dtype.kind in "fc")
            for row in values[1:]
        ):
            return False
    return True


def canonical_export_form(pose, co, chain_id):
    """Project detached host coordinates/metadata using shared canonical atom IDs.

    This is an export view, not a canonical construction record: inter-residue
    bonds are carried by the pose and assembled by the same exporter below.
    """
    from types import SimpleNamespace
    from tmol_core.cache import AnnotationKey, PreparationCache
    from tmol_core.io.canonical import _create_bt_to_canonical_atom_mapping
    from tmol_core.io.pdb_info import DEFAULT_ATOM_B_FACTOR, DEFAULT_ATOM_OCCUPANCY

    pbt = pose.packed_block_types
    cache = getattr(pbt, "_export_preparations", None)
    if cache is None:
        cache = pbt._export_preparations = PreparationCache()
    classes, atom_map = cache.get_or_create(
        "canonical_atoms",
        AnnotationKey.from_sources(co, pbt.chem_db),
        lambda: _create_bt_to_canonical_atom_mapping(co, pbt),
    )
    types = pose.block_type_ind64
    real = types >= 0
    poses, blocks = types.shape
    local = numpy.arange(pbt.max_n_atoms)
    counts = numpy.array([len(bt.atoms) for bt in pbt.active_block_types])
    present = real[..., None] & (local < counts[numpy.maximum(types, 0)][..., None])
    p, b, a = numpy.nonzero(present)
    canonical = atom_map[types[p, b], a]
    if numpy.any(canonical < 0):
        raise ValueError(
            "pose atom has no identity in the requested canonical ordering"
        )
    source = pose.block_coord_offset[p, b] + a
    shape = (poses, blocks, co.max_n_canonical_atoms)
    coords = numpy.full((*shape, 3), numpy.nan, pose.coords.dtype)
    coords[p, b, canonical] = pose.coords[p, source]
    res_types = numpy.full((poses, blocks), -1, numpy.int32)
    res_types[real] = classes[types[real]]
    metadata = pose.pdb_info
    values = dict(coords=coords, res_types=res_types, chain_id=chain_id)
    for name, default in (
        ("atom_occupancy", DEFAULT_ATOM_OCCUPANCY),
        ("atom_b_factor", DEFAULT_ATOM_B_FACTOR),
    ):
        out = numpy.full(shape, default, numpy.float32)
        old = getattr(metadata, name, None)
        if old is not None:
            out[p, b, canonical] = old[p, source]
        values[name] = out
    for name, field in (
        ("res_labels", "residue_labels"),
        ("chain_labels", "chain_labels"),
        ("residue_insertion_codes", "residue_insertion_codes"),
        ("metal_origins", "metal_origins"),
        ("residue_annotations", "residue_annotations"),
    ):
        value = getattr(metadata, field, None)
        values[name] = None if value is None else numpy.asarray(value)
    return SimpleNamespace(**values)


def atom_array_from_host_pose(
    pose, co, chain_id, *, merge_fragments=True, include_virtual_atoms=False
):
    """Export real molecular atoms, declared bonds, metadata and original identities."""
    from tmol_core.ligand.fragment_imposition import recombine_fragmented_ligands

    cf = canonical_export_form(pose, co, chain_id)
    structure, blocks = _biotite_from_canonical_form_impl(cf, co, include_virtual_atoms)
    structure.bonds = _bonds_from_pose_stack(pose, structure, blocks)
    if "is_polymer" not in structure.get_annotation_categories():
        polymer = numpy.array(
            [
                bt.properties.polymer.is_polymer
                for bt in pose.packed_block_types.active_block_types
            ]
        )
        structure.set_annotation(
            "is_polymer", polymer[pose.block_type_ind64[0, blocks]]
        )
    if not include_virtual_atoms and cf.metal_origins is not None:
        structure = _with_metals_rejoined(structure, blocks, cf.metal_origins[0])
    if merge_fragments and pose.split_block_mapping is not None:
        structure = recombine_fragmented_ligands(
            structure,
            pose.split_block_mapping,
            cf.res_labels,
            pose.block_type_ind64,
            pose.packed_block_types.active_block_types,
        )
    return _renumbered_for_cif(structure)
