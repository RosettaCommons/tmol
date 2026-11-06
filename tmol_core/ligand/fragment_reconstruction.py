"""Shared fragment reconstruction plans; coordinates stay in their runtime."""

from dataclasses import dataclass
import numpy as np


def _unsplit_group_entries(sbm):
    """Return (groups, split_block_set, entry_lookup) from a SplitBlockMapping."""
    from collections import defaultdict

    groups = defaultdict(list)
    split_block_set = defaultdict(set)
    for e in sbm.entries:
        groups[(e.pose_ind, e.group_ind)].append(e)
        split_block_set[e.pose_ind].add(e.block_ind)
    for key in groups:
        groups[key].sort(key=lambda e: e.block_ind)
    entry_lookup = {(e.pose_ind, e.block_ind): e for e in sbm.entries}
    return groups, split_block_set, entry_lookup


def _unsplit_per_pose_blocks(pose_stack, split_block_set, entry_lookup):
    """Build per-pose new block sequence (bt_ind, kind, src) for unsplitting."""
    n_poses = len(pose_stack.block_type_ind64)
    old_max = int(pose_stack.block_type_ind64.shape[1])
    per_pose = []
    for p in range(n_poses):
        seen, blocks = set(), []
        for b in range(old_max):
            bt = int(pose_stack.block_type_ind64[p, b].item())
            if bt < 0:
                continue
            if b not in split_block_set[p]:
                blocks.append((bt, "orig", b))
            else:
                e = entry_lookup[(p, b)]
                gkey = (p, e.group_ind)
                if gkey not in seen:
                    seen.add(gkey)
                    blocks.append((e.orig_block_type_ind, "group", gkey))
        per_pose.append(blocks)
    return per_pose


def _unsplit_build_coords(
    per_pose_blocks, pbt, groups, pose_stack, new_bco, new_max_n_atoms, n_poses
):
    """Compatibility host gather preserving coordinate precision."""
    indices = coordinate_sources(
        per_pose_blocks, pbt, groups, pose_stack, new_bco, new_max_n_atoms
    )
    out = pose_stack.coords[np.arange(n_poses)[:, None], np.maximum(indices, 0)]
    return np.where((indices >= 0)[..., None], out, 0), new_bco


def _unsplit_old_to_new(per_pose_blocks, pose_stack, split_block_set, entry_lookup):
    """Build per-pose mapping from old block indices to new (or -1 if absorbed)."""
    old_max = int(pose_stack.block_type_ind64.shape[1])
    old_to_new = []
    for p, blocks in enumerate(per_pose_blocks):  # noqa: B007
        seen, m, idx = set(), {}, 0
        for b in range(old_max):
            if int(pose_stack.block_type_ind64[p, b].item()) < 0:
                continue
            if b not in split_block_set[p]:
                m[b] = idx
                idx += 1
            else:
                e = entry_lookup[(p, b)]
                gkey = (p, e.group_ind)
                if gkey not in seen:
                    seen.add(gkey)
                    m[b] = idx
                    idx += 1
                else:
                    m[b] = -1
        old_to_new.append(m)
    return old_to_new


def _unsplit_connections(
    pose_stack,
    pbt,
    n_poses,
    new_max_n_blocks,
    split_block_set,
    entry_lookup,
    old_to_new,
    device,
):
    """Remap inter-residue connections, discarding intra-fragment ones."""

    max_n_conn = int(pose_stack.inter_residue_connections64.shape[2])
    out = np.full((n_poses, new_max_n_blocks, max_n_conn, 2), -1, dtype=np.int64)
    old_irc = pose_stack.inter_residue_connections64
    for p in range(n_poses):
        m = old_to_new[p]
        for old_b, new_b in m.items():
            if new_b < 0:
                continue
            if old_b in split_block_set[p]:
                e = entry_lookup[(p, old_b)]
                n_conn = len(pbt.active_block_types[e.orig_block_type_ind].connections)
            else:
                n_conn = len(
                    pbt.active_block_types[
                        int(pose_stack.block_type_ind64[p, old_b].item())
                    ].connections
                )
            for c in range(n_conn):
                partner = old_irc[p, old_b, c]
                pb = int(partner[0].item())
                if pb == -1 or (
                    old_b in split_block_set[p] and pb in split_block_set[p]
                ):
                    continue
                pc = int(partner[1].item())
                npb = m.get(pb, -1)
                if npb >= 0:
                    out[p, new_b, c] = (npb, pc)
    return out


def _unsplit_chain_and_pdb(
    pose_stack,
    old_to_new,
    pbt,
    per_pose_blocks,
    new_bco_np,
    n_poses,
    new_max_n_blocks,
    new_max_n_atoms,
    device,
):
    """Build new chain_id tensor and PDBInfo for the unsplit pose."""
    from tmol_core.io.pdb_info import (
        PDBInfo,
        DEFAULT_ATOM_OCCUPANCY,
        DEFAULT_ATOM_B_FACTOR,
    )

    old_pdb = pose_stack.pdb_info
    old_bco = pose_stack.block_coord_offset
    chain_np = np.full((n_poses, new_max_n_blocks), -1, dtype=np.int32)
    res_labels = np.zeros((n_poses, new_max_n_blocks), dtype=int)
    ins_codes = np.full((n_poses, new_max_n_blocks), "", dtype=object)
    chain_labels = np.full((n_poses, new_max_n_blocks), "", dtype=object)
    occ = np.full((n_poses, new_max_n_atoms), DEFAULT_ATOM_OCCUPANCY, dtype=np.float32)
    bf = np.full((n_poses, new_max_n_atoms), DEFAULT_ATOM_B_FACTOR, dtype=np.float32)
    origins = (
        None
        if old_pdb.metal_origins is None
        else np.full((n_poses, new_max_n_blocks), None, dtype=object)
    )
    annotations = (
        None
        if old_pdb.residue_annotations is None
        else np.zeros((n_poses, new_max_n_blocks), old_pdb.residue_annotations.dtype)
    )
    old_chain = pose_stack.chain_id
    for p, (blocks, m) in enumerate(zip(per_pose_blocks, old_to_new)):
        for old_b, new_b in m.items():
            if new_b < 0:
                continue
            chain_np[p, new_b] = old_chain[p, old_b]
            res_labels[p, new_b] = old_pdb.residue_labels[p, old_b]
            ins_codes[p, new_b] = old_pdb.residue_insertion_codes[p, old_b]
            chain_labels[p, new_b] = old_pdb.chain_labels[p, old_b]
            if origins is not None:
                origins[p, new_b] = old_pdb.metal_origins[p, old_b]
            if annotations is not None:
                annotations[p, new_b] = old_pdb.residue_annotations[p, old_b]
        for new_b, (bt_idx, kind, src) in enumerate(blocks):
            if kind != "orig":
                continue
            new_off = int(new_bco_np[p, new_b])
            n_at = int(pbt.n_atoms[bt_idx].item())
            old_off = int(old_bco[p, src])
            occ[p, new_off : new_off + n_at] = old_pdb.atom_occupancy[
                p, old_off : old_off + n_at
            ]
            bf[p, new_off : new_off + n_at] = old_pdb.atom_b_factor[
                p, old_off : old_off + n_at
            ]
    new_chain_id = chain_np
    new_pdb = PDBInfo(
        residue_labels=res_labels,
        residue_insertion_codes=ins_codes,
        chain_labels=chain_labels,
        atom_occupancy=occ,
        atom_b_factor=bf,
        metal_origins=origins,
        residue_annotations=annotations,
    )
    return new_chain_id, new_pdb


def coordinate_sources(blocks_by_pose, pbt, groups, pose, offsets, capacity):
    """Plan a gather; the last fragment owns duplicate atoms as in mature TMol."""
    source = np.full((len(blocks_by_pose), capacity), -1, np.int64)
    for p, blocks in enumerate(blocks_by_pose):
        for b, (bt, kind, src) in enumerate(blocks):
            offset = int(offsets[p, b])
            if kind == "orig":
                n = int(pbt.n_atoms[bt])
                source[p, offset : offset + n] = pose.block_coord_offset[
                    p, src
                ] + np.arange(n)
            else:
                for entry in groups[src]:
                    atoms = entry.split_to_orig_atom_inds
                    source[p, offset + atoms] = pose.block_coord_offset[
                        p, entry.block_ind
                    ] + np.arange(len(atoms))
    return source


@dataclass(frozen=True)
class ReconstructionPlan:
    block_types: np.ndarray
    block_offsets: np.ndarray
    connections: np.ndarray
    coordinate_sources: np.ndarray
    chain_ids: np.ndarray
    pdb_info: object
    atom_destinations: dict


def reconstruction_plan(pose, pbt, mapping):
    """Prepare shared block layout, atom gather, connections and host metadata."""
    groups, split_blocks, lookup = _unsplit_group_entries(mapping)
    blocks = _unsplit_per_pose_blocks(pose, split_blocks, lookup)
    poses = len(blocks)
    width = max(map(len, blocks), default=0)
    types = np.full((poses, width), -1, np.int32)
    counts = np.zeros_like(types)
    for p, row in enumerate(blocks):
        for b, (bt, _, _) in enumerate(row):
            types[p, b] = bt
            counts[p, b] = pbt.n_atoms[bt]
    offsets = np.cumsum(counts, axis=1, dtype=np.int32) - counts
    capacity = int(counts.sum(axis=1).max(initial=0))
    source = coordinate_sources(blocks, pbt, groups, pose, offsets, capacity)
    old_to_new = _unsplit_old_to_new(blocks, pose, split_blocks, lookup)
    connections = _unsplit_connections(
        pose, pbt, poses, width, split_blocks, lookup, old_to_new, None
    )
    chains, metadata = _unsplit_chain_and_pdb(
        pose, old_to_new, pbt, blocks, offsets, poses, width, capacity, None
    )
    # Restore author identity for the reconstructed original, while the private
    # metadata helper retains its historical survivor-block behavior.
    for p, row in enumerate(blocks):
        for b, (_, kind, src) in enumerate(row):
            if kind == "group":
                first = groups[src][0]
                metadata.residue_labels[p, b] = first.orig_residue_label
                metadata.chain_labels[p, b] = first.orig_chain_label
                metadata.residue_insertion_codes[p, b] = first.orig_ins_code
    # Per-atom observations follow the same physical atoms as coordinates.
    for name in ("atom_occupancy", "atom_b_factor"):
        old = getattr(pose.pdb_info, name)
        if old is not None:
            out = getattr(metadata, name)
            p, a = np.nonzero(source >= 0)
            out[p, a] = old[p, source[p, a]]
    destinations = {}
    for p, row in enumerate(blocks):
        for b, (bt, kind, src) in enumerate(row):
            if kind == "orig":
                destinations[p, src] = (
                    b,
                    np.arange(int(pbt.n_atoms[bt]), dtype=np.int32),
                )
            else:
                for entry in groups[src]:
                    destinations[p, entry.block_ind] = (
                        b,
                        entry.split_to_orig_atom_inds,
                    )
    return ReconstructionPlan(
        types, offsets, connections, source, chains, metadata, destinations
    )


def remap_constraints(topology, destinations):
    """Retain parameter/function rows while rebuilding physical atom topology."""
    from tmol_core.pose.constraints import ConstraintTopology

    arrays = {name: value.copy() for name, value in topology.arrays().items()}
    atoms = arrays["constraint_atoms"]
    valid = atoms[..., 0] >= 0
    p, b, a = atoms[valid].T
    mapped = np.empty((len(p), 2), np.int32)
    for row, (pose, block, atom) in enumerate(zip(p, b, a)):
        new_block, atom_map = destinations[pose, block]
        mapped[row] = new_block, atom_map[atom]
    atoms[valid, 1:] = mapped
    if np.any(atoms[valid] < 0):
        raise ValueError("constraint atom absent from reconstructed molecule")
    # Reuse the common topology constructor to derive unique-block contracts.
    for arity in range(1, 5):
        rows = np.flatnonzero(valid.sum(axis=1) == arity)
        if not len(rows):
            continue
        compact = atoms[rows][valid[rows]].reshape(len(rows), arity, 3)
        derived, _ = ConstraintTopology.empty(topology.n_poses).add(compact, 0)
        for name in ("constraint_num_unique_blocks", "constraint_unique_blocks"):
            arrays[name][rows] = getattr(derived, name)
    return ConstraintTopology(topology.n_poses, **arrays)
