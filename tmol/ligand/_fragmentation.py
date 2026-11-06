"""Torch pose assembly over shared ligand fragment chemistry."""

import numpy as np
import biotite.structure as struc
from tmol_core.ligand import _fragmentation as _core
from dataclasses import replace
from tmol_core.ligand._fragmentation import FragmentedLigandPoseMapping


def __getattr__(name):
    return getattr(_core, name)


def _apply_fragment_connections(pose_stack, mapping: FragmentedLigandPoseMapping):
    """Install fragment cut bonds and rebuild all inter-block bond separations."""

    resolved_blocks = []
    temporary_to_actual: dict[int, int] = {}
    for record in mapping.blocks:
        candidates = np.flatnonzero(
            pose_stack.pdb_info.residue_labels[record.pose_index]
            == record.pose_residue_label
        ).tolist()
        if len(candidates) != 1:
            raise ValueError(
                f"Could not uniquely map {record.fragment_name} "
                f"{record.chain_label}:{record.residue_label}{record.insertion_code}; "
                f"found blocks {candidates}"
            )
        actual_index = candidates[0]
        block_type_index = int(
            pose_stack.block_type_ind64[record.pose_index, actual_index].item()
        )
        block_type = pose_stack.packed_block_types.active_block_types[block_type_index]
        if block_type.name != record.fragment_name:
            raise ValueError(
                f"Mapped residue label {record.pose_residue_label} to "
                f"{block_type.name}, expected {record.fragment_name}"
            )
        previous_index = temporary_to_actual.setdefault(
            record.block_index, actual_index
        )
        if previous_index != actual_index:
            raise ValueError(
                "Fragmented AtomArrayStack models must have identical block "
                "topology; fragment block indices differ between models"
            )
        resolved_blocks.append(replace(record, block_index=actual_index))

    mapping = FragmentedLigandPoseMapping(
        blocks=tuple(resolved_blocks),
        connection_pairs=tuple(
            (
                temporary_to_actual[block_a],
                name_a,
                temporary_to_actual[block_b],
                name_b,
            )
            for block_a, name_a, block_b, name_b in mapping.connection_pairs
        ),
    )
    if not mapping.connection_pairs:
        pose_stack, sbm = _build_split_block_mapping(pose_stack, mapping)
        import attr as _attr

        return _attr.evolve(pose_stack, split_block_mapping=sbm)

    import attr
    import torch

    from tmol.pose._pose_stack_builder import PoseStackBuilder

    pbt = pose_stack.packed_block_types
    inter_residue_connections64 = pose_stack.inter_residue_connections64.clone()
    for pose_index in range(len(pose_stack)):
        for block_a, name_a, block_b, name_b in mapping.connection_pairs:
            type_a = int(pose_stack.block_type_ind64[pose_index, block_a].item())
            type_b = int(pose_stack.block_type_ind64[pose_index, block_b].item())
            restype_a = pbt.active_block_types[type_a]
            restype_b = pbt.active_block_types[type_b]
            conn_a = int(restype_a.connection_to_cidx[name_a])
            conn_b = int(restype_b.connection_to_cidx[name_b])
            if torch.any(
                inter_residue_connections64[pose_index, block_a, conn_a] != -1
            ):
                raise ValueError(
                    f"fragment connection {block_a}:{name_a} is already occupied"
                )
            if torch.any(
                inter_residue_connections64[pose_index, block_b, conn_b] != -1
            ):
                raise ValueError(
                    f"fragment connection {block_b}:{name_b} is already occupied"
                )
            inter_residue_connections64[pose_index, block_a, conn_a] = torch.tensor(
                (block_b, conn_b), dtype=torch.int64, device=pose_stack.device
            )
            inter_residue_connections64[pose_index, block_b, conn_b] = torch.tensor(
                (block_a, conn_a), dtype=torch.int64, device=pose_stack.device
            )

    real_res = pose_stack.block_type_ind64 >= 0
    inter_block_bondsep = PoseStackBuilder._inter_block_bondsep_from_connections(
        pbt, pose_stack.block_type_ind64, real_res, inter_residue_connections64
    )
    result = attr.evolve(
        pose_stack,
        coords=pose_stack.coords.clone(),
        inter_residue_connections=inter_residue_connections64.to(torch.int32),
        inter_residue_connections64=inter_residue_connections64,
        inter_block_bondsep=inter_block_bondsep,
    )
    result, sbm = _build_split_block_mapping(result, mapping)
    return attr.evolve(result, split_block_mapping=sbm)


def _build_split_block_mapping(
    pose_stack,
    resolved_mapping: FragmentedLigandPoseMapping,
):
    """Build a SplitBlockMapping from a fragmented PoseStack.

    Ensures the original (unfragmented) block types are present in the
    PackedBlockTypes of the returned PoseStack, then records for each
    fragment block: its pose/block indices, its group within that pose,
    the index of the original block type, and the per-atom mapping
    split_atom → orig_atom.

    Fragment block-type names are expected to follow the convention
    ``"ORIGNAME.FRAGMENT_ID"`` (e.g. ``"LIG.0"``).

    Returns the (possibly PBT-extended) PoseStack and the SplitBlockMapping.
    """
    import attr as _attr

    from tmol.pose._packed_block_types import PackedBlockTypes
    from tmol.pose._split_block_mapping import SplitBlockEntry, SplitBlockMapping

    pbt = pose_stack.packed_block_types
    active_name_to_idx = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}

    # ── collect original types that are not yet in the PBT ──────────────────
    missing_orig: dict[str, object] = {}  # name → RefinedResidueType
    for record in resolved_mapping.blocks:
        frag_name = pbt.active_block_types[
            int(
                pose_stack.block_type_ind64[
                    record.pose_index, record.block_index
                ].item()
            )
        ].name
        orig_name = frag_name.rsplit(".", 1)[0]
        if orig_name not in active_name_to_idx and orig_name not in missing_orig:
            orig_rt = next(
                (rt for rt in pbt.restype_set.residue_types if rt.name == orig_name),
                None,
            )
            if orig_rt is None:
                raise ValueError(
                    f"Original block type {orig_name!r} not found in ResidueTypeSet"
                )
            missing_orig[orig_name] = orig_rt

    if missing_orig:
        extended = list(pbt.active_block_types) + list(missing_orig.values())
        new_pbt = PackedBlockTypes.from_restype_list(
            pbt.chem_db, pbt.restype_set, extended, pose_stack.device
        )
        pose_stack = _attr.evolve(pose_stack, packed_block_types=new_pbt)
        pbt = new_pbt
        active_name_to_idx = {bt.name: i for i, bt in enumerate(pbt.active_block_types)}

    # ── build entries ────────────────────────────────────────────────────────
    group_ind_map: dict[tuple[int, str], int] = {}
    pose_group_counter: dict[int, int] = {}
    entries = []

    for record in resolved_mapping.blocks:
        pose_ind = record.pose_index
        block_ind = record.block_index
        bt_idx = int(pose_stack.block_type_ind64[pose_ind, block_ind].item())
        frag_bt = pbt.active_block_types[bt_idx]
        orig_name = frag_bt.name.rsplit(".", 1)[0]

        key = (pose_ind, orig_name)
        if key not in group_ind_map:
            pose_group_counter.setdefault(pose_ind, 0)
            group_ind_map[key] = pose_group_counter[pose_ind]
            pose_group_counter[pose_ind] += 1
        group_ind = group_ind_map[key]

        orig_bt_ind = active_name_to_idx[orig_name]
        orig_bt = pbt.active_block_types[orig_bt_ind]
        orig_atom_to_idx = orig_bt.atom_to_idx  # {name: local_index}

        split_to_orig = np.array(
            [orig_atom_to_idx[atom.name] for atom in frag_bt.atoms],
            dtype=np.int32,
        )

        entries.append(
            SplitBlockEntry(
                pose_ind=pose_ind,
                block_ind=block_ind,
                group_ind=group_ind,
                orig_block_type_ind=orig_bt_ind,
                split_to_orig_atom_inds=split_to_orig,
                orig_residue_label=record.residue_label,
                orig_chain_label=record.chain_label,
                orig_ins_code=record.insertion_code,
            )
        )

    return pose_stack, SplitBlockMapping(entries=tuple(entries))


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
    n_poses = len(pose_stack)
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
    """Gather atom coordinates from split blocks into original block layout."""
    old_coords = pose_stack.coords.cpu().numpy()
    old_bco = pose_stack.block_coord_offset.cpu().numpy()
    new_bco_np = new_bco.cpu().numpy()
    out = np.zeros((n_poses, new_max_n_atoms, 3), dtype=np.float32)
    for p, blocks in enumerate(per_pose_blocks):
        for new_b, (bt_idx, kind, src) in enumerate(blocks):
            new_off = int(new_bco_np[p, new_b])
            n_atoms = int(pbt.n_atoms[bt_idx].item())
            if kind == "orig":
                old_off = int(old_bco[p, src])
                out[p, new_off : new_off + n_atoms] = old_coords[
                    p, old_off : old_off + n_atoms
                ]
            else:
                dest = np.zeros((n_atoms, 3), dtype=np.float32)
                for fe in groups[src]:
                    fo = int(old_bco[p, fe.block_ind])
                    for li, oi in enumerate(fe.split_to_orig_atom_inds):
                        dest[oi] = old_coords[p, fo + li]
                out[p, new_off : new_off + n_atoms] = dest
    return out, new_bco_np


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
    import torch

    max_n_conn = int(pose_stack.inter_residue_connections64.shape[2])
    out = torch.full(
        (n_poses, new_max_n_blocks, max_n_conn, 2), -1, dtype=torch.int64, device=device
    )
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
                    out[p, new_b, c] = torch.tensor(
                        [npb, pc], dtype=torch.int64, device=device
                    )
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
    import torch
    from tmol.pose._pdb_info import (
        PDBInfo,
        DEFAULT_ATOM_OCCUPANCY,
        DEFAULT_ATOM_B_FACTOR,
    )

    old_pdb = pose_stack.pdb_info
    old_bco = pose_stack.block_coord_offset.cpu().numpy()
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
    old_chain = pose_stack.chain_id.cpu().numpy()
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
    new_chain_id = torch.from_numpy(chain_np).to(device)
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


def _unsplit_pose_stack(pose_stack):
    """Reconstruct an unsplit PoseStack from one containing split (fragment) blocks.

    Each group of split blocks sharing a ``(pose_ind, group_ind)`` in the
    PoseStack's ``split_block_mapping`` is collapsed back into a single block
    of the corresponding original block type.  Atom coordinates are gathered
    from the fragment blocks using the per-entry ``split_to_orig_atom_inds``
    arrays; atoms present in the original type but absent from all fragments
    are left at zero.

    Inter-residue connections *between* fragment blocks are removed (they
    become intra-block interactions in the original).  External connections
    from a fragment block to a non-fragment block are transferred to the
    reconstructed original block.

    Returns a new PoseStack with ``split_block_mapping=None``.
    """
    import attr
    import torch
    from tmol.pose._pose_stack_builder import PoseStackBuilder
    from tmol.utility.tensor._common_operations import _exclusive_cumsum2d

    sbm = pose_stack.split_block_mapping
    if sbm is None or not sbm.entries:
        return attr.evolve(pose_stack, split_block_mapping=None)

    pbt = pose_stack.packed_block_types
    device = pose_stack.device
    n_poses = len(pose_stack)

    groups, split_block_set, entry_lookup = _unsplit_group_entries(sbm)
    per_pose_blocks = _unsplit_per_pose_blocks(
        pose_stack, split_block_set, entry_lookup
    )

    new_max_n_blocks = max((len(bl) for bl in per_pose_blocks), default=0)
    new_bt64 = torch.full(
        (n_poses, new_max_n_blocks), -1, dtype=torch.int64, device=device
    )
    for p, blocks in enumerate(per_pose_blocks):
        for new_b, (bt_idx, *_) in enumerate(blocks):
            new_bt64[p, new_b] = bt_idx
    new_bt32 = new_bt64.to(torch.int32)

    real_new = new_bt64 >= 0
    new_n_atoms_blk = torch.zeros(
        (n_poses, new_max_n_blocks), dtype=torch.int32, device=device
    )
    new_n_atoms_blk[real_new] = pbt.n_atoms[new_bt64[real_new]]
    new_bco = _exclusive_cumsum2d(new_n_atoms_blk)
    new_max_n_atoms = int(torch.max(torch.sum(new_n_atoms_blk, dim=1)).item())

    coords_np, new_bco_np = _unsplit_build_coords(
        per_pose_blocks, pbt, groups, pose_stack, new_bco, new_max_n_atoms, n_poses
    )
    new_coords = torch.from_numpy(coords_np).to(device)

    old_to_new = _unsplit_old_to_new(
        per_pose_blocks, pose_stack, split_block_set, entry_lookup
    )
    new_irc64 = _unsplit_connections(
        pose_stack,
        pbt,
        n_poses,
        new_max_n_blocks,
        split_block_set,
        entry_lookup,
        old_to_new,
        device,
    )

    new_ibs = PoseStackBuilder._inter_block_bondsep_from_connections(
        pbt, new_bt64, real_new, new_irc64
    )

    new_chain_id, new_pdb_info = _unsplit_chain_and_pdb(
        pose_stack,
        old_to_new,
        pbt,
        per_pose_blocks,
        new_bco_np,
        n_poses,
        new_max_n_blocks,
        new_max_n_atoms,
        device,
    )

    return attr.evolve(
        pose_stack,
        coords=new_coords,
        block_coord_offset=new_bco,
        block_coord_offset64=new_bco.to(torch.int64),
        inter_residue_connections=new_irc64.to(torch.int32),
        inter_residue_connections64=new_irc64,
        inter_block_bondsep=new_ibs,
        block_type_ind=new_bt32,
        block_type_ind64=new_bt64,
        chain_id=new_chain_id,
        chain_id64=new_chain_id.to(torch.int64),
        pdb_info=new_pdb_info,
        split_block_mapping=None,
    )


def recombine_fragmented_ligands(
    structure: struc.AtomArray | struc.AtomArrayStack,
    pose_stack,
) -> struc.AtomArray | struc.AtomArrayStack:
    """Restore original residue identities on exported ligand fragments.

    Uses ``pose_stack.split_block_mapping`` to identify which residues in
    *structure* are split-block fragments and what their original PDB identity
    should be.  Atoms are matched by the residue label stored in the pose's
    PDBInfo (which is what ``atom_array_from_pose_stack`` writes to ``res_id``).
    """
    sbm = pose_stack.split_block_mapping
    if sbm is None or not sbm.entries:
        return structure

    pbt = pose_stack.packed_block_types
    result = structure.copy()

    for entry in sbm.entries:
        if entry.pose_ind != 0:
            continue
        pose_res_label = int(
            pose_stack.pdb_info.residue_labels[entry.pose_ind, entry.block_ind]
        )
        frag_name = pbt.active_block_types[
            int(pose_stack.block_type_ind[entry.pose_ind, entry.block_ind])
        ].name
        orig_name = pbt.active_block_types[entry.orig_block_type_ind].name

        fragment_atoms = (result.res_id == pose_res_label) & (
            result.res_name == frag_name
        )
        result.res_name[fragment_atoms] = orig_name
        result.res_id[fragment_atoms] = entry.orig_residue_label
        result.chain_id[fragment_atoms] = entry.orig_chain_label
        result.ins_code[fragment_atoms] = entry.orig_ins_code
    return result
