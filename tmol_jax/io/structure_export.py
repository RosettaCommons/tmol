"""Host structure export through the common molecular identity/bond policies."""

from types import SimpleNamespace
import numpy as np

from tmol_core.io.chain_deduction import chain_indices, connection_masks
from tmol_core.io.structure_export import atom_array_from_host_pose


def atom_array_from_pose_stack(
    pose_stack, co=None, merge_fragments=True, include_virtual_atoms=False
):
    """Export atoms, declared bonds, metadata and original ligand/metal identities.

    Export detaches coordinates to the host. Multi-model output requires identical
    topology and metadata; only coordinates may differ. The pose's chemistry is
    used for custom residue types, and virtual atoms are omitted by default.
    """
    from tmol_jax.io.canonical_ordering import (
        CanonicalOrdering,
        default_canonical_ordering,
    )

    pbt = pose_stack.packed_block_types
    if co is None:
        co = default_canonical_ordering()
        known = set(co.restype_io_equiv_classes)
        if any(bt.io_equiv_class not in known for bt in pbt.active_block_types):
            co = CanonicalOrdering.from_chemdb(pbt.chem_db)
    types = np.asarray(pose_stack.block_type_ind)
    connections = np.asarray(pose_stack.inter_residue_connections)
    chains = chain_indices(
        types, connections, connection_masks(pbt.active_block_types, pbt.max_n_conn)
    )
    host = SimpleNamespace(
        coords=np.asarray(pose_stack.coords),
        block_type_ind64=types,
        block_coord_offset=np.asarray(pose_stack.block_coord_offset),
        inter_residue_connections64=connections,
        packed_block_types=pbt,
        pdb_info=pose_stack.pdb_info,
        split_block_mapping=pose_stack.split_block_mapping,
    )
    return atom_array_from_host_pose(
        host,
        co,
        chains,
        merge_fragments=merge_fragments,
        include_virtual_atoms=include_virtual_atoms,
    )
