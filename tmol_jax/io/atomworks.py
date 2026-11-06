"""Standard AtomWorks encoding over shared chemistry and Atom37 execution."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.io.atomworks_encoding import (
    ATOMWORKS_NAME3S,
    atomworks_parameter_database,
    atomworks_restype_map,
    canonical_to_atomworks_mappings,
)
from .atom37 import prepare_atom37_tensor_builder


@lru_cache(maxsize=8)
def _protein_context(device):
    from tmol_jax.database import ParameterDatabase
    from tmol_jax.pose.packed_block_types import PackedBlockTypes
    from .canonical_ordering import CanonicalOrdering
    from .structure import AtomArrayBuildContext

    database = atomworks_parameter_database(ParameterDatabase.get_default())
    with jax.default_device(device):
        palette = PackedBlockTypes.default(database.chemical)
    return AtomArrayBuildContext(
        database, palette, CanonicalOrdering.from_chemdb(database.chemical)
    )


def prepare_canonical_aa_atom37_builder(residue_type, chain_iid, device=None):
    """Bind standard 1..20 protein tokens (0 padding) before JAX transforms.

    Arbitrary chemistry uses ``prepare_atom37_tensor_builder`` with its full
    context and slot map. The restricted encoding cannot identify such residues.
    """
    from .structure import _device

    tokens = np.asarray(residue_type)
    if not np.issubdtype(tokens.dtype, np.integer) or np.any(
        (tokens < 0) | (tokens > ATOMWORKS_NAME3S.index("VAL"))
    ):
        raise ValueError("residue_type must contain protein tokens 1..20 or padding 0")
    context = _protein_context(_device(device))
    types = atomworks_restype_map(context.canonical_ordering)[tokens]
    builder = prepare_atom37_tensor_builder(types, chain_iid, context)
    # The mature canonical convenience API detects geometric chain closures;
    # the topology-supplied API deliberately fixes explicit connectivity instead.
    builder.template.update(
        find_additional_disulfides=True, find_additional_cyclic_closures=True
    )
    return builder


def pose_stack_from_canonical_aa_atom37(coords, residue_type, chain_iid):
    """Construct eagerly from the standard protein-only AtomWorks encoding."""
    return prepare_canonical_aa_atom37_builder(residue_type, chain_iid, coords.device)(
        coords
    )


def atomworks_from_pose_stack(pose_stack):
    """Export standard protein slots; coordinate derivatives remain connected.

    Unused slots are zero, padding tokens are zero, and coordinates are float32,
    matching the standard encoding. Topology is prepared eagerly; JAX integer
    metadata uses its normal int32 storage. Nonprotein identities are rejected.
    """
    from tmol_core.io.canonical import _create_bt_to_canonical_atom_mapping
    from tmol_core.io.chain_deduction import chain_indices, connection_masks
    from .canonical_ordering import CanonicalOrdering

    pose = pose_stack
    pbt = pose.packed_block_types
    co = CanonicalOrdering.from_chemdb(pbt.chem_db)
    classes, atoms = _create_bt_to_canonical_atom_mapping(co, pbt)
    restypes, slots, valid = canonical_to_atomworks_mappings(co)
    types = np.asarray(pose.block_type_ind)
    safe = np.maximum(types, 0)
    tokens = np.where(types >= 0, restypes[classes[safe]], 0)
    if np.any((types >= 0) & (tokens == 0)):
        raise ValueError(
            "AtomWorks protein encoding cannot represent nonprotein residues"
        )
    local = np.arange(pbt.max_n_atoms)
    present = (types[..., None] >= 0) & (local < np.asarray(pbt.n_atoms)[safe, None])
    p, b, a = np.nonzero(present)
    canonical = atoms[types[p, b], a]
    selected = valid[classes[types[p, b]], canonical]
    p, b, a, canonical = (x[selected] for x in (p, b, a, canonical))
    target = slots[classes[types[p, b]], canonical]
    source = np.asarray(pose.block_coord_offset)[p, b] + a
    coords = (
        jax.device_put(np.zeros((*types.shape, 37, 3), np.float32), pbt.n_atoms.device)
        .at[p, b, target]
        .set(pose.coords[p, source].astype(jnp.float32))
    )
    chains = chain_indices(
        types,
        np.asarray(pose.inter_residue_connections),
        connection_masks(pbt.active_block_types, pbt.max_n_conn),
    )
    return (
        coords,
        jax.device_put(tokens.astype(np.int32), pbt.n_atoms.device),
        jax.device_put(chains.astype(np.int32), pbt.n_atoms.device),
    )
