"""Runtime packing for the shared whole-pose LJ/LK/electrostatic kernels."""

import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.atom_types import atom_type_parameters
from tmol_core.scoring.count_pair import electrostatic_distances, packed_ljlk_distances
from tmol_core.scoring.elec import (
    count_pair_table,
    native_elec_global_parameters,
    partial_charge_table,
    representative_mapping_for_block,
)
from tmol_core.scoring.ljlk import (
    ljlk_type_parameters,
    native_ljlk_global_parameters,
)
from tmol_jax.score._native import (
    coordinate_energy,
    pose_arguments,
    register,
    workspace_sizes,
)
from tmol_jax.score.elec.params import _charges_for_block_type


def _compact_connections(dense):
    """Pack existing capped graph distances into the native sparse slot layout.

    The frontend still owns a dense graph here. This is only an ABI layout
    conversion; shared graph preparation will remove that setup allocation.
    """
    dense = np.asarray(dense)
    poses, blocks, _, ports, _ = dense.shape
    minima = dense.min(axis=(-1, -2)) if ports else np.full(dense.shape[:3], 6)
    counts = np.sum(minima < 6, axis=2)
    slots = int(counts.max(initial=0)) + 1
    near = np.full((poses, blocks, slots, 2), -1, np.int32)
    near[..., 1] = 6
    distances = np.full((poses, blocks, slots, ports, ports), 6, np.int8)
    for pose, block in np.ndindex(poses, blocks):
        partners = np.flatnonzero(minima[pose, block] < 6)
        n = len(partners)
        near[pose, block, :n, 0] = partners
        near[pose, block, :n, 1] = minima[pose, block, partners]
        distances[pose, block, :n] = np.minimum(dense[pose, block, partners], 6)
    return near, distances


def prepare_nonbonded(pose, database):
    """Materialize canonical chemistry/parameters in the native operand layout."""
    pbt = pose.packed_block_types
    types = pbt.active_block_types
    names, flags = atom_type_parameters(database.chemical.atom_types)
    lj = ljlk_type_parameters(names, database.scoring.ljlk)
    for block in types:
        lj.validate_block_type(block)
    charge_lookup = partial_charge_table(database.scoring.elec)
    representatives = count_pair_table(database.scoring.elec)
    rosetta_typed = database.scoring.genbonded.rosetta_typed
    charges = np.zeros((len(types), pbt.max_n_atoms), np.float32)
    inter = np.full((len(types), pbt.max_n_atoms, pbt.max_n_atoms), 6, np.int32)
    intra = np.full_like(inter, 6)
    for i, block in enumerate(types):
        n = block.n_atoms
        charges[i, :n] = _charges_for_block_type(block, charge_lookup)
        rep = representative_mapping_for_block(block, representatives)
        inter[i, :n, :n], intra[i, :n, :n] = electrostatic_distances(
            block, rep, rosetta_typed
        )
    lj_paths, generic = packed_ljlk_distances(types, pbt.max_n_atoms, rosetta_typed)
    offsets, first, indices, _ = pose_arguments(pose)
    poses, blocks = indices.shape
    near, separation = _compact_connections(pose.inter_block_bondsep)
    if pbt.max_n_conn > 12:
        raise NotImplementedError(
            "native nonbonded scoring supports at most 12 ports per residue"
        )
    # The canonical tile stores connection atom indices in uint8.
    if np.any(np.asarray(pbt.conn_atoms) > 255):
        raise NotImplementedError(
            "native nonbonded connection atom index exceeds uint8"
        )
    lj_global = native_ljlk_global_parameters(database.scoring.ljlk.global_parameters)
    elec_global = native_elec_global_parameters(database.scoring.elec.global_parameters)
    parameters = tuple(
        jnp.asarray(a)
        for a in (
            offsets,
            first,
            np.tile(np.arange(blocks, dtype=np.int32), poses),
            np.repeat(np.arange(poses, dtype=np.int32), blocks),
            indices,
            near,
            separation,
            pbt.n_atoms,
            pbt.atom_types,
            pbt.n_conn,
            pbt.conn_atoms,
            lj_paths,
            generic,
            lj.native_parameters(flags),
            lj_global,
            charges,
            inter,
            intra,
            elec_global,
        )
    )
    for target in ("tmol_nonbonded", "tmol_nonbonded_weighted"):
        register(target)
    return parameters, float(max(lj_global[0, 0], elec_global[0, 4]))


def nonbonded_energy(coords, state, weights=None):
    parameters, reach = state
    target = "tmol_nonbonded" if weights is None else "tmol_nonbonded_weighted"
    if weights is not None:
        parameters = (*parameters, jnp.asarray(weights, dtype=coords.dtype))
    sizes = workspace_sizes(target, parameters[1].shape, coords.dtype.itemsize * 8)
    return coordinate_energy(
        target,
        coords,
        parameters,
        channels=4 if weights is None else 1,
        attributes={"reach": np.float64(reach)},
        workspace_bytes=sizes,
    )
