"""Runtime packing for the shared whole-pose LJ/LK/electrostatic kernels."""

import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.atom_types import atom_type_parameters
from tmol_core.scoring.count_pair import packed_ljlk_distances
from tmol_core.scoring.elec import prepare_electrostatic_parameters
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


def _compact_connections(dense):
    """Pack existing capped graph distances into the native sparse slot layout.

    Compatibility for explicitly supplied dense topology. New pose preparation
    retains the common sparse layout and does not invoke this conversion.
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


def _pose_connections(pose):
    if pose._inter_block_bondsep is None:
        return pose.near_blocks, pose.bondsep
    return _compact_connections(pose.inter_block_bondsep)


def _ljlk_parameters(pose, database):
    pbt = pose.packed_block_types
    names, flags = atom_type_parameters(database.chemical.atom_types)
    lj = ljlk_type_parameters(names, database.scoring.ljlk)
    active = np.asarray(pose.block_type_ind)
    for index in np.unique(active[active >= 0]):
        lj.validate_block_type(pbt.active_block_types[index])
    return (
        lj.native_parameters(flags),
        native_ljlk_global_parameters(database.scoring.ljlk.global_parameters),
    )


def _electrostatic_parameters(pose, database):
    pbt = pose.packed_block_types
    types = pbt.active_block_types
    return prepare_electrostatic_parameters(
        types,
        pbt.max_n_atoms,
        database.scoring.elec,
        database.scoring.genbonded.rosetta_typed,
        charge_overrides=tuple(
            getattr(bt, "elec_partial_charges", None) for bt in types
        ),
    )


def _validate_native_connections(pbt):
    if pbt.max_n_conn > 12:
        raise NotImplementedError(
            "native nonbonded scoring supports at most 12 ports per residue"
        )


def prepare_nonbonded(pose, database):
    """Materialize canonical chemistry/parameters in the native operand layout."""
    pbt = pose.packed_block_types
    types = pbt.active_block_types
    lj, lj_global = _ljlk_parameters(pose, database)
    elec = _electrostatic_parameters(pose, database)
    lj_paths, generic = packed_ljlk_distances(
        types, pbt.max_n_atoms, database.scoring.genbonded.rosetta_typed
    )
    offsets, first, indices, _ = pose_arguments(pose)
    poses, blocks = indices.shape
    near, separation = _pose_connections(pose)
    _validate_native_connections(pbt)
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
            lj,
            lj_global,
            elec.charges,
            elec.inter,
            elec.intra,
            elec.global_parameters,
        )
    )
    for target in ("tmol_nonbonded", "tmol_nonbonded_weighted"):
        register(target)
    return parameters, float(max(lj_global[0, 0], elec.global_parameters[0, 4]))


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


def prepare_independent_nonbonded(pose, database, *, electrostatic):
    pbt = pose.packed_block_types
    _validate_native_connections(pbt)
    types = pbt.active_block_types
    rosetta_typed = database.scoring.genbonded.rosetta_typed
    offsets, first, indices, _ = pose_arguments(pose)
    poses, blocks = indices.shape
    near, separation = _pose_connections(pose)
    if electrostatic:
        elec = _electrostatic_parameters(pose, database)
        generic = elec.all_ligand_typed
        specific = (elec.charges, elec.inter, elec.intra, elec.global_parameters)
        target, channels, reach = "tmol_elec", 1, elec.global_parameters[0, 4]
        # Preserve existing palette charge inspection and immutable scorer snapshots.
        pbt.elec_partial_charge = jnp.asarray(elec.charges)
        pbt._elec_owners = (database.scoring.elec, types)
    else:
        parameters, globals_ = _ljlk_parameters(pose, database)
        paths, generic = packed_ljlk_distances(types, pbt.max_n_atoms, rosetta_typed)
        specific = (pbt.atom_types, paths, parameters, globals_)
        target, channels, reach = "tmol_ljlk", 3, globals_[0, 0]
    operands = (
        offsets,
        first,
        np.tile(np.arange(blocks, dtype=np.int32), poses),
        np.repeat(np.arange(poses, dtype=np.int32), blocks),
        indices,
        near,
        separation,
        pbt.n_atoms,
        pbt.n_conn,
        pbt.conn_atoms,
        generic,
        *specific,
    )
    register(target)
    return target, channels, tuple(jnp.asarray(a) for a in operands), float(reach)


def independent_nonbonded_energy(coords, state):
    target, channels, parameters, reach = state
    return coordinate_energy(
        target,
        coords,
        parameters,
        channels=channels,
        workspace_bytes=workspace_sizes(
            target, parameters[1].shape, coords.dtype.itemsize * 8
        ),
        attributes={"reach": np.float64(reach)},
    )
