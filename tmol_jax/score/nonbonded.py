"""Runtime packing for the shared whole-pose LJ/LK/electrostatic kernels."""

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.atom_types import atom_type_parameters
from tmol_core.scoring.count_pair import packed_ljlk_distances
from tmol_core.scoring.elec import prepare_electrostatic_parameters
from tmol_core.scoring.ljlk import (
    ljlk_type_parameters,
    native_ljlk_global_parameters,
)
from tmol_jax.score.energy_term import prepared_palette
from tmol_jax.score._native import (
    coordinate_energy,
    pose_arguments,
    register,
    workspace_sizes,
)


class NonbondedState(NamedTuple):
    parameters: tuple
    reach: float
    precise_gradient: bool


class IndependentNonbondedState(NamedTuple):
    target: str
    channels: int
    parameters: tuple
    reach: float
    precise_gradient: bool


# Arrays are dynamic operands; native target, layout and cutoff are static.
# Named tuples retain existing positional state access and unpacking.
jax.tree_util.register_pytree_node(
    NonbondedState,
    lambda state: ((state.parameters,), (state.reach, state.precise_gradient)),
    lambda metadata, children: NonbondedState(children[0], *metadata),
)
jax.tree_util.register_pytree_node(
    IndependentNonbondedState,
    lambda state: (
        (state.parameters,),
        (state.target, state.channels, state.reach, state.precise_gradient),
    ),
    lambda metadata, children: IndependentNonbondedState(
        *metadata[:2], children[0], *metadata[2:]
    ),
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


def _validate_atom_type_order(pbt, names):
    if tuple(names[:-1]) != tuple(at.name for at in pbt.chem_db.atom_types):
        raise ValueError("scoring database atom-type order differs from pose chemistry")


def _ljlk_parameters(pose, database, *, soft_repulsive=False):
    pbt = pose.packed_block_types
    names, flags = atom_type_parameters(database.chemical.atom_types)
    _validate_atom_type_order(pbt, names)
    lj = ljlk_type_parameters(names, database.scoring.ljlk)
    active = np.asarray(pose.block_type_ind)
    for index in np.unique(active[active >= 0]):
        lj.validate_block_type(pbt.active_block_types[index])
    return (
        lj.native_parameters(flags),
        native_ljlk_global_parameters(
            database.scoring.ljlk.global_parameters, soft_repulsive=soft_repulsive
        ),
    )


def _electrostatic_parameters(pose, database):
    pbt = pose.packed_block_types
    types = pbt.active_block_types
    overrides = tuple(getattr(bt, "elec_partial_charges", None) for bt in types)
    return prepared_palette(
        pbt,
        database,
        "electrostatic_parameters",
        lambda: prepare_electrostatic_parameters(
            types,
            pbt.max_n_atoms,
            database.scoring.elec,
            database.scoring.genbonded.rosetta_typed,
            charge_overrides=overrides,
        ),
        settings=(
            tuple(
                None if row is None else tuple(sorted(row.items())) for row in overrides
            ),
        ),
    )


def _validate_native_connections(pbt):
    if pbt.max_n_conn > 12:
        raise NotImplementedError(
            "native nonbonded scoring supports at most 12 ports per residue"
        )


def _ljlk_distances(pose, database):
    pbt, device = pose.packed_block_types, pose.coords.device
    return prepared_palette(
        pbt,
        database,
        "ljlk_distances",
        lambda: jax.device_put(
            packed_ljlk_distances(
                pbt.active_block_types,
                pbt.max_n_atoms,
                database.scoring.genbonded.rosetta_typed,
            ),
            device,
        ),
        settings=(device,),
    )


def _nonbonded_topology(pose):
    _validate_native_connections(pose.packed_block_types)
    offsets, first, indices, _ = pose_arguments(pose)
    poses, blocks = indices.shape
    near, separation = _pose_connections(pose)
    return jax.device_put(
        (
            offsets,
            first,
            np.tile(np.arange(blocks, dtype=np.int32), poses),
            np.repeat(np.arange(poses, dtype=np.int32), blocks),
            indices,
            near,
            separation,
        ),
        device=pose.coords.device,
    )


def _nonbonded_chemistry(pose, database, *, electrostatic, soft_repulsive=False):
    """Share one device snapshot per family across whole and sparse scoring."""
    pbt, device = pose.packed_block_types, pose.coords.device
    if electrostatic:
        elec = _electrostatic_parameters(pose, database)
        generic = elec.all_ligand_typed
        specific = (elec.charges, elec.inter, elec.intra, elec.global_parameters)
        name, reach, sources = "elec_native", elec.global_parameters[0, 4], (elec,)
    else:
        parameters, globals_ = _ljlk_parameters(
            pose, database, soft_repulsive=soft_repulsive
        )
        paths, generic = _ljlk_distances(pose, database)
        specific = (pbt.atom_types, paths, parameters, globals_)
        name, reach, sources = "ljlk_native", globals_[0, 0], ()
    arrays = prepared_palette(
        pbt,
        database,
        name,
        lambda: jax.device_put(
            (pbt.n_atoms, pbt.n_conn, pbt.conn_atoms, generic, *specific), device
        ),
        device=device,
        settings=(soft_repulsive,),
        sources=sources,
    )
    return arrays, float(reach)


def prepare_nonbonded(pose, database, *, soft_repulsive=False):
    """Materialize canonical chemistry/parameters in the native operand layout."""
    topology = _nonbonded_topology(pose)
    ljlk, lj_reach = _nonbonded_chemistry(
        pose, database, electrostatic=False, soft_repulsive=soft_repulsive
    )
    elec, elec_reach = _nonbonded_chemistry(pose, database, electrostatic=True)
    n_atoms, n_conn, conn_atoms, generic, atom_types, paths, lj, globals_ = ljlk
    parameters = (
        *topology,
        n_atoms,
        atom_types,
        n_conn,
        conn_atoms,
        paths,
        generic,
        lj,
        globals_,
        *elec[4:],
    )
    precise_gradient = pose.coords.device.platform == "gpu"
    for target in ("tmol_nonbonded", "tmol_nonbonded_weighted"):
        register(target)
        if precise_gradient:
            register(target + "_precise")
    return NonbondedState(parameters, max(lj_reach, elec_reach), precise_gradient)


def nonbonded_energy(coords, state, weights=None):
    parameters, reach, precise_gradient = state
    target = "tmol_nonbonded" if weights is None else "tmol_nonbonded_weighted"
    if weights is not None:
        parameters = (*parameters, jnp.asarray(weights, dtype=coords.dtype))
    dimensions = parameters[1].shape
    if precise_gradient and coords.dtype == jnp.float32:
        target += "_precise"
        dimensions = (*dimensions, coords.shape[1])
    sizes = workspace_sizes(target, dimensions, coords.dtype.itemsize * 8)
    return coordinate_energy(
        target,
        coords,
        parameters,
        channels=4 if weights is None else 1,
        attributes={"reach": np.float64(reach)},
        workspace_bytes=sizes,
    )


def prepare_independent_nonbonded(
    pose, database, *, electrostatic, soft_repulsive=False
):
    topology = _nonbonded_topology(pose)
    chemical, reach = _nonbonded_chemistry(
        pose, database, electrostatic=electrostatic, soft_repulsive=soft_repulsive
    )
    if electrostatic:
        pbt = pose.packed_block_types
        # Inspection and older scorers retain the same frozen parameter snapshot.
        pbt.elec_partial_charge = chemical[4]
        pbt._elec_owners = (database.scoring.elec, pbt.active_block_types)
    target, channels = ("tmol_elec", 1) if electrostatic else ("tmol_ljlk", 3)
    register(target)
    precise_gradient = not electrostatic and pose.coords.device.platform == "gpu"
    if precise_gradient:
        register(target + "_precise")
    return IndependentNonbondedState(
        target, channels, (*topology, *chemical), reach, precise_gradient
    )


def independent_nonbonded_energy(coords, state):
    target, channels, parameters, reach, precise_gradient = state
    dimensions = parameters[1].shape
    if precise_gradient and coords.dtype == jnp.float32:
        target += "_precise"
        dimensions += coords.shape[1:2]
    return coordinate_energy(
        target,
        coords,
        parameters,
        channels=channels,
        workspace_bytes=workspace_sizes(target, dimensions, coords.dtype.itemsize * 8),
        attributes={"reach": np.float64(reach)},
    )
