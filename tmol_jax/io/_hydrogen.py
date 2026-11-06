"""JAX execution and derivatives for common native hydrogen geometry."""

from functools import partial

import jax
import numpy as np

from tmol_jax.score._native import register


@partial(jax.jit, static_argnames=("mode",))
def _stage(coords, distances, metadata, weights=None, directions=(), *, mode=0):
    register("tmol_hydrogen_completion")
    coordinate_shape = jax.ShapeDtypeStruct(coords.shape, coords.dtype)
    distance_shape = jax.ShapeDtypeStruct(distances.shape, distances.dtype)
    outputs = (coordinate_shape, distance_shape, coordinate_shape)[: mode + 1]
    args = (coords, *metadata[:9], distances, *metadata[9:])
    if mode:
        args += (weights, *directions)
    return jax.ffi.ffi_call(
        "tmol_hydrogen_completion", outputs, vmap_method="sequential"
    )(*args, operation=np.int64(mode), extent=np.int64(0), width=np.int64(0))


@jax.custom_vjp
def complete_hydrogens(coords, distances, metadata):
    """Reorient eligible rebuilt H, retaining coordinate and bond-length gradients.

    The shared native kernel supplies first and second reverse derivatives.
    Discrete topology/eligibility metadata is fixed. Higher derivatives and
    forward-mode differentiation are not supplied by this adapter.
    """
    return _stage(coords, distances, metadata)[0]


def _forward(coords, distances, metadata):
    return complete_hydrogens(coords, distances, metadata), (
        coords,
        distances,
        metadata,
    )


@jax.custom_vjp
def _gradient(coords, distances, weights, metadata):
    return tuple(_stage(coords, distances, metadata, weights, mode=1))


def _gradient_forward(coords, distances, weights, metadata):
    return _gradient(coords, distances, weights, metadata), (
        coords,
        distances,
        weights,
        metadata,
    )


def _gradient_backward(saved, directions):
    coords, distances, weights, metadata = saved
    derivatives = _stage(coords, distances, metadata, weights, directions, mode=2)
    return (*derivatives, None)


_gradient.defvjp(_gradient_forward, _gradient_backward)


def _backward(saved, weights):
    coords, distances, metadata = saved
    return (*_gradient(coords, distances, weights, metadata), None)


complete_hydrogens.defvjp(_forward, _backward)


@partial(jax.jit, static_argnames=("tolerance",))
def retained_hydrogens(coords, types, missing, parents, distances, *, tolerance):
    """Apply the shared observed-parent/distance test to named input hydrogens."""
    register("tmol_hydrogen_completion")
    return jax.ffi.ffi_call(
        "tmol_hydrogen_completion",
        jax.ShapeDtypeStruct(missing.shape, np.bool_),
        vmap_method="sequential",
    )(
        coords,
        types,
        missing,
        parents,
        distances,
        operation=np.int64(3),
        extent=np.float64(tolerance).view(np.int64),
        width=np.int64(0),
    )


def _completion_info(pbt):
    from tmol_core.io.missing_atoms import prepare_h_completion_metadata

    cached = getattr(pbt, "_hydrogen_completion_metadata", None)
    if cached is None:
        cached = prepare_h_completion_metadata(
            pbt.active_block_types, np.asarray(pbt.atom_is_hydrogen), pbt.max_n_atoms
        )
        object.__setattr__(pbt, "_hydrogen_completion_metadata", cached)
    return cached


def execution_metadata(pose, rebuilt):
    """Prepare common chemical metadata and arrange it for the native ABI."""
    from ._selection import _wide

    pbt = pose.packed_block_types
    host = _completion_info(pbt)
    metadata = (
        pose.block_type_ind,
        pose.block_coord_offset,
        pose.inter_residue_connections,
        pbt.conn_atoms,
        rebuilt,
        host.eligible,
        _wide(host.parent),
        _wide(host.neigh),
        _wide(host.n_neigh),
        _wide(host.par_conn),
        _wide(host.n_par_conn),
    )
    return jax.device_put((host.dist.astype(pose.coords.dtype), metadata), pose.device)


def apply_input_policy(pose, available, mode):
    """Retain valid generated input H or mark them for shared reconstruction."""
    from tmol_core.io.missing_atoms import (
        BONDED_H_TOLERANCE,
        prepare_hydrogen_parents,
        validate_ligand_hydrogens,
    )
    from ._selection import _wide

    validate_ligand_hydrogens(mode)
    pbt = pose.packed_block_types
    generated = np.asarray(
        [bool(t.hydrogens_regenerated) for t in pbt.active_block_types]
    )
    if not generated.any():
        return available
    types = np.asarray(pose.block_type_ind)
    offsets = np.asarray(pose.block_coord_offset)
    is_h = np.asarray(pbt.atom_is_hydrogen)
    shape = (*types.shape, pbt.max_n_atoms)
    missing = np.ones(shape, bool)
    block_coords = np.zeros((*shape, 3), pose.coords.dtype)
    coords = np.asarray(pose.coords)
    for p, row in enumerate(available):
        for r, mask in enumerate(row):
            missing[p, r, : len(mask)] = ~mask
            start = offsets[p, r]
            block_coords[p, r, : len(mask)] = coords[p, start : start + len(mask)]
    retained = np.zeros(shape, bool)
    if mode == "input":
        parents, distances = prepare_hydrogen_parents(
            pbt.active_block_types, is_h, pbt.max_n_atoms
        )
        args = jax.device_put(
            (
                block_coords,
                types,
                missing,
                _wide(parents),
                distances.astype(coords.dtype),
            ),
            pose.device,
        )
        retained = np.asarray(retained_hydrogens(*args, tolerance=BONDED_H_TOLERANCE))
    return [
        [
            mask
            & ~(
                generated[types[p, r]]
                & is_h[types[p, r], : len(mask)]
                & ~retained[p, r, : len(mask)]
            )
            for r, mask in enumerate(row)
        ]
        for p, row in enumerate(available)
    ]


def reorient_rebuilt_hydrogens(pose, rebuilt, coords):
    pbt = pose.packed_block_types
    host = _completion_info(pbt)
    types = np.asarray(pose.block_type_ind)
    eligible = host.eligible[np.maximum(types, 0)] & (types >= 0)[..., None]
    if not np.any(eligible & rebuilt):
        return coords
    distances, metadata = execution_metadata(pose, rebuilt)
    return complete_hydrogens(coords, distances, metadata)
