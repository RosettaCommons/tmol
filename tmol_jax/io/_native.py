"""Framework execution of common canonical selection and missing-atom kernels."""

from functools import partial

import jax
import numpy as np

from tmol_jax.score._native import register


@partial(jax.jit, static_argnames=("backward",))
def _leaf_stage(coords, metadata, weights=None, *, backward=False):
    register("tmol_leaf_atoms")
    arguments = (coords, *metadata) + ((weights,) if backward else ())
    return jax.ffi.ffi_call(
        "tmol_leaf_atoms",
        jax.ShapeDtypeStruct(coords.shape, coords.dtype),
        vmap_method="sequential",
    )(*arguments, operation=np.int64(backward), extent=np.int64(0), width=np.int64(0))


@jax.custom_vjp
def generate_leaf_atoms(coords, metadata):
    """Reconstruct from shared frames, preserving first coordinate gradients.

    Native reconstruction retains observed coordinates, zeroes padding and marks
    unbuildable atoms NaN for the input adapter to reject. Metadata includes the
    original missing masks, topology and primary/backup construction frames.
    """
    return _leaf_stage(coords, metadata)


def _leaf_forward(coords, metadata):
    return generate_leaf_atoms(coords, metadata), (coords, metadata)


def _leaf_backward(saved, weights):
    coords, metadata = saved
    return _leaf_stage(coords, metadata, weights, backward=True), None


generate_leaf_atoms.defvjp(_leaf_forward, _leaf_backward)


@jax.jit
def canonical_candidates(present, real, io_class, terminus, variant, metadata):
    """Execute common candidate ranking without temporary candidate/atom grids.

    Return selected types, exact int64 score bytes and candidate indices. The
    input adapter applies the shared warning/failure policy to the scores.
    """
    register("tmol_canonical_candidates")
    rows = real.shape
    return jax.ffi.ffi_call(
        "tmol_canonical_candidates",
        (
            jax.ShapeDtypeStruct(rows, np.int32),
            jax.ShapeDtypeStruct((*rows, 8), np.uint8),
            jax.ShapeDtypeStruct(rows, np.int32),
        ),
        vmap_method="sequential",
    )(
        present,
        real,
        io_class,
        terminus,
        variant,
        *metadata,
        operation=np.int64(0),
        extent=np.int64(0),
        width=np.int64(0),
    )


@jax.jit
def canonical_chain_endings(
    chains, types, disconnected, real, polymeric, ports, cycles, explicit_ports
):
    """Shared polymer/cap/cycle policy with fixed-size framework-owned outputs."""
    register("tmol_canonical_topology")
    shape = types.shape
    return jax.ffi.ffi_call(
        "tmol_canonical_topology",
        (
            jax.ShapeDtypeStruct(shape, np.int32),
            *(jax.ShapeDtypeStruct(shape, np.bool_) for _ in range(3)),
        ),
        vmap_method="sequential",
    )(
        chains,
        types,
        disconnected,
        real,
        polymeric,
        ports,
        cycles,
        explicit_ports,
        operation=np.int64(0),
        extent=np.int64(0),
        width=np.int64(0),
    )


@jax.jit
def canonical_atom_termini(present, types, variants, termini, broken, real, metadata):
    """Apply common atom-stated terminal chemistry without dropping heavy atoms."""
    register("tmol_canonical_topology")
    return jax.ffi.ffi_call(
        "tmol_canonical_topology",
        (
            jax.ShapeDtypeStruct(types.shape, np.int32),
            jax.ShapeDtypeStruct((*types.shape, 2, metadata[0].shape[3]), np.int32),
        ),
        vmap_method="sequential",
    )(
        present,
        types,
        variants,
        termini,
        broken,
        real,
        *metadata,
        operation=np.int64(1),
        extent=np.int64(0),
        width=np.int64(0),
    )[
        0
    ]


@jax.jit
def histidine_state(
    coords, types, variants, selected_poses, selected_residues, present, atom_indices
):
    """Common discrete tautomer decisions and atom alias remapping.

    Selected rows, remapping and state codes use opaque int64 transport; variant
    codes remain int32. Input arrays are never mutated. Decisions have no AD rule.
    """
    register("tmol_histidine")
    p, r, n, _ = coords.shape
    return jax.ffi.ffi_call(
        "tmol_histidine",
        (
            jax.ShapeDtypeStruct((p, r), np.int32),
            jax.ShapeDtypeStruct((p, r, n, 8), np.uint8),
            jax.ShapeDtypeStruct((p, r, 8), np.uint8),
        ),
        vmap_method="sequential",
    )(
        coords,
        types,
        variants,
        selected_poses,
        selected_residues,
        present,
        atom_indices,
        operation=np.int64(0),
        extent=np.int64(0),
        width=np.int64(0),
    )


@partial(jax.jit, static_argnames=("neutral_base",))
def canonical_variants(types, variants, termini, valid, *, neutral_base):
    """Shared neutral-terminus fallback, retaining the supplied state elsewhere."""
    register("tmol_canonical_policy")
    return jax.ffi.ffi_call(
        "tmol_canonical_policy",
        jax.ShapeDtypeStruct(types.shape, np.int32),
        vmap_method="sequential",
    )(
        types,
        variants,
        termini,
        valid,
        operation=np.int64(0),
        extent=np.int64(neutral_base),
        width=np.int64(0),
    )


@partial(jax.jit, static_argnames=("cutoff",))
def canonical_cycles(types, chains, up, down, coords, explicit, *, cutoff=2.0):
    """Common cycle search with a static positive distance cutoff in Angstroms."""
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cycle cutoff must be finite and positive")
    register("tmol_canonical_policy")
    return jax.ffi.ffi_call(
        "tmol_canonical_policy",
        jax.ShapeDtypeStruct(types.shape, np.int32),
        vmap_method="sequential",
    )(
        types,
        chains,
        up,
        down,
        coords,
        explicit,
        operation=np.int64(1),
        extent=np.float64(cutoff).view(np.int64),
        width=np.int64(0),
    )


@partial(jax.jit, static_argnames=("max_connections",))
def canonical_connections(
    types,
    polymeric,
    first,
    last,
    disconnected,
    up,
    down,
    disulfide_port,
    disulfides,
    cycles,
    explicit,
    *,
    max_connections,
):
    """Execute common port wiring and return its per-pose validation status."""
    register("tmol_canonical_policy")
    return jax.ffi.ffi_call(
        "tmol_canonical_policy",
        (
            jax.ShapeDtypeStruct((*types.shape, max_connections, 2), np.int32),
            jax.ShapeDtypeStruct((types.shape[0],), np.int32),
        ),
        vmap_method="sequential",
    )(
        types,
        polymeric,
        first,
        last,
        disconnected,
        up,
        down,
        disulfide_port,
        disulfides,
        cycles,
        explicit,
        operation=np.int64(2),
        extent=np.int64(0),
        width=np.int64(max_connections),
    )


@jax.jit
def histidine_coordinates(coords, types, variants, poses, residues, present, atoms):
    """Execute common tautomer decisions and apply their atom permutation on device.

    Canonical atom indices fit int32; read that low word from the opaque int64
    transport without enabling global JAX x64 mode. Input arrays remain immutable.
    """
    variants, mapping, _ = histidine_state(
        coords, types, variants, poses, residues, present, atoms
    )
    indices = jax.lax.bitcast_convert_type(mapping[..., :4], np.int32)
    return (
        variants,
        jax.numpy.take_along_axis(coords, indices[..., None], axis=2),
        jax.numpy.take_along_axis(present, indices, axis=2),
    )
