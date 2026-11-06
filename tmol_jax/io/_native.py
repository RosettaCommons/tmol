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
