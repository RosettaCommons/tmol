"""Optional CUDA FFI implementation of the hydrogen-bond pair sum."""

from __future__ import annotations

import jax.numpy as jnp

from tmol_jax.score._cuda import pair_sum, register_ffi_targets


def available() -> bool:
    """Return whether this installation contains its optional CUDA kernel."""
    return register_ffi_targets("tmol_jax_hbond")


def _pack_parameters(state):
    atoms = state.atom_params
    donors = jnp.stack(
        (atoms.donor_H_atom, atoms.donor_heavy_atom, atoms.donor_type),
        axis=-1,
        dtype=jnp.int32,
    )
    acceptors = jnp.stack(
        (
            atoms.acceptor_atom,
            atoms.acceptor_base,
            atoms.acceptor_base0,
            atoms.acceptor_type,
        ),
        axis=-1,
        dtype=jnp.int32,
    )
    pairs = jnp.stack(
        (
            state.pair_params.weight,
            state.pair_params.acceptor_hybridization.astype(jnp.float32),
        ),
        axis=-1,
        dtype=jnp.float32,
    )
    globals_ = state.global_params
    global_values = jnp.stack(
        (
            globals_.hb_sp2_range_span,
            globals_.hb_sp2_BAH180_rise,
            globals_.hb_sp2_outer_width,
            globals_.hb_sp3_softmax_fade,
            globals_.threshold_distance,
            globals_.max_ah_dist,
        ),
        dtype=jnp.float32,
    )
    return donors, acceptors, pairs, global_values


def score(state, coordinates):
    """Evaluate the hydrogen-bond channel with the installed CUDA kernel."""
    donors, acceptors, pairs, global_values = _pack_parameters(state)
    return pair_sum(
        "tmol_jax_hbond",
        (coordinates,),
        (
            donors,
            acceptors,
            jnp.asarray(state.donor_acceptor_bond_sep, dtype=jnp.int32),
            pairs,
            jnp.asarray(state.polynomials.native_poly_words, dtype=jnp.uint32),
            global_values,
        ),
    )


__all__ = ["available", "score"]
