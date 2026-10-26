"""Optional CUDA FFI implementation of the LK-ball pair sum."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from tmol_jax.score._cuda import pair_sum, register_ffi_targets

_PARAMETER_COUNT = 9


def available() -> bool:
    """Return whether this installation contains its optional CUDA kernel."""
    return register_ffi_targets("tmol_jax_lkball")


def _pack_parameters(term, state):
    coefficient = -state.lk_dgfree / (jnp.float32(2 * 5.56832799683) * state.lk_lambda)
    packed = jnp.stack(
        (
            state.lj_radius,
            coefficient,
            1 / (state.lk_lambda * state.lk_lambda),
            state.lk_volume,
            state.is_donor,
            state.is_hydroxyl,
            state.is_polarh,
            state.is_acceptor,
            state.is_carbon_lk,
        ),
        axis=-1,
        dtype=jnp.float32,
    )
    assert packed.shape[-1] == _PARAMETER_COUNT
    defaults = term.defaults
    global_values = jnp.asarray(
        [
            defaults.lj_hbond_dis,
            defaults.lj_hbond_OH_donor_dis,
            defaults.lj_hbond_hdis,
            defaults.lkb_water_dist,
            defaults.distance_threshold,
        ],
        dtype=jnp.float32,
    )
    return packed, global_values


def score(term, state, coordinates):
    """Evaluate four LK-ball channels with the installed CUDA kernel."""

    def place_one(x, donor, acceptor, polar, base, base0, hybrid, hydrogens, mask, n):
        return term._place_all_waters(
            x, donor, acceptor, polar, base, base0, hybrid, hydrogens, mask, n
        )

    def place(x):
        return jax.vmap(place_one)(
            x,
            state.is_donor,
            state.is_acceptor,
            state.is_polar,
            state.B_idx,
            state.B0_idx,
            state.acc_hybridization,
            state.attached_H_idx,
            state.attached_H_mask,
            state.n_attached_H,
        )

    parameters, global_values = _pack_parameters(term, state)
    return pair_sum(
        "tmol_jax_lkball",
        (coordinates, place(coordinates)),
        (
            parameters,
            jnp.asarray(state.bonded_path_length, dtype=jnp.uint8),
            jnp.asarray(state.polar_indices, dtype=jnp.int32),
            jnp.asarray(state.occluder_indices, dtype=jnp.int32),
            global_values,
        ),
        channels=4,
    )


__all__ = ["available", "score"]
