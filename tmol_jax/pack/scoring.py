"""Sparse conformer value/VJP execution with shared nonbonded parameters."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register
from tmol_jax.score.nonbonded import prepare_independent_nonbonded
from .dispatch import rotamer_spheres, block_spheres, candidate_pairs


@partial(jax.jit, static_argnames=("target", "channels", "reach", "backward"))
def _evaluate(
    coords,
    parameters,
    indices,
    weights=None,
    *,
    target,
    channels,
    reach,
    backward=False,
):
    register(target)
    operands = (coords, *parameters, indices)
    if backward:
        operands += (weights,)
    shape = coords.shape if backward else (channels, indices.shape[1])
    return jax.ffi.ffi_call(
        target, jax.ShapeDtypeStruct(shape, coords.dtype), vmap_method="sequential"
    )(
        *operands,
        operation=np.int64(backward),
        extent=np.int64(0),
        width=np.int64(0),
        reach=np.float64(reach),
    )


def coordinate_vjp(coords, parameters, evaluate):
    """Apply a native coordinate reverse rule with fixed chemistry parameters."""
    parameters = tuple(
        jax.lax.stop_gradient(
            p.astype(coords.dtype) if jnp.issubdtype(p.dtype, jnp.floating) else p
        )
        for p in parameters
    )

    @jax.custom_vjp
    def value(x, *args):
        return evaluate(x, args)

    def forward(x, *args):
        return value(x, *args), (x, args)

    def backward(residual, cotangent):
        x, args = residual
        gradient = evaluate(x, args, cotangent, backward=True)
        return (gradient, *(None for _ in args))

    value.defvjp(forward, backward)
    return value(coords, *parameters)


def sparse_nonbonded_values(coords, state, indices):
    """Score explicit sparse candidate identities with a native coordinate VJP.

    The caller supplies current discrete candidates. Only coordinate derivatives
    are defined; chemistry/parameter tables and candidate identities are fixed.
    Values and reverse-mode derivatives support jit/vmap. Forward-mode and
    higher derivatives require additional native rules and are not implemented.
    """
    target, channels, parameters, reach = state

    def evaluate(x, args, weights=None, *, backward=False):
        return _evaluate(
            x,
            args,
            indices,
            weights,
            target=target,
            channels=channels,
            reach=reach,
            backward=backward,
        )

    return coordinate_vjp(coords, parameters, evaluate)


def prepare_rotamer_nonbonded(pose, rotamers, database, *, electrostatic=False):
    """Prepare shared chemistry for sparse LJ/LK or electrostatic execution."""
    if rotamers.context is None:
        raise ValueError("sparse scoring requires conformers with pose provenance")
    rotamers.context.validate(pose)
    if rotamers.device != pose.device:
        raise ValueError("conformers and pose must be on the same device")
    target, channels, parameters, reach = prepare_independent_nonbonded(
        pose,
        database,
        electrostatic=electrostatic,
    )
    common = (
        rotamers.coord_offset_for_rot,
        rotamers.block_ind_for_rot,
        rotamers.pose_for_rot,
        rotamers.block_type_ind_for_rot,
        rotamers.first_rot_block_type,
        rotamers.rot_offset_for_block,
    )
    return (
        target.replace("tmol_", "tmol_rotamer_", 1),
        channels,
        (*common, *parameters[5:]),
        reach,
    )


def nonbonded_candidate_pairs(
    rotamers, packed_block_types, *, reach, coords=None, begin=0, count=None
):
    """Recompute sparse candidates for the supplied conformer coordinates."""
    if (
        rotamers.context is None
        or rotamers.context.packed_block_types is not packed_block_types
    ):
        raise ValueError("conformers belong to a different packed atom layout")
    coords = rotamers.coords if coords is None else coords
    spheres = rotamer_spheres(
        coords,
        rotamers.coord_offset_for_rot,
        rotamers.block_type_ind_for_rot,
        packed_block_types.n_atoms,
    )
    sites = block_spheres(
        spheres, rotamers.n_rots_for_block, rotamers.rot_offset_for_block
    )
    return candidate_pairs(
        rotamers.first_rot_block_type,
        spheres,
        sites,
        rotamers.n_rots_for_block,
        rotamers.rot_offset_for_block,
        rotamers.group_for_block,
        reach=reach,
        begin=begin,
        count=count,
    )
