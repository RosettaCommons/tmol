"""Sparse conformer value/VJP execution with shared nonbonded parameters."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register, workspace_sizes
from tmol_jax.score.nonbonded import prepare_independent_nonbonded
from .dispatch import rotamer_spheres, block_spheres, candidate_pair_groups


@partial(
    jax.jit, static_argnames=("target", "channels", "reach", "backward", "workspace")
)
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
    workspace=None,
):
    if target in ("tmol_rotamer_backbone_precise", "tmol_rotamer_elec_precise"):
        if coords.dtype == jnp.float32:
            workspace = workspace_sizes(target, (coords.shape[0],), 32)
        else:
            target = target.removesuffix("_precise")
    register(target)
    operands = (coords, *parameters, indices)
    if backward:
        operands += (weights,)
    shape = coords.shape if backward else (channels, indices.shape[1])
    output = jax.ShapeDtypeStruct(shape, coords.dtype)
    outputs = (
        output
        if workspace is None
        else (output, jax.ShapeDtypeStruct((workspace[backward],), jnp.uint8))
    )
    result = jax.ffi.ffi_call(target, outputs, vmap_method="sequential")(
        *operands,
        operation=np.int64(backward),
        extent=np.int64(0),
        width=np.int64(0),
        reach=np.float64(reach),
    )
    return result if workspace is None else result[0]


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


def sparse_nonbonded_values(coords, state, indices, *, workspace=None):
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
            args[:-1],
            args[-1],
            weights,
            target=target,
            channels=channels,
            reach=reach,
            backward=backward,
            workspace=workspace,
        )

    # Candidate identities are dynamic device operands. Carry them through the
    # reverse rule explicitly so nested jit/vmap never captures a tracer.
    return coordinate_vjp(coords, (*parameters, indices), evaluate)


def validate_rotamer_context(pose, rotamers):
    if rotamers.context is None:
        raise ValueError("sparse scoring requires conformers with pose provenance")
    rotamers.context.validate(pose)
    if rotamers.device != pose.device:
        raise ValueError("conformers and pose must be on the same device")
    sources = (
        rotamers.n_rots_for_block,
        rotamers.group_for_block,
        rotamers.correlated_groups,
    )
    prior = getattr(rotamers, "_validated_correspondence", None)
    if prior is None or any(old is not new for old, new in zip(prior, sources)):
        from tmol_core.pack.correlation import correlation_indices

        expected = correlation_indices(
            sources[0].shape, sources[2], np.asarray(sources[0])
        )
        if not np.array_equal(expected, sources[1]):
            raise ValueError(
                "conformer group indices disagree with declared correspondence"
            )
        object.__setattr__(rotamers, "_validated_correspondence", sources)


def prepare_rotamer_nonbonded(
    pose, rotamers, database, *, electrostatic=False, soft_repulsive=False
):
    """Prepare shared chemistry for sparse LJ/LK or electrostatic execution."""
    validate_rotamer_context(pose, rotamers)
    with jax.default_device(pose.device):
        target, channels, parameters, reach, _ = prepare_independent_nonbonded(
            pose,
            database,
            electrostatic=electrostatic,
            soft_repulsive=soft_repulsive,
        )
    common = (
        rotamers.coord_offset_for_rot,
        rotamers.block_ind_for_rot,
        rotamers.pose_for_rot,
        rotamers.block_type_ind_for_rot,
        rotamers.first_rot_block_type,
        rotamers.rot_offset_for_block,
    )
    target = target.replace("tmol_", "tmol_rotamer_", 1)
    if electrostatic and pose.device.platform == "gpu":
        target += "_precise"
    return (
        target,
        channels,
        (*common, *parameters[5:]),
        reach,
    )


def nonbonded_candidate_groups(
    rotamers, packed_block_types, *, reaches, coords=None, begin=0, count=None
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
    return candidate_pair_groups(
        rotamers.first_rot_block_type,
        spheres,
        sites,
        rotamers.n_rots_for_block,
        rotamers.rot_offset_for_block,
        rotamers.group_for_block,
        reaches=reaches,
        begin=begin,
        count=count,
    )


def nonbonded_candidate_pairs(
    rotamers, packed_block_types, *, reach, coords=None, begin=0, count=None
):
    """Recompute one sparse candidate set from the current conformer geometry."""
    return nonbonded_candidate_groups(
        rotamers,
        packed_block_types,
        reaches=(reach,),
        coords=coords,
        begin=begin,
        count=count,
    )[0]
