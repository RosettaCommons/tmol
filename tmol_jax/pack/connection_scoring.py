"""Connection-specific disulfide/metal scoring through common native physics."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register
from tmol_jax.score.disulfide.disulfide_energy_term import DisulfideEnergyTerm
from tmol_jax.score.metal.metal_energy_term import MetalCoordinationEnergyTerm
from .graph import prefix_counts
from .scoring import validate_rotamer_context, sparse_nonbonded_values


@partial(jax.jit, static_argnames=("target", "operation", "outputs", "extent", "width"))
def _stage(arguments, *, target, operation, outputs, extent=0, width=0):
    register(target)
    return jax.ffi.ffi_call(target, outputs, vmap_method="sequential")(
        *arguments,
        operation=np.int64(operation),
        extent=np.int64(extent),
        width=np.int64(width),
    )


def prepare_rotamer_connection(pose, rotamers, database, *, metal=False):
    validate_rotamer_context(pose, rotamers)
    if rotamers.correlated_groups:
        raise NotImplementedError(
            "sparse connection lockstep candidate filtering is not implemented"
        )
    with jax.default_device(pose.device):
        cls = MetalCoordinationEnergyTerm if metal else DisulfideEnergyTerm
        parameters = cls(database).setup_for_pose_stack(
            pose, **({"include_empty": True} if metal else {})
        )
        parameters = (
            rotamers.coord_offset_for_rot,
            rotamers.rot_offset_for_block,
            rotamers.first_rot_block_type,
            *parameters[3:],
            rotamers.block_ind_for_rot,
            rotamers.pose_for_rot,
            rotamers.block_type_ind_for_rot,
            rotamers.n_rots_for_block,
        )
        count_parameters = tuple(
            (
                p.astype(rotamers.coords.dtype)
                if jnp.issubdtype(p.dtype, jnp.floating)
                else p
            )
            for p in parameters
        )
        family = "metal" if metal else "disulfide"
        target = f"tmol_{family}_rotamer_topology"
        shape = jax.ShapeDtypeStruct
        counts = _stage(
            (rotamers.coords, *count_parameters),
            target=target,
            operation=0,
            outputs=shape((rotamers.n_rotamers_total, 8), np.uint8),
        )
        offsets = prefix_counts(counts)
        maximum = prefix_counts(counts, maximum=True)[-1]
        total, largest = (
            np.asarray(jnp.stack((offsets[-1], maximum))).copy().view(np.int64).ravel()
        )
        total, largest = int(total), int(largest)
        extent = rotamers.n_rotamers_total * largest
        if max(total, largest, extent) > np.iinfo(np.int32).max:
            raise OverflowError(
                "connection candidate dispatch exceeds native int32 capacity"
            )
        indices, identities = _stage(
            (rotamers.coords, *count_parameters, counts, offsets[:-1]),
            target=target,
            operation=1,
            outputs=(
                shape((3, total), np.int32),
                shape((3 if metal else 2, total), np.int32),
            ),
            extent=largest,
            width=extent,
        )
    return (f"tmol_rotamer_{family}", 1, (*parameters, identities), 0.0), indices


def prepare_rotamer_disulfide(pose, rotamers, database):
    return prepare_rotamer_connection(pose, rotamers, database)


def prepare_rotamer_metal(pose, rotamers, database):
    return prepare_rotamer_connection(pose, rotamers, database, metal=True)


def sparse_connection_scores(coords, state):
    prepared, indices = state
    # The mature disulfide forward kernel evaluates derivatives with its value.
    # XLA owns the discarded coordinate-sized buffer; reverse uses weighted
    # entries and a single coordinate output, never an entry-by-atom Jacobian.
    workspace = (
        (coords.size * coords.dtype.itemsize, 0)
        if prepared[0] == "tmol_rotamer_disulfide"
        else None
    )
    return (
        sparse_nonbonded_values(coords, prepared, indices, workspace=workspace),
        indices,
    )
