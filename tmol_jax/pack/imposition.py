"""Native conformer selection and layout with differentiable JAX coordinate gathers."""

from dataclasses import dataclass
from functools import partial
import copy

import jax
import jax.numpy as jnp
import numpy as np

from tmol_jax.score._native import register
from .graph import prefix_counts, load_balance
from .dispatch import _narrow_prefix
from .scoring import validate_rotamer_context


@jax.jit
def _wide(values):
    """Sign-extend int32 indices without depending on JAX's x64 setting."""
    high = jnp.where(values < 0, jnp.full_like(values, -1), jnp.zeros_like(values))
    return jnp.stack((values, high), axis=-1).view(jnp.uint8)


@partial(jax.jit, static_argnames=("shape",))
def _gather_coordinates(coords, source, target, *, shape):
    return (
        jnp.zeros((shape[0] * shape[1], 3), coords.dtype)
        .at[target]
        .set(coords[source])
        .reshape(shape)
    )


@partial(jax.jit, static_argnames=("operation", "outputs", "extent", "compact"))
def _stage(arguments, *, operation, outputs, extent=0, compact=False):
    register("tmol_pose_imposition")
    return jax.ffi.ffi_call("tmol_pose_imposition", outputs, vmap_method="sequential")(
        *arguments,
        operation=np.int64(operation),
        extent=np.int64(extent),
        width=np.int64(compact),
    )


def _check_errors(errors):
    if bool(np.asarray(jnp.any(errors != 0))):
        raise ValueError("invalid pose imposition assignment, topology or atom offsets")


@dataclass(frozen=True)
class PoseImpositionPlan:
    """Discrete identities prepared eagerly; coordinate application supports AD.

    ``coordinates`` supports jit/vmap/JVP and higher coordinate derivatives.
    ``apply`` returns a pose with the selected topology and parameter context.
    Neither method differentiates the discrete conformer assignment.
    """

    pose: object
    source: jax.Array
    target: jax.Array
    selected: jax.Array
    input_shape: tuple

    def coordinates(self, rotamer_coords):
        if rotamer_coords.shape != self.input_shape:
            raise ValueError(
                "conformer coordinates no longer match the imposition plan"
            )
        if rotamer_coords.dtype not in (jnp.float32, jnp.float64):
            raise TypeError("pose imposition requires float32/64 coordinates")
        if (
            not isinstance(rotamer_coords, jax.core.Tracer)
            and rotamer_coords.device != self.source.device
        ):
            raise ValueError("imposition coordinates and plan must share a device")
        return _gather_coordinates(
            rotamer_coords, self.source, self.target, shape=self.pose.coords.shape
        )

    def apply(self, rotamer_coords):
        return self.pose.replace(coords=self.coordinates(rotamer_coords))


def prepare_pose_imposition(pose, rotamers, retained, assignments):
    """Prepare the shared native selection/layout for retained local assignments.

    ``retained`` comes from ``build_interaction_graph``. ``assignments`` is
    int32[poses, trajectories, molten_blocks]; trajectory zero is selected.
    Atom layout changes require eager allocation. Fixed-layout coordinates
    remain on their input device, and unchanged layouts preserve gaps/capacity.
    Constraints on a changed identity and correlated groups currently reject.
    """
    if any(isinstance(x, jax.core.Tracer) for x in (pose.coords, assignments)):
        raise TypeError("pose imposition preparation requires eager discrete choices")
    validate_rotamer_context(pose, rotamers)
    if rotamers.correlated_groups:
        raise NotImplementedError(
            "pose imposition of correlated groups is not implemented"
        )
    p, b = pose.block_type_ind.shape
    m = retained["block_rotamer_counts"].shape[1]
    if (
        assignments.dtype != jnp.int32
        or assignments.ndim != 3
        or assignments.shape[0] != p
        or assignments.shape[1] < 1
        or assignments.shape[2] != m
    ):
        raise ValueError(
            "assignments must be int32[poses, trajectories, molten_blocks]"
        )
    arrays = tuple(retained.values()) + (assignments, rotamers.coords)
    if any(isinstance(x, jax.Array) and x.device != pose.device for x in arrays):
        raise ValueError("pose imposition arrays must share the pose device")
    choices = assignments[:, 0, :]
    counts = retained["block_rotamer_counts"]
    real = jnp.arange(m)[None, :] < retained["block_pose_counts"][:, None]
    if bool(np.asarray(jnp.any(real & ((choices < 0) | (choices >= counts))))):
        raise ValueError("assignment exceeds a block's retained conformer count")
    shape = jax.ShapeDtypeStruct

    def wide(*dims):
        return shape((*dims, 8), np.uint8)

    def integer(*dims):
        return shape(dims, np.int32)

    with jax.default_device(pose.device):
        original = _wide(pose.block_type_ind)
        args = (
            original,
            retained["fixed"],
            _wide(retained["block_pose_counts"]),
            _wide(retained["block_rotamer_offsets"]),
            retained["inverse_rotamers"],
            choices,
        )
        flags = _stage(args, operation=0, outputs=wide(p * b))
        prefix = prefix_counts(flags)[:-1]
        selected, errors = _stage(
            (*args, flags, prefix), operation=1, outputs=(wide(p, b), integer(p, b))
        )
        _check_errors(errors)
        types, counts, changed, errors = _stage(
            (
                original,
                selected,
                _wide(rotamers.block_type_ind_for_rot),
                pose.packed_block_types.n_atoms,
                rotamers.coord_offset_for_rot,
            ),
            operation=2,
            extent=rotamers.coords.shape[0],
            outputs=(wide(p, b), wide(p * b), integer(p * b), integer(p, b)),
        )
        _check_errors(errors)
        compact = bool(np.asarray(jnp.any(changed != 0)))
        prefix = prefix_counts(counts)
        total = int(np.asarray(prefix[-1]).copy().view(np.int64)[0])
        if total > np.iinfo(np.int32).max:
            raise OverflowError("pose imposition atom count exceeds int32 capacity")
        offsets, totals, errors = _stage(
            (counts, prefix[:-1], pose.block_coord_offset),
            operation=3,
            extent=pose.coords.shape[1],
            compact=compact,
            outputs=(integer(p, b), wide(p), integer(p, b)),
        )
        _check_errors(errors)
        capacity = (
            int(np.asarray(totals).copy().view(np.int64).max(initial=0))
            if compact
            else pose.coords.shape[1]
        )
        if p * capacity > np.iinfo(np.int32).max:
            raise OverflowError("pose imposition coordinate capacity exceeds int32")
        owners = load_balance(prefix[:-1], total=total)
        source, target = _stage(
            (selected, rotamers.coord_offset_for_rot, prefix[:-1], owners, offsets),
            operation=4,
            extent=capacity,
            outputs=(wide(total), wide(total)),
        )
        types = _narrow_prefix(types.reshape(-1, 8)).reshape(p, b)
        source, target = _narrow_prefix(source), _narrow_prefix(target)
        result = pose.replace(
            coords=jnp.zeros((p, capacity, 3), pose.coords.dtype),
            block_type_ind=types,
            block_coord_offset=offsets,
        )
        if compact:
            if pose.constraint_set is not None and pose.constraint_set.n_constraints:
                raise NotImplementedError(
                    "design with constraints requires atom-identity remapping"
                )
            old, new = np.asarray(pose.block_type_ind), np.asarray(types)
            connections = np.asarray(pose.inter_residue_connections)
            blocks = pose.packed_block_types.active_block_types
            for q, block, port in zip(*np.nonzero(connections[..., 0] >= 0)):
                old_bt, new_bt = blocks[old[q, block]], blocks[new[q, block]]
                old_name = old_bt.connections[port].name
                if new_bt.connection_to_cidx.get(old_name, -1) != port:
                    raise NotImplementedError(
                        "design must preserve bonded connection port identities"
                    )
            from tmol_jax.pose.pose_stack_builder import _bond_separation

            distances = _bond_separation(pose.packed_block_types, new, connections)
            info = copy.deepcopy(pose.pdb_info)
            if info is not None:
                for q, block in zip(*np.nonzero(old != new)):
                    if info.res_name is not None and block < len(info.res_name[q]):
                        info.res_name[q][block] = blocks[new[q, block]].name3
                    if info.atom_record_type is not None and block < len(
                        info.atom_record_type[q]
                    ):
                        info.atom_record_type[q][block] = []
            result = result.replace(
                inter_block_bondsep=None,
                near_blocks=jnp.asarray(distances.near_blocks),
                bondsep=jnp.asarray(distances.bondsep),
                pdb_info=info,
            )
    return PoseImpositionPlan(result, source, target, selected, rotamers.coords.shape)


def impose_top_rotamer_assignments(pose, rotamers, graph, assignments):
    """Return the selected pose using a graph's retained conformer identities."""
    plan = prepare_pose_imposition(pose, rotamers, graph["retained"], assignments)
    return plan.apply(rotamers.coords)
