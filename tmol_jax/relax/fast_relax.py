"""Array execution and public defaults for the common FastRelax protocol."""

import copy

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.relax import fast_relax as _shared_fast_relax, default_task_operation
from tmol_jax.pack import (
    PackerTask,
    DunbrackChiSampler,
    FixedAAChiSampler,
    IncludeCurrentSampler,
    NaChiRotamerSampler,
    pack_rotamers,
)
from tmol_jax.pack.conjugated import add_conjugated_group_sampler
from tmol_jax.optimization import run_cart_min, run_kin_min
from tmol_jax.score.score_types import ScoreType


class DefaultCartesianMinimizer:
    """Rendered JAX Cartesian L-BFGS; options are passed to run_cart_min."""

    def __init__(self, **options):
        self.options = options

    def __call__(self, pose, sfxn, *, fold_forest, move_map, verbose):
        return run_cart_min(
            pose,
            sfxn,
            getattr(move_map, "coord_mask", None),
            verbose=verbose,
            **self.options,
        )


class DefaultKinematicMinimizer:
    """Common native kinematics and JAX coordinate-gradient minimization."""

    def __init__(self, **options):
        self.options = options

    def __call__(self, pose, sfxn, *, fold_forest, move_map, verbose):
        return run_kin_min(
            pose,
            sfxn,
            move_map,
            fold_forest=fold_forest,
            verbose=verbose,
            **self.options,
        )


def _select_pose(best, candidate, better):
    if best.packed_block_types is not candidate.packed_block_types:
        raise ValueError("FastRelax candidates must share their chemistry palette")
    width = max(best.coords.shape[1], candidate.coords.shape[1])

    def coords(pose):
        return jnp.pad(pose.coords, ((0, 0), (0, width - pose.coords.shape[1]), (0, 0)))

    def select(name):
        left, right = getattr(best, name), getattr(candidate, name)
        mask = better.reshape((-1,) + (1,) * (left.ndim - 1))
        return jnp.where(mask, right, left)

    types = select("block_type_ind")
    connections = select("inter_residue_connections")
    from tmol_jax.pose.pose_stack_builder import _bond_separation

    topology = _bond_separation(
        best.packed_block_types, np.asarray(types), np.asarray(connections)
    )
    info = copy.deepcopy(best.pdb_info)
    rows = np.flatnonzero(np.asarray(better))
    if info is not None:
        if candidate.pdb_info is None:
            raise ValueError("FastRelax candidate has lost pose metadata")
        import attrs

        for name in attrs.fields_dict(type(info)):
            left, right = getattr(info, name), getattr(candidate.pdb_info, name)
            if (left is None) != (right is None):
                raise ValueError("FastRelax candidates have incompatible pose metadata")
            if left is not None:
                if (
                    isinstance(left, np.ndarray)
                    and left.ndim == 2
                    and left.shape[1] != right.shape[1]
                ):
                    width = max(left.shape[1], right.shape[1])
                    left = np.pad(left, ((0, 0), (0, width - left.shape[1])))
                    right = np.pad(right, ((0, 0), (0, width - right.shape[1])))
                    setattr(info, name, left)
                for row in rows:
                    left[row] = copy.deepcopy(right[row])
    elif candidate.pdb_info is not None:
        raise ValueError("FastRelax candidates have incompatible pose metadata")
    with jax.default_device(best.device):
        return best.replace(
            coords=jnp.where(better[:, None, None], coords(candidate), coords(best)),
            block_coord_offset=select("block_coord_offset"),
            block_type_ind=types,
            inter_residue_connections=connections,
            inter_block_bondsep=None,
            near_blocks=jnp.asarray(topology.near_blocks),
            bondsep=jnp.asarray(topology.bondsep),
            pdb_info=info,
        )


class _JaxRelaxExecution:
    score_type = ScoreType
    task = staticmethod(PackerTask)
    select_pose = staticmethod(_select_pose)

    def __init__(self, seed):
        self.random = np.random.PCG64(seed)

    @staticmethod
    def clone(pose):
        return pose.clone()

    @staticmethod
    def synchronize(pose):
        pose.coords.block_until_ready()

    def pack(self, pose, sfxn, task, verbose):
        return pack_rotamers(pose, sfxn, task, seed=int(self.random.random_raw()))

    @staticmethod
    def select_scores(best, candidate, better):
        return jnp.where(better, candidate, best)

    @staticmethod
    def default_minimizer(pose, cuda_graph):
        return DefaultCartesianMinimizer()

    @staticmethod
    def default_task_operations(pose, sfxn):
        samplers = [
            DunbrackChiSampler.from_database(sfxn.param_db),
            FixedAAChiSampler(),
            IncludeCurrentSampler(),
        ]
        if any(
            bt.properties.polymer.backbone_type in ("dna", "rna")
            for bt in pose.packed_block_types.active_block_types
        ):
            samplers.append(NaChiRotamerSampler.from_database(sfxn.param_db))
        return default_task_operation(
            pose, samplers, lambda task, pose: add_conjugated_group_sampler(task)
        )


def fast_relax(
    pose_stack,
    sfxn,
    packer_pallete,
    move_map,
    fold_forest,
    *,
    seed,
    task_operations=None,
    num_repeats=2,
    ramp_constraints=None,
    schedule=None,
    min_fn=None,
    release_between_stages=True,
    verbose=False,
):
    """Execute shared FastRelax with native packing and a JAX minimizer.

    Schedule, per-pose acceptance, task callbacks and release hooks match the
    mature protocol. The explicit unsigned 64-bit seed controls local packing
    randomness. The default minimizer uses JAX strong-Wolfe L-BFGS on the summed
    batch objective; mature Torch per-pose Armijo trajectories are not matched.
    A custom minimizer receives fold_forest, move_map and verbose keywords.
    Existing packing capability boundaries apply to every stage.
    The score function's stage weights are updated as in the mature API.
    """
    if not isinstance(seed, (int, np.integer)) or not 0 <= seed < 1 << 64:
        raise ValueError("FastRelax seed must be an unsigned 64-bit integer")
    if isinstance(pose_stack.coords, jax.core.Tracer):
        raise TypeError("FastRelax requires eager discrete packing")
    with jax.default_device(pose_stack.device):
        return _shared_fast_relax(
            pose_stack,
            sfxn,
            packer_pallete,
            move_map,
            fold_forest,
            execution=_JaxRelaxExecution(int(seed)),
            task_operations=task_operations,
            num_repeats=num_repeats,
            ramp_constraints=ramp_constraints,
            schedule=schedule,
            min_fn=min_fn,
            release_between_stages=release_between_stages,
            verbose=verbose,
        )
