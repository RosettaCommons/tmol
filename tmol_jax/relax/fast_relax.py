"""Array execution and public defaults for the common FastRelax protocol."""

import copy

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.relax import (
    fast_relax as _shared_fast_relax,
    default_task_operation,
    KINEMATIC_RELAX_LBFGS_OPTIONS,
)
from tmol_jax.pack import (
    PackerTask,
    PackerPalette,
    DunbrackChiSampler,
    FixedAAChiSampler,
    IncludeCurrentSampler,
    NaChiRotamerSampler,
)
from tmol_jax.pack.packing import _pack_rotamers, _new_seed_stream, _seed_value
from tmol_jax.pack.conjugated import add_conjugated_group_sampler
from tmol_jax.optimization import run_cart_min, run_kin_min
from tmol_jax.kinematics import CartesianMoveMap, MoveMap, FoldForest
from tmol_jax.score.score_types import ScoreType


def _check_graph_mode(cuda_graph):
    if cuda_graph:
        raise NotImplementedError(
            "Explicit CUDA graph capture is not implemented by the JAX minimizer"
        )


class CartesianRelaxMinimizer:
    """JAX Cartesian minimization; None frees all real atoms.

    Extra options are passed to ``run_cart_min``. JAX compiles its scoring
    execution; explicit ``cuda_graph=True`` is unsupported.
    """

    def __init__(self, move_map=None, *, cuda_graph=None, **options):
        if move_map is None:
            move_map = CartesianMoveMap()
        if not isinstance(move_map, CartesianMoveMap):
            raise TypeError("CartesianRelaxMinimizer needs a CartesianMoveMap")
        _check_graph_mode(cuda_graph)
        self.move_map = move_map
        self.options = options

    def __call__(self, pose, sfxn, *, verbose=False, rtol=None):
        # PR626 retains Cartesian LBFGS defaults, independent of stage rtol.
        return run_cart_min(
            pose, sfxn, self.move_map.coord_mask, verbose=verbose, **self.options
        )


class KinematicRelaxMinimizer:
    """JAX torsion/jump execution of the shared kinematic relax policy."""

    def __init__(self, move_map=None, fold_forest=None, *, cuda_graph=None, **options):
        if move_map is not None and not isinstance(move_map, MoveMap):
            raise TypeError("KinematicRelaxMinimizer needs a MoveMap")
        _check_graph_mode(cuda_graph)
        self.move_map = move_map
        self.fold_forest = fold_forest
        self.options = options

    def __call__(self, pose, sfxn, *, verbose=False, rtol=None):
        move_map = self.move_map
        if move_map is None:
            move_map = MoveMap.from_pose_stack(pose)
            move_map.move_all_mc = True
            move_map.move_all_sc = True
            move_map.move_all_jumps = True
        fold_forest = self.fold_forest
        if fold_forest is None:
            fold_forest = FoldForest.reasonable_fold_forest(pose)
        options = dict(self.options)
        convergence = dict(KINEMATIC_RELAX_LBFGS_OPTIONS)
        convergence.update(options.pop("optimizer_kwargs", {}) or {})
        if rtol is not None:
            convergence["rtol"] = rtol
        return run_kin_min(
            pose,
            sfxn,
            move_map,
            fold_forest=fold_forest,
            verbose=verbose,
            optimizer_kwargs=convergence,
            **options,
        )


# Retain existing JAX imports while exposing the shared public API names.
DefaultCartesianMinimizer = CartesianRelaxMinimizer
DefaultKinematicMinimizer = KinematicRelaxMinimizer


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
        self.seed = seed
        self.seed_states = {}

    @staticmethod
    def clone(pose):
        return pose.clone()

    @staticmethod
    def synchronize(pose):
        pose.coords.block_until_ready()

    def pack(self, pose, sfxn, task, verbose):
        if pose.device not in self.seed_states:
            self.seed_states[pose.device] = _new_seed_stream(pose.device, self.seed)
        return _pack_rotamers(
            pose, sfxn, task, self.seed, None, None, self.seed_states[pose.device]
        )

    @staticmethod
    def select_scores(best, candidate, better):
        return jnp.where(better, candidate, best)

    @staticmethod
    def default_minimizer():
        return CartesianRelaxMinimizer()

    default_palette = staticmethod(PackerPalette)

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
    *,
    seed,
    minimizer=None,
    packer_palette=None,
    task_operations=None,
    num_repeats=2,
    ramp_constraints=None,
    schedule=None,
    release_between_stages=True,
    verbose=False,
):
    """Execute shared FastRelax with native packing and a JAX minimizer.

    Schedule, per-pose acceptance, task callbacks and release hooks match the
    mature protocol. The explicit integer seed controls local packing
    randomness. The default minimizer uses segmented Armijo L-BFGS with the
    shared per-pose convergence, history and restart policy. Native sampling
    seeds follow the mature CPU/CUDA streams across stages and pose chunks;
    finite-precision results can still differ between frameworks.
    A custom minimizer receives verbose and optional stage rtol keywords.
    The minimizer owns its movement map and optional fold forest.
    Existing packing capability boundaries apply to every stage.
    The score function's stage weights are updated as in the mature API.
    """
    seed = _seed_value(seed)
    if isinstance(pose_stack.coords, jax.core.Tracer):
        raise TypeError("FastRelax requires eager discrete packing")
    with jax.default_device(pose_stack.device):
        return _shared_fast_relax(
            pose_stack,
            sfxn,
            minimizer=minimizer,
            packer_palette=packer_palette,
            execution=_JaxRelaxExecution(seed),
            task_operations=task_operations,
            num_repeats=num_repeats,
            ramp_constraints=ramp_constraints,
            schedule=schedule,
            release_between_stages=release_between_stages,
            verbose=verbose,
        )
