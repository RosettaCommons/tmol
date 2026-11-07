"""Pose coordinate layout around the common native kinematic forest."""

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.chemical.packed import PackedBlockTypesData
from tmol_core.kinematics.scan_ordering import ScanOrdering, ScanPlan

from ._host import NativePosePlanner
from .fold_forest import FoldForest
from .forest import KinematicForest


def _prepare_palette(pbt):
    from tmol_core.kinematics.chemical import prepare_packed_kinematic_stencils

    if not hasattr(pbt, "_kinematic_preparation"):
        host = PackedBlockTypesData.from_restypes(
            pbt.chem_db, pbt.active_block_types, min_n_conn=pbt.max_n_conn
        )
        stencils = prepare_packed_kinematic_stencils(
            host.active_block_types,
            max_n_atoms=host.max_n_atoms,
            max_n_conn=host.max_n_conn,
            max_n_torsions=host.max_n_torsions,
        )
        pbt._kinematic_preparation = (host, stencils)
    return pbt._kinematic_preparation


def _prepare_pose(pose, fold_forest):
    from tmol_core.kinematics.pose import validate_pose_kinematics_inputs

    host, stencils = _prepare_palette(pose.packed_block_types)
    offsets = np.asarray(pose.block_coord_offset)
    kinds = np.asarray(pose.block_type_ind)
    connections = np.asarray(pose.inter_residue_connections)
    edges = np.asarray(fold_forest.edges)
    validate_pose_kinematics_inputs(
        kinds,
        offsets,
        connections,
        edges,
        host.n_atoms,
        host.n_conn,
        host.polymeric_conn_inds,
        pose.max_n_atoms_per_pose,
    )
    conn_info = stencils["scan_path_seg_that_builds_output_conn"]
    return NativePosePlanner().assemble_pose(
        pose.max_n_atoms_per_pose,
        offsets,
        kinds,
        connections,
        edges,
        host.n_atoms,
        host.atom_is_real,
        host.n_conn,
        host.conn_atom,
        host.polymeric_conn_inds,
        *(
            stencils[name]
            for name in (
                "jump_atom",
                "parents",
                "dof_type",
                "n_gens",
                "scan_path_seg_that_builds_output_conn",
                "nodes_for_gen",
                "n_scan_path_segs",
                "scan_path_seg_starts",
                "scan_path_seg_is_real",
                "scan_path_seg_is_inter_block",
                "scan_path_seg_lengths",
            )
        ),
        edges,
        kinds,
        connections,
        conn_info,
        host.polymeric_conn_inds,
        host.n_conn,
    )


@dataclass(frozen=True, eq=False, init=False)
class PoseStackKinematics:
    """Prepare molecular kinematics once from exact pose chemistry and topology.

    Assembly runs on CPU without Torch. Coordinate execution uses the pose's
    JAX device. All real atoms retain their pose identities; padded coordinates
    are preserved. Empty poses are not supported by the native assembly planner.
    """

    forest: KinematicForest
    atom_ids: object
    keep_atom_fixed: object
    block_in_and_first_out: object
    atom_for_jump: object
    atom_for_root_jump: object
    _pose: object = field(repr=False)
    _host_data: tuple = field(repr=False)

    def __init__(self, pose, fold_forest=None):
        if (
            pose.coords.ndim != 3
            or pose.coords.shape[0] != pose.n_poses
            or pose.coords.shape[2] != 3
        ):
            raise ValueError("pose coordinates must have shape (poses, atoms, 3)")
        if fold_forest is None:
            fold_forest = FoldForest.reasonable_fold_forest(pose)
        values = _prepare_pose(pose, fold_forest)
        ids, kinds, parent, frame_x, frame_y, frame_z = values[:6]
        ordering = ScanOrdering(ScanPlan(*values[6:9]), ScanPlan(*values[9:12]))
        forest = KinematicForest(
            parent,
            frame_x,
            frame_y,
            frame_z,
            kinds,
            device=pose.device,
            scan_ordering=ordering,
        )
        object.__setattr__(self, "forest", forest)
        object.__setattr__(self, "_pose", pose)
        object.__setattr__(self, "_host_data", values)
        for name, array in zip(
            (
                "atom_ids",
                "block_in_and_first_out",
                "keep_atom_fixed",
                "atom_for_jump",
                "atom_for_root_jump",
            ),
            (ids, *values[12:]),
        ):
            object.__setattr__(self, name, jax.device_put(array, pose.device))

    def _coordinates(self, coords):
        dtype = getattr(coords, "dtype", None)
        coords = self._pose.coords if coords is None else jnp.asarray(coords)
        if dtype is not None and coords.dtype != dtype:
            raise ValueError("enable JAX float64 before using float64 coordinates")
        if coords.shape != self._pose.coords.shape:
            raise ValueError("coordinate shape must match the prepared pose")
        return jax.device_put(coords, self.forest.device)

    def inverse(self, coords=None):
        """Initialize nine-column DOFs from coordinates in the original pose layout."""
        coords = self._coordinates(coords)
        gathered = coords.reshape((-1, 3))[self.atom_ids[1:]]
        root = jax.device_put(np.zeros((1, 3), dtype=coords.dtype), self.forest.device)
        return self.forest.inverse(jnp.concatenate((root, gathered), axis=0))

    def bind(self, coords=None):
        """Snapshot pose coordinates and return a differentiable six-column map."""
        coords = self._coordinates(coords)
        mapping = self.forest.bind(self.inverse(coords))
        return PoseCoordinates(mapping, self.atom_ids[1:], coords)

    def resolve_move_map(self, move_map):
        """Resolve hierarchical movement rules through the common native planner.

        Returns a snapshotted boolean [nodes,6] mask on the forest's device.
        Reference rotations are fixed by the coordinate map and cannot move.
        """
        pose = self._pose
        host, stencils = _prepare_palette(pose.packed_block_types)
        expected = (
            pose.n_poses,
            pose.max_n_blocks,
            host.max_n_torsions,
            pose.max_n_atoms_per_block,
        )
        dimensions = (
            move_map.n_poses,
            move_map.max_n_blocks,
            move_map.max_n_named_torsions,
            move_map.max_n_atoms_per_block,
        )
        if dimensions != expected:
            raise ValueError("MoveMap dimensions must match the prepared pose")
        data = self._host_data
        mask = NativePosePlanner().create_minimizer_map(
            data[0],
            pose.max_n_atoms_per_pose,
            np.asarray(pose.block_coord_offset),
            np.asarray(pose.block_type_ind),
            np.asarray(pose.inter_residue_connections),
            data[12],
            data[14],
            data[15],
            data[13],
            host.n_torsions,
            stencils["uaid_for_torsion_by_inconn"],
            host.is_torsion_mc,
            host.which_mcsc_torsions,
            host.atom_downstream_of_conn,
            *move_map.native_arguments,
        )
        if mask[:, 6:].any():
            raise ValueError("reference rotations cannot be movable DOFs")
        return jax.device_put(mask[:, :6], self.forest.device)


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True, eq=False)
class PoseCoordinates:
    """Map movable DOFs [nodes,6] to the original padded pose coordinates.

    This callable supports first reverse derivatives, jit and vmap. Binding
    fixes reference rotations and padding; inverse-coordinate AD is unavailable.
    """

    _mapping: object = field(repr=False)
    _atom_ids: object = field(repr=False)
    _coordinates: object = field(repr=False)

    @property
    def initial_dofs(self):
        return self._mapping.initial_dofs

    def __call__(self, dofs):
        coordinates = self._mapping(dofs)
        return (
            self._coordinates.reshape((-1, 3))
            .at[self._atom_ids]
            .set(coordinates[1:])
            .reshape(self._coordinates.shape)
        )

    def tree_flatten(self):
        return (self._mapping, self._atom_ids, self._coordinates), None

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(*children)
