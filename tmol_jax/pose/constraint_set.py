"""Differentiable parameter storage around common constraint atom topology."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pose.constraints import ConstraintTopology, merge_constraint_functions
from tmol_jax.types import pytree_dataclass


def _pad_parameters(parameters, width):
    return jnp.pad(parameters, ((0, 0), (0, width - parameters.shape[1])))


@pytree_dataclass
class ConstraintSet:
    """Immutable pose/block/atom constraints with framework-native callbacks.

    Atom topology is prepared on the host. Parameter arrays remain JAX leaves,
    including when adding, repeating, splitting or concatenating constraints.
    A callback accepts gathered atoms[N,4,3] and parameters[N,P], returning N
    energies. Builtin callbacks live on ConstraintEnergyTerm.
    """

    n_poses: int
    constraint_function_inds: jax.Array
    constraint_atoms: jax.Array
    constraint_params: jax.Array
    constraint_num_unique_blocks: jax.Array
    constraint_unique_blocks: jax.Array
    constraint_functions: tuple

    @property
    def device(self):
        return getattr(self.constraint_params, "device", None)

    @property
    def n_constraints(self):
        return self.constraint_atoms.shape[0]

    def _topology(self):
        return ConstraintTopology(
            self.n_poses,
            **{
                name: np.asarray(getattr(self, name))
                for name in ConstraintTopology.empty(self.n_poses).arrays()
            },
        )

    @classmethod
    def _from_topology(cls, topology, params, functions, device=None):
        return cls(
            n_poses=topology.n_poses,
            constraint_params=jax.device_put(params, device),
            constraint_functions=tuple(functions),
            **{
                name: jax.device_put(value, device)
                for name, value in topology.arrays().items()
            },
        )

    @classmethod
    def create_empty(cls, device=None, n_poses=1):
        return cls._from_topology(
            ConstraintTopology.empty(n_poses),
            jnp.empty((0, 1), jnp.float32),
            (),
            device,
        )

    def add_constraints(self, fn, atom_indices, params=None):
        if not callable(fn):
            raise TypeError("constraint function must be callable")
        atom_indices = np.asarray(atom_indices)
        if params is None:
            params = jnp.zeros((len(atom_indices), 1), jnp.float32)
        params = jnp.asarray(params)
        if params.ndim != 2 or params.shape[0] != len(atom_indices):
            raise ValueError("params must have shape [constraint, parameter]")
        if not jnp.issubdtype(params.dtype, jnp.floating):
            raise TypeError("constraint parameters must be floating point")
        functions, remaps = merge_constraint_functions(
            (self.constraint_functions, (fn,))
        )
        empty = not self.constraint_functions
        topology, repeats = self._topology().add(
            atom_indices, int(remaps[1][0]), replace_empty=empty
        )
        params = jnp.tile(params, (repeats, 1))
        if not empty:
            width = max(self.constraint_params.shape[1], params.shape[1])
            params = jnp.concatenate(
                (
                    _pad_parameters(self.constraint_params, width),
                    _pad_parameters(params, width),
                )
            )
        return self._from_topology(topology, params, functions, self.device)

    def add_constraints_to_all_poses(self, fn, atom_indices, params=None):
        atoms = np.asarray(atom_indices)
        if atoms.ndim == 3 and atoms.shape[2] == 3:
            atoms = atoms[:, :, 1:]
        return self.add_constraints(fn, atoms, params)

    def split(self, index):
        if index < 0 or index >= self.n_poses:
            raise IndexError("constraint pose index is out of range")
        topology, rows = self._topology().split(index)
        return self._from_topology(
            topology,
            self.constraint_params[rows],
            self.constraint_functions,
            self.device,
        )

    @classmethod
    def concatenate(
        cls,
        constraint_sets,
        from_multiple_pose_stacks=True,
        n_poses=None,
        ps_offset=None,
    ):
        present = [cs for cs in constraint_sets if cs is not None]
        if not present:
            return None
        device = present[0].device
        if any(cs.device != device for cs in present):
            raise ValueError("constraint sets must be on the same device")
        functions, remaps = merge_constraint_functions(
            [None if cs is None else cs.constraint_functions for cs in constraint_sets]
        )
        topology = ConstraintTopology.concatenate(
            [None if cs is None else cs._topology() for cs in constraint_sets],
            remaps,
            from_multiple_pose_stacks=from_multiple_pose_stacks,
            n_poses=n_poses,
            ps_offset=ps_offset,
        )
        width = max(cs.constraint_params.shape[1] for cs in present)
        params = jnp.concatenate(
            [_pad_parameters(cs.constraint_params, width) for cs in present]
        )
        return cls._from_topology(topology, params, functions, device)
