from collections.abc import Callable, Sequence

import attr
import torch
import torch.nn.functional as F

from tmol.types._torch import Tensor
from tmol.utility import resolve_device
from tmol_core.pose.constraints import (
    MAX_N_ATOMS,
    ConstraintTopology,
    count_unique_blocks,
    merge_constraint_functions,
    replicate_constraint_atoms,
)

ConstraintFunction = Callable[
    [Tensor[torch.float32][:, :, 3], Tensor[torch.float32][:, :]],
    Tensor[torch.float32][:],
]
_TOPOLOGY_ARRAYS = (
    "constraint_function_inds",
    "constraint_atoms",
    "constraint_num_unique_blocks",
    "constraint_unique_blocks",
)


def _tensor(array, device):
    # Empty NumPy dimensions can have zero strides; native tensor views require
    # the normal contiguous Torch layout even when no element is read.
    if not array.size:
        return torch.empty(array.shape, dtype=torch.int32, device=device)
    return torch.from_numpy(array).to(device=device)


@attr.s(frozen=True, slots=True, auto_attribs=True)
class ConstraintSet:
    """Immutable batched coordinate constraints and their tensor storage."""

    MAX_N_ATOMS = MAX_N_ATOMS
    device: torch.device
    n_poses: int
    constraint_function_inds: Tensor[torch.int32][:]
    constraint_atoms: Tensor[torch.int32][:, 4, 3]
    constraint_params: Tensor[torch.float32][:, :]
    constraint_num_unique_blocks: Tensor[torch.int32][:]
    constraint_unique_blocks: Tensor[torch.int32][:, 3]
    constraint_functions: tuple[ConstraintFunction, ...]

    def _topology(self):
        return ConstraintTopology(
            self.n_poses,
            **{
                name: getattr(self, name).numpy(force=True) for name in _TOPOLOGY_ARRAYS
            },
        )

    @classmethod
    def _from_topology(cls, topology, params, functions, device):
        return cls(
            device=device,
            n_poses=topology.n_poses,
            constraint_params=params,
            constraint_functions=functions,
            **{k: _tensor(v, device) for k, v in topology.arrays().items()},
        )

    @classmethod
    def create_empty(cls, device: torch.device, n_poses: int) -> "ConstraintSet":
        """Create an empty constraint set on a concrete device."""
        device = resolve_device(device)
        return cls._from_topology(
            ConstraintTopology.empty(n_poses),
            torch.empty((0, 1), dtype=torch.float32, device=device),
            (),
            device,
        )

    @classmethod
    def concatenate(
        cls,
        constraint_sets: Sequence["ConstraintSet | None"],
        from_multiple_pose_stacks: bool = True,
        n_poses: int | None = None,
        ps_offset: Tensor[torch.int64][:] | None = None,
    ) -> "ConstraintSet | None":
        """Combine constraints and callbacks, optionally shifting pose indices."""
        present = [cs for cs in constraint_sets if cs is not None]
        if not present:
            return None
        device = present[0].device
        assert all(
            cs.device == device for cs in present
        ), "All ConstraintSets must be on the same device"
        functions, remaps = merge_constraint_functions(
            [
                cs.constraint_functions if cs is not None else None
                for cs in constraint_sets
            ]
        )
        topology = ConstraintTopology.concatenate(
            [cs._topology() if cs is not None else None for cs in constraint_sets],
            remaps,
            from_multiple_pose_stacks=from_multiple_pose_stacks,
            n_poses=n_poses,
            ps_offset=None if ps_offset is None else ps_offset.numpy(force=True),
        )
        if len(topology.constraint_atoms):
            width = max(cs.constraint_params.shape[1] for cs in present)
            params = torch.cat(
                [
                    F.pad(
                        cs.constraint_params, (0, width - cs.constraint_params.shape[1])
                    )
                    for cs in present
                ]
            )
        else:
            params = torch.empty((0, 0), dtype=torch.float32, device=device)
        return cls._from_topology(topology, params, functions, device)

    def clone(self) -> "ConstraintSet":
        """Return a copy with independent tensor storage."""
        return attr.evolve(
            self,
            **{
                name: getattr(self, name).clone()
                for name in (*_TOPOLOGY_ARRAYS, "constraint_params")
            },
        )

    def to(self, device: torch.device) -> "ConstraintSet":
        """Return a copy whose tensors and metadata use ``device``."""
        device = resolve_device(device)
        return attr.evolve(
            self,
            device=device,
            **{
                name: getattr(self, name).to(device)
                for name in (*_TOPOLOGY_ARRAYS, "constraint_params")
            },
        )

    def split(self, index: int) -> "ConstraintSet":
        """Split out a single pose's worth of constraints from a batch."""
        topology, rows = self._topology().split(index)
        rows = torch.as_tensor(rows, dtype=torch.int64, device=self.device)
        return self._from_topology(
            topology,
            self.constraint_params[rows].clone(),
            self.constraint_functions,
            self.device,
        )

    @staticmethod
    def count_unique_blocks(
        atom_indices: Tensor[torch.int32][:, :, 3],
    ) -> Tensor[torch.int32][:]:
        """Count distinct consecutive blocks referenced by each constraint."""
        return _tensor(
            count_unique_blocks(atom_indices.numpy(force=True)), atom_indices.device
        )

    def add_constraints_to_all_poses(
        self,
        fn: ConstraintFunction,
        atom_indices: Tensor[torch.int32][:, :, :],
        params: Tensor[torch.float32][:, :] | None = None,
    ) -> "ConstraintSet":
        """Replicate block/atom constraints over every pose in this batch."""
        if atom_indices.ndim == 3 and atom_indices.size(2) == 3:
            atom_indices = atom_indices[:, :, 1:3]
        return self.add_constraints(fn, atom_indices, params)

    def add_constraints(
        self,
        fn: ConstraintFunction,
        atom_indices: Tensor[torch.int32][:, :, :],
        params: Tensor[torch.float32][:, :] | None = None,
    ) -> "ConstraintSet":
        """Add constraints identified by block/atom or pose/block/atom triples.

        Parameters retain their Torch autograd history. Only integer topology
        and callback registry preparation cross the framework-neutral boundary.
        """
        functions, remaps = merge_constraint_functions(
            [self.constraint_functions, (fn,)]
        )
        empty = not self.constraint_functions
        topology, repeats = self._topology().add(
            atom_indices.numpy(force=True), int(remaps[1][0]), replace_empty=empty
        )
        if params is not None:
            if params.ndim != 2 or params.size(0) != atom_indices.size(0):
                raise ValueError(
                    "params must have shape [constraint, parameter] with one row per constraint"
                )
            new_params = params.to(device=self.device, dtype=torch.float32)
        else:
            new_params = torch.empty(
                (atom_indices.shape[0], 0), dtype=torch.float32, device=self.device
            )
        new_params = new_params.repeat(repeats, 1)
        if empty:
            combined = new_params
        else:
            width = max(new_params.shape[1], self.constraint_params.shape[1])
            combined = torch.cat(
                (
                    F.pad(
                        self.constraint_params,
                        (0, width - self.constraint_params.shape[1]),
                    ),
                    F.pad(new_params, (0, width - new_params.shape[1])),
                )
            )
        return self._from_topology(topology, combined, functions, self.device)

    @staticmethod
    def replicate_constraints(
        n_poses: int,
        c_atms: Tensor[torch.int32][:, :, 3],
        c_params: Tensor[torch.float32][:, :] | None,
    ):
        """Replicate pose-independent constraints and optional parameters."""
        atoms = replicate_constraint_atoms(n_poses, c_atms.numpy(force=True))
        return _tensor(atoms, c_atms.device), (
            None if c_params is None else c_params.repeat(n_poses, 1)
        )
