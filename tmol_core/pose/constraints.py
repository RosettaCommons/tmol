"""Constraint atom topology, independent of parameter storage and callbacks.

Indices describe pose/block/atom triples. Parameters remain frontend arrays so
assembly preserves differentiation with respect to user-supplied parameters.
"""

from dataclasses import dataclass, fields
import numpy as np

MAX_N_ATOMS = 4


def count_unique_blocks(atom_indices):
    """Count block runs, preserving TMol's one-/two-body packing convention."""
    blocks = np.asarray(atom_indices)[:, :, 1]
    return np.sum(blocks[:, 1:] != blocks[:, :-1], axis=1, dtype=np.int32) + 1


def replicate_constraint_atoms(n_poses, atoms):
    atoms = np.asarray(atoms, dtype=np.int32)
    result = np.tile(atoms, (n_poses, 1, 1))
    result[:, :, 0] = np.repeat(np.arange(n_poses, dtype=np.int32), len(atoms))[:, None]
    return result


@dataclass(frozen=True)
class ConstraintTopology:
    n_poses: int
    constraint_function_inds: np.ndarray
    constraint_atoms: np.ndarray
    constraint_num_unique_blocks: np.ndarray
    constraint_unique_blocks: np.ndarray

    def __post_init__(self):
        if (
            not isinstance(self.n_poses, (int, np.integer))
            or not 0 <= self.n_poses <= np.iinfo(np.int32).max
        ):
            raise ValueError("n_poses must be a nonnegative int32 count")

    @classmethod
    def empty(cls, n_poses):
        return cls(
            n_poses,
            np.zeros(0, np.int32),
            np.zeros((0, 4, 3), np.int32),
            np.zeros(0, np.int32),
            np.zeros((0, 3), np.int32),
        )

    def arrays(self):
        return {
            f.name: getattr(self, f.name) for f in fields(self) if f.name != "n_poses"
        }

    def add(self, atom_indices, function_index, *, replace_empty=False):
        """Return (topology, repeat count for the new parameter rows).

        A two-column block/atom input repeats over every pose; a three-column
        input already identifies its poses. ``replace_empty`` retains the old
        public API's first-add behavior and its parameter width semantics.
        """
        atoms = np.asarray(atom_indices)
        if atoms.ndim != 3 or atoms.shape[2] not in (2, 3):
            raise ValueError("atom_indices must have shape [constraint, atom, 2 or 3]")
        if not 0 < atoms.shape[1] <= MAX_N_ATOMS:
            raise ValueError(
                f"constraints must contain between 1 and {MAX_N_ATOMS} atoms"
            )
        if atoms.dtype not in (np.int32, np.int64):
            raise TypeError("atom_indices must contain 32- or 64-bit integers")
        if np.any(atoms < 0) or np.any(atoms > np.iinfo(np.int32).max):
            raise ValueError(
                "input pose/block/atom indices must be nonnegative int32 values"
            )
        if atoms.shape[2] == 3 and np.any(atoms[:, :, 0] >= self.n_poses):
            raise ValueError("constraint pose indices must be less than n_poses")
        atoms = atoms.astype(np.int32, copy=False)
        repeats = 1
        if atoms.shape[2] == 2:
            filled = np.zeros((*atoms.shape[:2], 3), np.int32)
            filled[:, :, 1:] = atoms
            atoms = replicate_constraint_atoms(self.n_poses, filled)
            repeats = self.n_poses
        if np.any(atoms[:, :, 0] != atoms[:, :1, 0]):
            raise ValueError(
                "One or more constraints contains atoms from multiple poses"
            )
        blocks = atoms[:, :, 1]
        changed = blocks != np.roll(blocks, 1, axis=1)
        changed[:, 0] = False
        second = blocks[np.arange(len(atoms)), np.argmax(changed, axis=1)]
        unique = np.stack((atoms[:, 0, 0], blocks[:, 0], second), axis=1)
        padded = np.full((len(atoms), MAX_N_ATOMS, 3), -1, np.int32)
        padded[:, : atoms.shape[1]] = atoms
        padded[:, -1] = atoms[:, -1]
        new = ConstraintTopology(
            self.n_poses,
            np.full(len(atoms), function_index, np.int32),
            padded,
            count_unique_blocks(atoms),
            unique,
        )
        if replace_empty:
            return new, repeats
        return (
            ConstraintTopology(
                self.n_poses,
                **{
                    k: np.concatenate((getattr(self, k), v))
                    for k, v in new.arrays().items()
                },
            ),
            repeats,
        )

    def split(self, index):
        """Return (single-pose topology, parameter-row indices)."""
        rows = np.flatnonzero(np.any(self.constraint_atoms[:, :, 0] == index, axis=1))
        arrays = {k: v[rows].copy() for k, v in self.arrays().items()}
        poses = arrays["constraint_atoms"][:, :, 0]
        poses[poses != -1] = 0
        arrays["constraint_unique_blocks"][:, 0] = 0
        return ConstraintTopology(1, **arrays), rows

    @classmethod
    def concatenate(
        cls,
        topologies,
        function_remaps,
        *,
        from_multiple_pose_stacks=True,
        n_poses=None,
        ps_offset=None,
    ):
        """Concatenate row order; remaps translate each source callback registry.

        ``None`` entries are retained in the offset/remap sequences. The caller
        concatenates its differentiable parameter arrays in the same row order.
        """
        present = [t for t in topologies if t is not None]
        if not present:
            return None
        if n_poses is None:
            n_poses = (
                sum(t.n_poses for t in present)
                if from_multiple_pose_stacks
                else present[0].n_poses
            )
        if ps_offset is None:
            sizes = [t.n_poses if t is not None else 0 for t in topologies]
            ps_offset = (
                np.cumsum([0, *sizes[:-1]])
                if from_multiple_pose_stacks
                else np.zeros(len(topologies), np.int64)
            )
        parts = []
        for t, remap, offset in zip(topologies, function_remaps, ps_offset):
            if t is None:
                continue
            arrays = {k: v.copy() for k, v in t.arrays().items()}
            arrays["constraint_function_inds"] = np.asarray(remap, dtype=np.int32)[
                t.constraint_function_inds
            ]
            poses = arrays["constraint_atoms"][:, :, 0]
            poses[poses != -1] += offset
            arrays["constraint_unique_blocks"][:, 0] += offset
            parts.append(arrays)
        return cls(
            n_poses, **{k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        )


def merge_constraint_functions(registries):
    """Preserve public callback equality/deduplication and source order."""
    functions, remaps = [], []
    for registry in registries:
        remap = []
        for function in registry or ():
            try:
                index = functions.index(function)
            except ValueError:
                index = len(functions)
                functions.append(function)
            remap.append(index)
        remaps.append(np.asarray(remap, dtype=np.int32))
    return tuple(functions), remaps
