"""Explicit, lossless input for already parameterized molecular systems."""

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from tmol_jax.chemical.restypes import _refined_from_raw
from tmol_jax.pose.packed_block_types import PackedBlockTypes
from tmol_jax.pose.pose_stack_builder import PoseStackBuilder


@dataclass(frozen=True)
class ResidueBlock:
    """One exact database residue type, with coordinates labeled by atom name.

    Supply the complete type name, including variants, and every atom exactly
    once. Input atom order is arbitrary. No termini, hydrogens or bonds are
    inferred; parameterization belongs to the chemical database.
    """

    residue_type: str
    atom_names: Sequence[str]
    coords: Any


@dataclass(frozen=True)
class PoseBuildContext:
    """Keep the molecular palette and its full parameter database together.

    Pass ``context.parameter_database`` to the score-function constructor.
    Input construction does not certify support by every scoring term.
    """

    parameter_database: Any
    packed_block_types: PackedBlockTypes


def pose_stack_from_residue_blocks(
    blocks,
    *,
    parameter_database=None,
    context=None,
    connections=None,
    return_context=False,
):
    """Build a heterogeneous batch of fully specified residue blocks.

    ``blocks`` is a sequence of poses, each a sequence of :class:`ResidueBlock`.
    ``connections`` supplies ``(block, port, block, port)`` edges per pose,
    using connection names from the residue definitions. Omitted connections
    mean disconnected blocks, including adjacent polymer residues. This input
    path supports custom residue types without coercing them to amino acids.

    Reuse a returned context only with the same database and residue palette.
    Atom layout, connectivity and batch changes require a new rendered scorer.
    """
    if context is not None and parameter_database is not None:
        raise ValueError("pass either context or parameter_database")
    blocks = [list(pose) for pose in blocks]
    if not blocks or not any(blocks):
        raise ValueError("pose batch contains no residues")
    if context is None:
        if parameter_database is None:
            from tmol_jax.database import ParameterDatabase

            parameter_database = ParameterDatabase.get_default()
        names = dict.fromkeys(block.residue_type for pose in blocks for block in pose)
        available = {rt.name: rt for rt in parameter_database.chemical.residues}
        unknown = names.keys() - available.keys()
        if unknown:
            raise ValueError(f"unknown residue types: {sorted(unknown)}")
        types = [_refined_from_raw(available[name]) for name in names]
        context = PoseBuildContext(
            parameter_database,
            PackedBlockTypes.from_restypes(parameter_database.chemical, types),
        )
    pbt = context.packed_block_types
    type_indices = {rt.name: i for i, rt in enumerate(pbt.active_block_types)}
    rows, coordinates = [], []
    for pose in blocks:
        kinds, xyz = [], []
        for block in pose:
            if block.residue_type not in type_indices:
                raise ValueError(
                    f"residue type absent from context: {block.residue_type}"
                )
            kind = type_indices[block.residue_type]
            expected = [atom.name for atom in pbt.active_block_types[kind].atoms]
            names = list(block.atom_names)
            if len(set(names)) != len(names) or set(names) != set(expected):
                raise ValueError(
                    f"atoms must match {block.residue_type} exactly; "
                    f"missing={sorted(set(expected) - set(names))}, "
                    f"extra={sorted(set(names) - set(expected))}, "
                    f"duplicates={len(names) != len(set(names))}"
                )
            coords = np.asarray(block.coords)
            if coords.shape != (len(names), 3) or not np.isfinite(coords).all():
                raise ValueError(
                    "block coordinates must be finite with shape [atoms, 3]"
                )
            indices = {name: i for i, name in enumerate(names)}
            xyz.append(coords[[indices[name] for name in expected]])
            kinds.append(kind)
        rows.append(kinds)
        coordinates.append(np.concatenate(xyz) if xyz else np.empty((0, 3), np.float32))
    pose = PoseStackBuilder.from_block_types_and_coords(
        pbt,
        rows,
        coordinates,
        [[str(i) for i in range(len(p))] for p in blocks],
        [list(range(1, len(p) + 1)) for p in blocks],
        connections_per_pose=[()] * len(blocks) if connections is None else connections,
    )
    return (pose, context) if return_context else pose
