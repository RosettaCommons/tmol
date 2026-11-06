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

    Supply the complete type name, including variants. Input atom order is
    arbitrary; every atom is required unless the loader explicitly enables
    missing-leaf reconstruction. No residue identity or bond is inferred;
    parameterization belongs to the chemical database.
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
    build_missing_leaf_atoms=False,
    build_missing_context=False,
):
    """Build a heterogeneous batch of fully specified residue blocks.

    ``blocks`` is a sequence of poses, each a sequence of :class:`ResidueBlock`.
    ``connections`` supplies ``(block, port, block, port)`` edges per pose,
    using connection names from the residue definitions. Omitted connections
    mean disconnected blocks, including adjacent polymer residues. This input
    path supports custom residue types without coercing them to amino acids.

    Reuse a returned context only with the same database and residue palette.
    Atom layout, connectivity and batch changes require a new rendered scorer.
    With ``build_missing_leaf_atoms=True``, omitted leaf atoms use the shared
    native construction frames. Observed coordinates remain exact. Unknown or
    duplicate atom names and missing non-leaf atoms still raise; this does not
    perform general missing-heavy-atom or ligand generation.
    Set ``build_missing_context=True`` to also build anchored missing heavy atoms
    with the shared context planner; unanchored or unavailable frames still raise.
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
    complete_missing = build_missing_leaf_atoms or build_missing_context
    type_indices = {rt.name: i for i, rt in enumerate(pbt.active_block_types)}
    rows, coordinates, availability = [], [], []
    if complete_missing:
        from tmol_core.io.missing_atoms import prepare_leaf_atom_flags

        leaves = prepare_leaf_atom_flags(
            pbt.active_block_types, np.asarray(pbt.atom_is_hydrogen), pbt.max_n_atoms
        ).is_leaf_atom
    for pose in blocks:
        kinds, xyz, masks = [], [], []
        for block in pose:
            if block.residue_type not in type_indices:
                raise ValueError(
                    f"residue type absent from context: {block.residue_type}"
                )
            kind = type_indices[block.residue_type]
            expected = [atom.name for atom in pbt.active_block_types[kind].atoms]
            names = list(block.atom_names)
            missing_names, extra_names = set(expected) - set(names), set(names) - set(
                expected
            )
            if (
                len(set(names)) != len(names)
                or extra_names
                or (missing_names and not complete_missing)
            ):
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
            if complete_missing:
                present = np.array([name in indices for name in expected], bool)
                nonleaf = ~present & ~leaves[kind, : len(expected)]
                if nonleaf.any() and not build_missing_context:
                    raise ValueError(
                        f"missing non-leaf atoms on {block.residue_type}: "
                        + ", ".join(np.asarray(expected)[nonleaf])
                    )
                packed = np.zeros((len(expected), 3), coords.dtype)
                packed[present] = coords[
                    [indices[name] for name in expected if name in indices]
                ]
                xyz.append(packed)
                masks.append(present)
            else:
                xyz.append(coords[[indices[name] for name in expected]])
            kinds.append(kind)
        rows.append(kinds)
        availability.append(masks)
        coordinates.append(np.concatenate(xyz) if xyz else np.empty((0, 3), np.float32))
    pose = PoseStackBuilder.from_block_types_and_coords(
        pbt,
        rows,
        coordinates,
        [[str(i) for i in range(len(p))] for p in blocks],
        [list(range(1, len(p) + 1)) for p in blocks],
        connections_per_pose=[()] * len(blocks) if connections is None else connections,
    )
    if complete_missing:
        from ._missing import complete_missing_atoms

        pose = complete_missing_atoms(
            pose, availability, build_context=build_missing_context
        )
    return (pose, context) if return_context else pose
