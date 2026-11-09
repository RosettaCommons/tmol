"""Runtime adapter for the shared residue/group rotamer geometry preparation."""

import attr
import jax

from tmol_core.pack.kinematics import (
    annotate_residue_rotamer_kinematics,
    pack_rotamer_kinematics,
    prepare_block_group_rotamer_kinematics,
)
from tmol_jax.kinematics._host import NativePosePlanner


def prepare_rotamer_kinematics(packed_block_types):
    """Prepare shared residue-local trees and place ideal DOFs on the JAX device.

    Native CPU initialization does not compile once per residue shape. This
    prepares geometry only; sampling, packing and design are separate APIs.
    """
    pbt = packed_block_types
    if not hasattr(pbt, "rotamer_kinforest"):
        inverse = NativePosePlanner().inverse_kinematics
        for block in pbt.active_block_types:
            annotate_residue_rotamer_kinematics(block, inverse)
        packed = pack_rotamer_kinematics(pbt.active_block_types)
        pbt.rotamer_kinforest = attr.evolve(
            packed, dofs_ideal=jax.device_put(packed.dofs_ideal, pbt.n_atoms.device)
        )
    return pbt.rotamer_kinforest


def prepare_block_group_kinematics(block_types, links, *, anchor=0):
    """Return the common host tree/offsets for a chemically connected group."""
    return prepare_block_group_rotamer_kinematics(
        block_types,
        links,
        anchor=anchor,
        inverse_kinematics=NativePosePlanner().inverse_kinematics,
    )
