"""Concrete packing task rows and sampling policy, independent of execution."""

from dataclasses import dataclass
import operator

import numpy as np


def chi_sample_budget(expanded_limit, limit):
    """Validate the two positive integer limits without coercing real values."""
    try:
        if isinstance(expanded_limit, bool) or isinstance(limit, bool):
            raise TypeError
        expanded_limit, limit = operator.index(expanded_limit), operator.index(limit)
    except TypeError:
        raise ValueError("chi sample budget limits must be positive integers") from None
    if expanded_limit < 1 or limit < 1:
        raise ValueError("chi sample budget limits must be positive integers")
    return expanded_limit, limit


@dataclass(frozen=True)
class PackingTaskRows:
    real_block_pose: np.ndarray
    real_block_block: np.ndarray
    cons_bt_pose: np.ndarray
    cons_bt_block: np.ndarray
    cons_bt_which_block_type: np.ndarray
    cons_bt_block_type: np.ndarray
    global_block_ind_for_considered_block_types: np.ndarray
    allowed_bt_pose: np.ndarray
    allowed_bt_block: np.ndarray
    allowed_bt_which_block_type: np.ndarray
    allowed_bt_block_type: np.ndarray
    is_cons_bt_allowed: np.ndarray
    allowed_cons_bt: np.ndarray


def prepare_packing_task_rows(is_real_block, considered, allowed):
    """Map padded choices to stable pose/block/slot-ordered concrete rows.

    Disabled choices remain in the considered rows: background residues still
    need a conformation for scoring. The allowed rows preserve the task mask,
    including its padding; samplers select real considered rows separately.
    """
    real_pose, real_block = np.nonzero(is_real_block)
    pose, block, slot = np.nonzero(considered != -1)
    allowed_pose, allowed_block, allowed_slot = np.nonzero(allowed)
    selected = allowed[pose, block, slot]
    return PackingTaskRows(
        real_pose,
        real_block,
        pose,
        block,
        slot,
        considered[pose, block, slot],
        considered.shape[1] * pose + block,
        allowed_pose,
        allowed_block,
        allowed_slot,
        considered[allowed_pose, allowed_block, allowed_slot],
        selected,
        np.flatnonzero(selected),
    )
