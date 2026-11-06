"""Torch storage adapter for common physical chain membership."""

import torch
from tmol_core.io.chain_deduction import chain_indices, connection_masks


def _chain_inds_for_pose_stack(pose_stack):
    """Label physical polymer chains; nonpolymer bonds join their components."""
    pbt = pose_stack.packed_block_types
    _annotate_pbt_w_valid_connection_masks(pbt)
    return chain_indices(
        pose_stack.block_type_ind.cpu().numpy(),
        pose_stack.inter_residue_connections.cpu().numpy(),
        pbt.connection_mask_for_chain_detection.cpu().numpy(),
    )


def _annotate_pbt_w_valid_connection_masks(pbt):
    if not hasattr(pbt, "connection_mask_for_chain_detection"):
        pbt.connection_mask_for_chain_detection = torch.as_tensor(
            connection_masks(pbt.active_block_types, pbt.max_n_conn), device=pbt.device
        )
