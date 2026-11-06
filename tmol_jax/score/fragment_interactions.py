"""JAX execution of the common original-fragment attribution contract."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.fragment_interactions import (
    FragmentInteractionScores,
    fragment_mapping_plan,
    fragments_per_chunk,
)


class FragmentScorer:
    """Prepared fragment/partner identities with differentiable coordinates."""

    def __init__(self, pose, partner_mask, sfxn, mapping, sum_terms):
        mapping = pose.split_block_mapping if mapping is None else mapping
        if mapping is None:
            raise ValueError(
                "No split-block mapping was supplied or attached to the pose"
            )
        if partner_mask.shape != pose.block_type_ind.shape:
            raise ValueError("partner_mask must have shape [n_poses, max_n_blocks]")
        if partner_mask.dtype != jnp.bool_:
            raise TypeError("partner_mask must be a boolean tensor")
        if (
            not isinstance(partner_mask, jax.Array)
            or partner_mask.device != pose.device
        ):
            raise ValueError("partner_mask must be on the same device as pose_stack")
        if sfxn is None:
            raise ValueError(
                "sfxn is required and must use the ligand-extended parameter database"
            )
        plan = fragment_mapping_plan(mapping, *pose.block_type_ind.shape)
        if np.asarray(partner_mask).reshape(-1)[list(plan.record_linear_indices)].any():
            raise ValueError("partner_mask must not include ligand fragment blocks")
        self.mapping = plan.fragment_records
        self.indices = jax.device_put(
            np.asarray(plan.fragment_block_indices, np.int32), pose.device
        )
        self.partner_mask = partner_mask
        self.scorer = sfxn.render_block_pair_scoring_module(pose, interaction_only=True)
        self.sum_terms = sum_terms

    def __call__(self, coords):
        matrix = self.scorer(coords, sum_terms=False)
        terms, poses, blocks, _ = matrix.shape
        fragments = self.indices.size
        if not terms or not poses or not blocks or not fragments:
            result = jnp.broadcast_to(
                matrix.sum((-2, -1))[..., None], (terms, poses, fragments)
            )
        else:
            size = fragments_per_chunk(
                terms, poses, blocks, coords.dtype.itemsize, 256 * 1024 * 1024
            )
            chunks = []
            for start in range(0, fragments, size):
                ids = self.indices[start : start + size]
                selected = jnp.take(matrix, ids, axis=2) + jnp.take(
                    matrix, ids, axis=3
                ).swapaxes(-1, -2)
                chunks.append(
                    jnp.einsum(
                        "tpfb,pb->tpf", selected, self.partner_mask.astype(coords.dtype)
                    )
                )
            result = jnp.concatenate(chunks, axis=2)
        return result.sum(0) if self.sum_terms else result


def render_fragment_interactions(
    pose_stack, partner_mask, *, sfxn, mapping=None, sum_terms=False
):
    """Prepare once; the returned scorer supports jit/vmap and first reverse AD.

    Its ``mapping`` defines output columns. Re-render after changing topology,
    score configuration or selected partners. Physical neighbors remain dynamic.
    """
    return FragmentScorer(pose_stack, partner_mask, sfxn, mapping, sum_terms)


def calculate_fragment_interactions(
    pose_stack, partner_mask, *, sfxn, mapping=None, sum_terms=False
):
    """Eager convenience returning per-fragment partner scores and column records."""
    scorer = render_fragment_interactions(
        pose_stack, partner_mask, sfxn=sfxn, mapping=mapping, sum_terms=sum_terms
    )
    return FragmentInteractionScores(scorer(pose_stack.coords), scorer.mapping)
