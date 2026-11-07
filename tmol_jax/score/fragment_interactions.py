"""JAX execution of the common original-fragment attribution contract."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.scoring.fragment_interactions import (
    FragmentInteractionScores,
    fragment_mapping_plan,
)


@partial(jax.jit, static_argnames=("n_fragments", "sum_terms"))
def _fragment_values(values, indices, columns, partner_mask, *, n_fragments, sum_terms):
    poses, left, right = indices
    result = jnp.zeros(
        (values.shape[0], partner_mask.shape[0], n_fragments), values.dtype
    )
    if n_fragments:
        for fragment, partner in ((left, right), (right, left)):
            column = columns[fragment]
            selected = (column >= 0) & partner_mask[poses, partner]
            result = result.at[:, poses, jnp.maximum(column, 0)].add(
                jnp.where(selected[None], values, 0)
            )
    return result.sum(0) if sum_terms else result


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
        from .block_pair import BlockPairScorer

        self.scorer = BlockPairScorer(
            sfxn,
            pose,
            interaction_only=True,
            _partner_mask=np.asarray(partner_mask),
            _fragment_blocks=plan.fragment_block_indices,
        )
        columns = np.full(pose.block_type_ind.shape[1], -1, np.int32)
        columns[list(plan.fragment_block_indices)] = np.arange(len(self.mapping))
        self.columns = jax.device_put(columns, pose.device)
        self.sum_terms = sum_terms

    def __call__(self, coords):
        values, indices = self.scorer.pair_values(coords)
        return _fragment_values(
            values,
            indices,
            self.columns,
            self.partner_mask,
            n_fragments=self.indices.size,
            sum_terms=self.sum_terms,
        )


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
