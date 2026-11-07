"""Block-pair execution through the existing shared native conformer kernels."""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.conformer_merge import prepare_conformer_origins
from tmol_core.pack.correlation import correlation_indices
from tmol_jax.pack.rotamer_set import RotamerSet, ConformerContext
from tmol_jax.pack.score_function import RotamerScorer
from ._execution import render_execution

_INTERACTION_EXECUTORS = {}


@partial(jax.jit, static_argnames=("weights", "interaction_only"))
def _physical_pair_values(values, indices, blocks, *, weights, interaction_only):
    p, i, j = indices
    i, j = blocks[i], blocks[j]
    if interaction_only:
        values = jnp.where((i != j)[None], values, 0)
    if weights is not None:
        values *= jnp.asarray(weights, values.dtype)[:, None]
    return values, (p, i, j)


def _pair_matrix(values, indices, block_shape, sum_terms):
    p, i, j = indices
    poses, blocks = block_shape
    matrix = jnp.zeros((values.shape[0], poses, blocks, blocks), values.dtype)
    matrix = matrix.at[:, p, i, j].add(values)
    return matrix.sum(0) if sum_terms else matrix


def _current_conformers(pose):
    """View each physical block as one conformation without copying coordinates."""
    types = np.asarray(pose.block_type_ind)
    p, b = np.nonzero(types >= 0)
    origins = prepare_conformer_origins(types.shape, p, b, np.arange(len(p)))

    def integer(value):
        values = np.asarray(value)
        if values.size and (values.min() < -(1 << 31) or values.max() >= 1 << 31):
            raise OverflowError("block-pair identities exceed native int32 capacity")
        return jax.device_put(values.astype(np.int32), pose.device)

    counts_pose, offsets_pose, counts, offsets, _, _ = origins
    return RotamerSet(
        n_rots_for_pose=integer(counts_pose),
        rot_offset_for_pose=integer(offsets_pose),
        n_rots_for_block=integer(counts),
        rot_offset_for_block=integer(offsets),
        pose_for_rot=integer(p),
        block_type_ind_for_rot=integer(types[p, b]),
        block_ind_for_rot=integer(b),
        coord_offset_for_rot=integer(
            np.asarray(pose.block_coord_offset)[p, b] + p * pose.coords.shape[1]
        ),
        coords=pose.coords.reshape(-1, 3),
        group_for_block=integer(correlation_indices(types.shape, (), counts)),
        first_rot_block_type=pose.block_type_ind,
        pose_ind_for_atom=integer(
            np.repeat(np.arange(types.shape[0]), pose.coords.shape[1])
        ),
        max_n_rots_per_pose=int(counts_pose.max(initial=0)),
        context=ConformerContext(pose),
    )


class BlockPairScorer:
    """Fixed topology with fresh coordinates, first reverse AD, jit and vmap.

    All physical pairs remain candidates so coordinate changes cannot invalidate
    a captured neighbor list. Native terms still apply their physical cutoffs.
    This dense output API has quadratic block storage.
    """

    def __init__(
        self,
        sfxn,
        pose,
        *,
        interaction_only=False,
        _partner_mask=None,
        _fragment_blocks=(),
    ):
        self.shape = pose.coords.shape
        self.block_shape = pose.block_type_ind.shape
        self.interaction_only = interaction_only
        with jax.default_device(pose.device):
            rotamers = _current_conformers(pose)
            sparse = RotamerScorer(sfxn, pose, rotamers, False, False)
            # Dense block output already has quadratic storage. Retaining all
            # upper-triangle identities makes this plan valid after arbitrary
            # coordinate changes, without a stale spatial-neighbor capture.
            counts = np.asarray(rotamers.n_rots_for_pose, dtype=np.int64)
            total = sum(int(n) * (int(n) + 1) // 2 for n in counts)
            if total > np.iinfo(np.int32).max:
                raise OverflowError(
                    "dense block-pair count exceeds native int32 capacity"
                )
            rows = []
            for p, (count, offset) in enumerate(
                zip(
                    counts,
                    np.asarray(rotamers.rot_offset_for_pose),
                )
            ):
                if _partner_mask is None:
                    i, j = np.triu_indices(int(count))
                else:
                    # Attribution only needs named fragment/partner pairs.
                    # Preserve every such pair after coordinate motion; native
                    # physics still applies its cutoff at execution time.
                    blocks = np.flatnonzero(np.asarray(pose.block_type_ind)[p] >= 0)
                    left, right = np.meshgrid(
                        np.searchsorted(blocks, _fragment_blocks),
                        np.flatnonzero(_partner_mask[p, blocks]),
                        indexing="ij",
                    )
                    pairs = np.unique(
                        np.sort(np.stack((left.ravel(), right.ravel())), axis=0), axis=1
                    )
                    i, j = pairs
                rows.append(np.stack((np.full(i.size, p), i + offset, j + offset)))
            host_pairs = (
                np.concatenate(rows, axis=1) if rows else np.empty((3, 0), np.int32)
            )
            pairs = jax.device_put(host_pairs.astype(np.int32), pose.device)
        self.sparse = sparse
        self.candidates = tuple(pairs if entry[5] else None for entry in sparse.entries)
        self.blocks = rotamers.block_ind_for_rot
        self.weights = tuple(sfxn.get_weight(st) for st in sparse.score_types)

    def pair_values(self, coords, apply_weights=True):
        """Return native contributions and physical block identities."""
        if coords.shape != self.shape:
            raise ValueError(f"expected coordinates with shape {self.shape}")
        values, indices = self.sparse._evaluate(
            coords.reshape(-1, 3),
            arrays=self.sparse._arrays,
            candidates=self.candidates,
        )
        return _physical_pair_values(
            values,
            indices,
            self.blocks,
            weights=self.weights if apply_weights else None,
            interaction_only=self.interaction_only,
        )

    def __call__(self, coords, sum_terms=True, apply_weights=True):
        values, indices = self.pair_values(coords, apply_weights)
        return _pair_matrix(values, indices, self.block_shape, sum_terms)

    def unweighted_scores(self, coords):
        """Return all retained channels with shape [channels, poses, blocks, blocks]."""
        return self(coords, sum_terms=False, apply_weights=False)

    def score_interactions(
        self, coords, block_pair_indices, *, sum_terms=True, apply_weights=True
    ):
        """Sum selected pairs, including repeated entries, for every pose.

        Indices have shape [pairs, 2] on the coordinate device. Negative indices
        follow ordinary array indexing. Invalid eager indices raise; invalid
        traced indices produce NaN instead of silently selecting a boundary pair.
        """
        self._validate_indices(coords, block_pair_indices)
        i, j = block_pair_indices.T
        scores = self(coords, sum_terms=sum_terms, apply_weights=apply_weights)
        return scores.at[..., i, j].get(mode="fill").sum(axis=-1)

    def _validate_indices(self, coords, block_pair_indices):
        if block_pair_indices.ndim != 2 or block_pair_indices.shape[1] != 2:
            raise ValueError("block_pair_indices must have shape [n_pairs, 2]")
        if block_pair_indices.dtype not in (jnp.int32, jnp.int64):
            raise TypeError("block_pair_indices must have an integer dtype")
        if not isinstance(block_pair_indices, jax.core.Tracer):
            if not isinstance(coords, jax.core.Tracer) and (
                block_pair_indices.device != coords.device
            ):
                raise ValueError(
                    "block_pair_indices must be on the same device as coordinates"
                )
            indices = np.asarray(block_pair_indices)
            blocks = self.block_shape[1]
            if np.any((indices < -blocks) | (indices >= blocks)):
                raise IndexError("block pair index out of range")

    def prepare_interactions(
        self, block_pair_indices, *, sum_terms=True, apply_weights=True
    ):
        """Bind a fixed selection for reusable compiled scores and gradients.

        Prepare outside transformations, then call the returned scorer,
        ``jitted()`` or ``value_and_grad()`` on fresh coordinates. A new selection
        gets fresh index operands while sharing the same executable when possible.
        """
        if isinstance(block_pair_indices, jax.core.Tracer):
            raise ValueError("prepare pair selections outside JAX transformations")
        self._validate_indices(self.blocks, block_pair_indices)
        evaluate = self.sparse._evaluate
        weights = self.weights if apply_weights else None
        interaction_only, shape = self.interaction_only, self.block_shape

        def execute(coords, state):
            arrays, candidates, blocks, selected = state
            values, indices = evaluate(coords.reshape(-1, 3), arrays, candidates)
            values, indices = _physical_pair_values(
                values,
                indices,
                blocks,
                weights=weights,
                interaction_only=interaction_only,
            )
            scores = _pair_matrix(values, indices, shape, sum_terms)
            i, j = selected.T
            return scores.at[..., i, j].get(mode="fill").sum(axis=-1)

        return render_execution(
            _INTERACTION_EXECUTORS,
            (evaluate, weights, interaction_only, shape, sum_terms),
            execute,
            (self.sparse._arrays, self.candidates, self.blocks, block_pair_indices),
            self.shape,
            native_only=False,
            # The native CPU calls already compile independently. Combining the
            # dense pair adjoint into one CPU executable increases warm latency.
            compile_gradient=self.blocks.device.platform != "cpu",
        )
