"""Block-pair execution through the existing shared native conformer kernels."""

import jax
import jax.numpy as jnp
import numpy as np

from tmol_core.pack.conformer_merge import prepare_conformer_origins
from tmol_core.pack.correlation import correlation_indices
from tmol_jax.pack.rotamer_set import RotamerSet, ConformerContext
from tmol_jax.pack.score_function import RotamerScorer


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

    def __init__(self, sfxn, pose, *, interaction_only=False):
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
                i, j = np.triu_indices(int(count))
                rows.append(np.stack((np.full(i.size, p), i + offset, j + offset)))
            host_pairs = (
                np.concatenate(rows, axis=1) if rows else np.empty((3, 0), np.int32)
            )
            pairs = jax.device_put(host_pairs.astype(np.int32), pose.device)
        self.sparse = sparse
        self.candidates = tuple(pairs if entry[5] else None for entry in sparse.entries)
        self.blocks = rotamers.block_ind_for_rot
        self.weights = tuple(sfxn.get_weight(st) for st in sparse.score_types)

    def __call__(self, coords, sum_terms=True, apply_weights=True):
        if coords.shape != self.shape:
            raise ValueError(f"expected coordinates with shape {self.shape}")
        values, indices = self.sparse._evaluate(
            coords.reshape(-1, 3),
            arrays=self.sparse._arrays,
            candidates=self.candidates,
        )
        poses, blocks = self.block_shape
        matrix = jnp.zeros((len(self.weights), poses, blocks, blocks), coords.dtype)
        p, i, j = indices
        matrix = matrix.at[:, p, self.blocks[i], self.blocks[j]].add(values)
        if self.interaction_only:
            diagonal = jnp.arange(blocks)
            matrix = matrix.at[:, :, diagonal, diagonal].set(0)
        if apply_weights:
            matrix *= jnp.asarray(self.weights, coords.dtype)[:, None, None, None]
        return matrix.sum(0) if sum_terms else matrix
