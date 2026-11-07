"""Logical chain identities survive layout changes independently of author labels."""

from importlib import import_module

import attrs
import numpy as np

from .molecules import BACKBONE_CASES


def test_logical_chains_survive_split_batching_and_author_label_changes(backend):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    api = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    expected = np.array([[0, 0, 0, 1], [0, 0, 0, -1]], np.int32)
    np.testing.assert_array_equal(b.numpy(b.pose.chain_id), expected)
    labels = attrs.evolve(b.pose.pdb_info, chain_labels=np.full((2, 4), "same-author"))
    custom = np.array([[9, 9, 9, 2], [7, 7, 7, -1]], np.int32)
    if b.framework == "jax":
        indices = b.module.device_put(custom, b.device)
        pose = b.pose.replace(chain_id=indices, pdb_info=labels)
    else:
        indices = b.module.as_tensor(custom, device=b.device)
        pose = attrs.evolve(
            b.pose,
            chain_id=indices,
            chain_id64=indices.to(b.module.int64),
            pdb_info=labels,
        )
    reordered = api.PoseStackBuilder.from_poses(
        [pose.split(1), pose.split(0)], b.device
    )
    np.testing.assert_array_equal(b.numpy(reordered.chain_id), custom[[1, 0]])
    np.testing.assert_array_equal(
        reordered.pdb_info.chain_labels, labels.chain_labels[[1, 0]]
    )
    np.testing.assert_array_equal(b.numpy(reordered.coords), b.coords()[[1, 0]])
    np.testing.assert_array_equal(b.numpy(b.pose.chain_id), expected)
