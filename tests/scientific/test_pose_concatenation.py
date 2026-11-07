"""Concatenation preserves molecular identity, constraints and physical scores."""

from contextlib import nullcontext
from importlib import import_module

import numpy as np
import pytest

from .constraint_execution import Constraints
from .molecules import BACKBONE_CASES, DOMAIN_CASES


@pytest.mark.parametrize("case", [DOMAIN_CASES["mixed_batch"], BACKBONE_CASES["mixed"]])
def test_concatenation_reorders_molecular_batches_and_constraints(backend, case):
    b = backend.molecule(case, terms=("constraint",))
    Constraints(b).attach(
        [
            (
                "harmonic",
                np.array([[[0, 0, 0], [0, 0, 1]], [[1, 0, 0], [1, 0, 1]]]),
                np.array([[1.3, 0.4], [1.5, 0.6]]),
            )
        ]
    )
    original = b.pose
    front = import_module("tmol_jax" if b.framework == "jax" else "tmol")

    def take(index):
        return original[index] if b.framework == "jax" else original.split(index)

    first, second = take(0), take(1)
    combined = front.PoseStackBuilder.from_poses([second, first, second], b.device)
    order = [1, 0, 1]
    for field in ("coords", "block_coord_offset", "inter_residue_connections"):
        np.testing.assert_array_equal(
            b.numpy(getattr(combined, field)), b.numpy(getattr(original, field))[order]
        )
    np.testing.assert_array_equal(
        combined.pdb_info.residue_labels, original.pdb_info.residue_labels[order]
    )
    np.testing.assert_array_equal(
        combined.pdb_info.atom_occupancy, original.pdb_info.atom_occupancy[order]
    )
    assert combined.constraint_set.constraint_atoms.shape[0] == 3
    b.use_beta2016()
    scoretype = front.ScoreType
    b.score_function.set_weight(scoretype.constraint, 0.7)
    if b.framework == "torch":
        b._render_torch()
    precision = b.module.enable_x64() if b.framework == "jax" else nullcontext()
    with precision:
        expected, expected_grad = b.channel_values_and_gradients(
            b.numpy(original.coords).astype(np.float64)
        )
        b.pose = combined
        if b.framework == "torch":
            b._render_torch()
        actual, actual_grad = b.channel_values_and_gradients(
            b.numpy(combined.coords).astype(np.float64)
        )
    np.testing.assert_allclose(actual, expected[:, order], rtol=2e-10, atol=2e-10)
    np.testing.assert_allclose(
        actual_grad, expected_grad[:, order], rtol=2e-10, atol=2e-10
    )
    assert abs(actual).max() > 1


def test_concatenation_merges_palettes_and_retains_source_coordinate_derivatives(
    backend,
):
    from .execution import Execution
    from .molecules import MolecularCase

    b = backend.molecule(DOMAIN_CASES["mixed_batch"], terms=())
    chemistry = b.parameter_database
    left = Execution(b.name).molecule(
        MolecularCase((), (("LG1",),), ((1,),)), terms=(), parameter_database=chemistry
    )
    right = Execution(b.name).molecule(
        DOMAIN_CASES["d_peptide"], terms=(), parameter_database=chemistry
    )
    front = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    combine = front.PoseStackBuilder.from_poses
    poses = [left.pose, right.pose, left.pose]
    combined = combine(poses, b.device)
    types = b.numpy(combined.block_type_ind)
    names = [
        [combined.packed_block_types.active_block_types[t].name for t in row if t >= 0]
        for row in types
    ]
    assert names == [["LG1"], ["DALA", "DPRO", "DVAL"], ["LG1"]]
    for p, original in enumerate(poses):
        np.testing.assert_array_equal(
            b.numpy(combined.coords)[p, : original.coords.shape[1]],
            b.numpy(original.coords)[0],
        )
    if b.framework == "jax":

        def objective(x):
            pose = left.pose.replace(coords=x)
            return combine([pose, right.pose, pose], b.device).coords.sum()

        gradient = b.module.jit(b.module.grad(objective))(left.pose.coords)
    else:
        import attrs

        x = left.pose.coords.detach().clone().requires_grad_()
        pose = attrs.evolve(left.pose, coords=x)
        value = combine([pose, right.pose, pose], b.device).coords.sum()
        gradient = b.module.autograd.grad(value, x)[0]
    np.testing.assert_array_equal(b.numpy(gradient), 2)


def test_concatenation_keeps_fragment_group_identity(backend):
    from .fragment_molecules import fragment_fixture
    from .test_fragment_input import _build, _frontend

    _, atoms, parameters = fragment_fixture(repeated=True, mixed=False)
    pose, _ = _build(backend, atoms, parameters)
    joined = _frontend(backend).PoseStackBuilder.from_poses(
        [pose, pose], backend.device
    )
    source = pose.split_block_mapping.entries
    entries = joined.split_block_mapping.entries
    assert len(entries) == 2 * len(source)
    for p in range(2):
        actual = [e for e in entries if e.pose_ind == p]
        assert [(e.group_ind, e.block_ind, e.orig_residue_label) for e in actual] == [
            (e.group_ind, e.block_ind, e.orig_residue_label) for e in source
        ]
        for e in actual:
            assert (
                joined.packed_block_types.active_block_types[e.orig_block_type_ind].name
                == pose.packed_block_types.active_block_types[
                    source[actual.index(e)].orig_block_type_ind
                ].name
            )
    np.testing.assert_array_equal(
        backend.numpy(joined.coords)[0], backend.numpy(joined.coords)[1]
    )
