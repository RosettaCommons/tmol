"""Common molecular selection and repetition preserve topology and cotangents."""

from contextlib import nullcontext
from importlib import import_module

import attrs
import numpy as np
import pytest

from .constraint_execution import Constraints
from .molecules import BACKBONE_CASES, DATA


def dense(b, pose):
    value = pose.inter_block_bondsep
    return value.to_dense() if b.framework == "torch" else value


@pytest.fixture
def selection_backend(backend, request):
    b = backend
    if request.param == "mixed":
        return b.molecule(BACKBONE_CASES["mixed"], terms=("constraint",)), request.param
    front = import_module("tmol_jax" if b.framework == "jax" else "tmol")
    b.parameter_database = front.ParameterDatabase.get_default()
    b.score_function = front.ScoreFunction(
        b.parameter_database, *((b.device,) if b.framework == "torch" else ())
    )
    b.score_function.set_weight(front.ScoreType.constraint, 1.0)
    poses = [
        front.pose_stack_from_file(
            DATA / "pdb/1ubq.pdb",
            b.device,
            residue_start=0,
            residue_end=end,
            no_optH=True,
            missing_density_distance_threshold=0.0,
        )
        for end in (40, 60)
    ]
    b.pose = front.PoseStackBuilder.from_poses(poses, b.device)
    return b, request.param


@pytest.mark.parametrize(
    "selection_backend,blocks",
    [
        ("mixed", [3, 0, 2]),
        ("mixed", [-1, 1, 2]),
        ("mixed", [3, 3]),
        ("ubq", [3, 4, 9]),
        ("ubq", [9, 3, 4]),
    ],
    indirect=["selection_backend"],
)
def test_select_repeat_preserves_molecular_identity_paths_and_cotangents(
    selection_backend, blocks
):
    b, case = selection_backend
    front = import_module("tmol_jax.pose" if b.framework == "jax" else "tmol.pose")
    pose = b.pose.split(0)
    source = np.concatenate(
        [
            np.arange(
                int(b.numpy(pose.block_coord_offset)[0, block]),
                int(b.numpy(pose.block_coord_offset)[0, block])
                + pose.block_type(0, block).n_atoms,
            )
            for block in blocks
        ]
    )
    selected = front.select_blocks(pose, blocks)
    assert selected.max_n_blocks == len(blocks)
    assert selected.coords.dtype == pose.coords.dtype
    assert selected.coords.device == pose.coords.device
    if blocks == [3, 4, 9]:
        partners = b.numpy(selected.inter_residue_connections)[0, :, :, 0]
        assert set(partners.flatten()) <= {-1, 0, 1}
        assert (partners[0] == 1).any() and (partners[1] == 0).any()
        assert (partners[2] == -1).all()
    np.testing.assert_array_equal(
        b.numpy(selected.coords), b.numpy(pose.coords)[:, source]
    )
    actual, original = dense(b, selected), dense(b, pose)
    assert actual.dtype == original.dtype and actual.device == original.device
    np.testing.assert_array_equal(
        b.numpy(actual), b.numpy(original)[:, blocks][:, :, blocks]
    )
    np.testing.assert_array_equal(
        b.numpy(selected.chain_id), b.numpy(pose.chain_id)[:, blocks]
    )
    np.testing.assert_array_equal(
        selected.pdb_info.residue_labels, pose.pdb_info.residue_labels[:, blocks]
    )
    repeated = front.repeat_pose_stack(selected, 3)
    np.testing.assert_array_equal(
        b.numpy(repeated.coords), np.repeat(b.numpy(selected.coords), 3, axis=0)
    )
    precision = b.module.enable_x64() if b.framework == "jax" else nullcontext()
    with precision:
        coords = b.numpy(pose.coords).astype(np.float64)
        weights = np.arange(
            repeated.coords.size if b.framework == "jax" else repeated.coords.numel(),
            dtype=np.float64,
        ).reshape(repeated.coords.shape)
        expected = np.zeros_like(coords)
        np.add.at(expected[0], source, weights.sum(axis=0))
        if b.framework == "jax":
            x = b.module.device_put(coords, b.device)
            w = b.module.device_put(weights, b.device)

            def objective(x):
                out = front.repeat_pose_stack(
                    front.select_blocks(pose.replace(coords=x), blocks), 3
                )
                return (out.coords * w).sum()

            gradient = b.module.grad(objective)(x)
            out = front.repeat_pose_stack(
                front.select_blocks(pose.replace(coords=x), blocks), 3
            )
        else:
            x = b.module.tensor(coords, device=b.device, requires_grad=True)
            w = b.module.as_tensor(weights, device=b.device)
            selected_input = front.select_blocks(attrs.evolve(pose, coords=x), blocks)
            out = front.repeat_pose_stack(selected_input, 3)
            gradient = b.module.autograd.grad((out.coords * w).sum(), x)[0]
        assert out.coords.dtype == x.dtype and out.coords.device == x.device
        assert gradient.dtype == x.dtype and gradient.device == x.device
        np.testing.assert_array_equal(b.numpy(gradient), expected)
        if b.framework == "torch":
            out.coords[1].detach().zero_()
            np.testing.assert_array_equal(
                b.numpy(out.coords[:1]), b.numpy(selected_input.coords)
            )
    if case == "mixed" and len(set(blocks)) == len(blocks):
        b.pose = selected
        b.use_beta2016()
        value, gradient = b.channel_values_and_gradients(b.coords())
        assert value.shape[0] == 25
        assert np.isfinite(value).all() and np.isfinite(gradient).all()
        assert np.linalg.norm(gradient) > 0


@pytest.mark.parametrize("selection_backend", ("mixed", "ubq"), indirect=True)
def test_repeat_constraints_and_selection_limits(selection_backend):
    b, case = selection_backend
    ubq = case == "ubq"
    atoms = [[[0, 3, 0], [0, 4, 0]]] if ubq else [[[0, 0, 0], [0, 0, 1]]]
    params = np.array([[1.47, 0.1] if ubq else [1.3, 0.4]], np.float32)
    constraints = Constraints(b).attach([("harmonic", np.array(atoms), params)])
    front = import_module("tmol_jax.pose" if b.framework == "jax" else "tmol.pose")
    pose = b.pose.split(0)
    repeated = front.repeat_pose_stack(pose, 3)
    builder = import_module(
        "tmol_jax" if b.framework == "jax" else "tmol"
    ).PoseStackBuilder
    expected = builder.from_poses([pose] * 3, b.device)
    assert repeated.n_poses == 3
    for field in (
        "coords",
        "block_type_ind",
        "block_coord_offset",
        "inter_residue_connections",
        "chain_id",
    ):
        field += "64" if b.framework == "torch" and field != "coords" else ""
        actual, wanted = getattr(repeated, field), getattr(expected, field)
        assert actual.dtype == wanted.dtype and actual.device == wanted.device
        np.testing.assert_array_equal(b.numpy(actual), b.numpy(wanted))
    actual, wanted = dense(b, repeated), dense(b, expected)
    assert actual.dtype == wanted.dtype and actual.device == wanted.device
    np.testing.assert_array_equal(b.numpy(actual), b.numpy(wanted))
    np.testing.assert_array_equal(
        repeated.pdb_info.residue_labels, expected.pdb_info.residue_labels
    )
    for field in ("dtype", "device"):
        assert getattr(repeated.constraint_set.constraint_params, field) == getattr(
            constraints.constraint_params, field
        )
    np.testing.assert_array_equal(
        b.numpy(repeated.constraint_set.constraint_params), np.repeat(params, 3, axis=0)
    )
    assert (
        repeated.constraint_set.constraint_functions == constraints.constraint_functions
    )
    atoms = repeated.constraint_set.constraint_atoms
    assert atoms.device == b.device
    atoms = b.numpy(atoms)
    assert atoms.dtype == np.int32
    np.testing.assert_array_equal(
        atoms[:, :2, 0], np.repeat(np.arange(3)[:, None], 2, axis=1)
    )
    assert front.select_blocks(pose, [3, 4] if ubq else [0, 1]).constraint_set is None
    original = pose.pdb_info.residue_labels.copy()
    repeated.pdb_info.residue_labels[1] = -99
    np.testing.assert_array_equal(pose.pdb_info.residue_labels, original)
    np.testing.assert_array_equal(repeated.pdb_info.residue_labels[0], original[0])
    with pytest.raises(ValueError, match="single reference"):
        front.repeat_pose_stack(b.pose, 2)
    if case == "mixed":
        with pytest.raises(ValueError, match="homogeneous"):
            front.select_blocks(b.pose, [0])


@pytest.mark.parametrize("field", ("residue_labels", "chain_labels"))
def test_repeat_retains_absent_labels_and_independent_present_metadata(backend, field):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    front = import_module("tmol_jax.pose" if b.framework == "jax" else "tmol.pose")
    pose = b.pose.split(0)
    metadata = attrs.evolve(pose.pdb_info, **{field: None})
    pose = (
        pose.replace(pdb_info=metadata)
        if b.framework == "jax"
        else attrs.evolve(pose, pdb_info=metadata)
    )
    repeated = front.repeat_pose_stack(pose, 3)
    assert getattr(repeated.pdb_info, field) is None
    np.testing.assert_array_equal(
        b.numpy(repeated.coords), np.repeat(b.numpy(pose.coords), 3, axis=0)
    )
    present = "chain_labels" if field == "residue_labels" else "residue_labels"
    original = np.asarray(getattr(pose.pdb_info, present)).copy()
    actual = getattr(repeated.pdb_info, present)
    np.testing.assert_array_equal(actual, np.repeat(original, 3, axis=0))
    actual[1, 0] = "changed" if present == "chain_labels" else -99
    np.testing.assert_array_equal(getattr(pose.pdb_info, present), original)
    np.testing.assert_array_equal(actual[0], original[0])
