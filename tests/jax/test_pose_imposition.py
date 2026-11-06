"""Fixed native imposition plans preserve dtype, gaps and coordinate transformations."""

import attrs
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tests.scientific.execution import Execution
from tests.scientific.test_pose_imposition import imposition_fixture
from tmol_jax.pack.imposition import prepare_pose_imposition


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("bump", (False, True))
def test_joint_design_expands_original_member_identity_before_atom_layout(
    platform, bump
):
    from tests.scientific.molecules import MolecularCase
    from tests.scientific.test_correlated_graph import joint_fixture
    from tmol_jax.pack.interaction_graph import build_interaction_graph
    from tmol_jax.pack.rotamer_set import ConformerContext
    from tmol_jax.pose.pose_stack_builder import PoseStackBuilder

    b = Execution("jax-" + platform)
    with jax.default_device(b.device):
        case = MolecularCase(
            (), (("ARG", "ALA", "LYS", "VAL", "SER", "GLY"),), ((1,) * 6,)
        )
        b.molecule(case, terms=("ref",))
        pbt = b.pose.packed_block_types
        source = np.asarray(b.pose.coords)[0].copy()
        offsets = np.asarray(b.pose.block_coord_offset)[0].copy()
        types = np.asarray(b.pose.block_type_ind)[0].copy()
        natoms = np.asarray(pbt.n_atoms)
        blocks = [
            source[start : start + natoms[kind]] for start, kind in zip(offsets, types)
        ]
        pose = PoseStackBuilder.from_block_types_and_coords(
            pbt,
            [[*types[[0, 2, 4]], -1], [-1] * 4],
            [np.concatenate(blocks[::2]), np.empty((0, 3), np.float32)],
            [[0, 1, 2, -1], [-1] * 4],
            [[1, 2, 3, -1], [-1] * 4],
            connections_per_pose=[(), ()],
        )
        r, indices, values = joint_fixture(b)
        r.coords = jnp.asarray(source)
        r.coord_offset_for_rot = jnp.asarray(offsets)
        r.block_type_ind_for_rot = jnp.asarray(types)
        r.context = ConformerContext(pose)
        if bump:
            # Prune state zero so the fixed joint survivor has a nonzero original
            # identity; both member types and atom counts must follow state one.
            values = (values * 0.01).at[0].add(100)
        graph = build_interaction_graph(
            r, values, indices, chunk_size=2, bump_check=bump
        )
        original = np.asarray(pose.coords).copy()
        for state in ((1,) if bump else range(2)):
            for external in range(2):
                choices = jnp.asarray(
                    [[[external]], [[0]]] if bump else [[[state, external]], [[0, 0]]],
                    jnp.int32,
                )
                plan = prepare_pose_imposition(pose, r, graph["retained"], choices)
                out = plan.apply(r.coords)
                selected = [state, 2 + external, 4 + state]
                np.testing.assert_array_equal(
                    out.block_type_ind, [[*types[selected], -1], [-1] * 4]
                )
                np.testing.assert_array_equal(
                    np.asarray(plan.selected).copy().view(np.int64).reshape(2, 4),
                    [[*selected, -1], [-1] * 4],
                )
                expected = np.concatenate([blocks[i] for i in selected])
                np.testing.assert_array_equal(
                    np.asarray(out.coords)[0, : len(expected)], expected
                )
                np.testing.assert_array_equal(out.coords[1], 0)
                np.testing.assert_array_equal(
                    jax.jit(plan.coordinates)(r.coords), out.coords
                )
                np.testing.assert_array_equal(pose.coords, original)
                assert out.packed_block_types is pbt
                assert out.coords.device == b.device
        invalid = dict(graph["retained"])
        del invalid["group_collapse"]
        with pytest.raises(ValueError, match="shared collapse plan"):
            prepare_pose_imposition(pose, r, invalid, choices)


@pytest.mark.parametrize("platform", ("cpu", "cuda"))
@pytest.mark.parametrize("dtype", (np.float32, np.float64))
def test_imposition_gaps_coordinate_ad_and_batching(platform, dtype):
    b = Execution("jax-" + platform)
    with jax.enable_x64(dtype == np.float64), jax.default_device(b.device):
        r, graph, choices, selected = imposition_fixture(b, "mixed_batch")
        original = b.pose
        offsets = np.asarray(original.block_coord_offset).copy()
        types = np.asarray(original.block_type_ind)
        offsets += 2 * (np.arange(offsets.shape[1], dtype=np.int32)[None, :] + 1)
        capacity = original.coords.shape[1] + 2 * (offsets.shape[1] + 1)
        old_coords = np.asarray(original.coords)
        gapped = np.zeros((len(types), capacity, 3), dtype)
        mask = np.zeros(gapped.shape[:2], bool)
        natoms = np.asarray(original.packed_block_types.n_atoms)
        for p, block in np.argwhere(types >= 0):
            n = natoms[types[p, block]]
            start = offsets[p, block]
            old = int(original.block_coord_offset[p, block])
            gapped[p, start : start + n] = old_coords[p, old : old + n]
            mask[p, start : start + n] = True
        b.pose = original.replace(
            coords=jnp.asarray(gapped), block_coord_offset=jnp.asarray(offsets)
        )
        np.testing.assert_array_equal(b.pose.atom_is_real, mask)
        r = attrs.evolve(r, coords=r.coords.astype(dtype))
        plan = prepare_pose_imposition(b.pose, r, graph["retained"], choices)
        out = plan.apply(r.coords)
        np.testing.assert_array_equal(out.block_coord_offset, offsets)
        np.testing.assert_array_equal(out.atom_is_real, mask)
        assert out.coords.dtype == dtype and out.coords.device == b.device
        np.testing.assert_array_equal(np.asarray(out.coords)[~mask], 0)
        selected_atoms = np.zeros(len(r.coords), bool)
        rot_offsets = np.asarray(r.coord_offset_for_rot)
        rot_types = np.asarray(r.block_type_ind_for_rot)
        for rot in selected[selected >= 0]:
            start = rot_offsets[rot]
            selected_atoms[start : start + natoms[rot_types[rot]]] = True

        def loss(x):
            return jnp.square(plan.coordinates(x)).sum()

        gradient = jax.jit(jax.grad(loss))(r.coords)
        np.testing.assert_array_equal(
            gradient, 2 * np.asarray(r.coords) * selected_atoms[:, None]
        )
        _, hvp = jax.jit(lambda x: jax.jvp(jax.grad(loss), (x,), (jnp.ones_like(x),)))(
            r.coords
        )
        np.testing.assert_array_equal(
            hvp, np.broadcast_to(2 * selected_atoms[:, None], r.coords.shape)
        )
        direction = jnp.ones_like(r.coords) * jnp.asarray(0.25, dtype)
        _, tangent = jax.jit(lambda x: jax.jvp(plan.coordinates, (x,), (direction,)))(
            r.coords
        )
        np.testing.assert_array_equal(
            tangent, np.broadcast_to(mask[..., None] * 0.25, out.coords.shape)
        )
        batched = jax.jit(jax.vmap(plan.coordinates))(
            jnp.stack((r.coords, r.coords + direction))
        )
        np.testing.assert_array_equal(batched[0], out.coords)
        np.testing.assert_allclose(
            batched[1] - batched[0],
            tangent,
            rtol=0,
            atol=2e-6 if dtype == np.float32 else 2e-15,
        )
        with pytest.raises(TypeError, match="eager"):
            jax.jit(
                lambda x: prepare_pose_imposition(
                    b.pose, r, graph["retained"], x
                ).coordinates(r.coords)
            )(choices)
        invalid = choices.at[0, 0, 0].set(-1)
        with pytest.raises(ValueError, match="retained conformer"):
            prepare_pose_imposition(b.pose, r, graph["retained"], invalid)
