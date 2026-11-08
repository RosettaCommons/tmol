"""Selecting molecular conformers preserves atoms, topology and parameter context."""

import numpy as np
import pytest

from .test_complete_preset import CASES
from .rotamer_execution import RotamerExecution
from .test_current_conformers import ensure_fallback_sampler

IMPOSITION_CASES = (
    "ligand",
    "connected_noncanonical",
    "d_peptide",
    "mixed_batch",
    "dna",
    "disulfide_mixed",
    "metal_mixed",
)


def imposition_fixture(backend, name, *, design=False):
    if design:
        from .molecules import MolecularCase

        types = (
            (("ARG", "ALA", "GLY"), ("DARG", "DALA", "GLY"))
            if name == "fixed_design"
            else (("ARG", "LYS"), ("DARG", "DLYS"))
        )
        case = MolecularCase((), types, tuple((len(row),) for row in types))
    else:
        case = CASES[name]
    backend.molecule(case, terms=("ref",))
    if not design:
        e = RotamerExecution(
            backend, family="na_torsion" if name == "dna" else "genbonded", sampled=True
        )
        rotamers = e.rotamers
    else:
        if backend.framework == "jax":
            from tmol_jax.pack import (
                PackerTask,
                PackerPalette,
                SetPackerTask,
                DunbrackChiSampler,
                FixedAAChiSampler,
                build_rotamers,
            )

            sampler = DunbrackChiSampler.from_database(backend.parameter_database)
        else:
            from tmol import PackerTask, PackerPalette
            from tmol.pack import (
                SetPackerTask,
                create_dunbrack_sampler_from_database,
                build_rotamers,
                FixedAAChiSampler,
            )

            sampler = create_dunbrack_sampler_from_database(
                backend.parameter_database, backend.device
            )
        task = PackerTask(backend.pose, PackerPalette())
        task.add_conformer_sampler(sampler)
        if name == "fixed_design":
            task.add_conformer_sampler(FixedAAChiSampler())
        ensure_fallback_sampler(backend, task)
        _, rotamers = build_rotamers(
            backend.pose,
            SetPackerTask.from_packer_task(task),
            backend.parameter_database.chemical,
        )
    b = backend
    if b.framework == "jax":
        from tmol_jax.pack.graph import retained_topology

        retained = retained_topology(
            *(
                getattr(rotamers, k)
                for k in (
                    "n_rots_for_block",
                    "rot_offset_for_block",
                    "rot_offset_for_pose",
                    "pose_for_rot",
                    "block_ind_for_rot",
                )
            )
        )
        graph = {"retained": retained}
    else:
        from tmol.pack.compiled._compiled import _build_interaction_graph

        graph = _build_interaction_graph(
            False,
            16,
            b.pose.packed_block_types.n_types,
            *(
                getattr(rotamers, k)
                for k in (
                    "n_rots_for_pose",
                    "rot_offset_for_pose",
                    "n_rots_for_block",
                    "rot_offset_for_block",
                    "pose_for_rot",
                    "block_type_ind_for_rot",
                    "block_ind_for_rot",
                )
            ),
            b.module.zeros((3, 0), dtype=b.module.int32, device=b.device),
            b.module.zeros(0, device=b.device),
            -1,
            False,
        )
    counts = b.numpy(rotamers.n_rots_for_block)
    offsets = b.numpy(rotamers.rot_offset_for_block)
    types = b.numpy(rotamers.block_type_ind_for_rot)
    original = b.numpy(b.pose.block_type_ind)
    choices = np.zeros(
        (len(counts), 1, int((counts > 1).sum(axis=1).max(initial=0))), np.int32
    )
    selected = np.full(counts.shape, -1, np.int32)
    for p, row in enumerate(counts):
        molten = 0
        for block, count in enumerate(row):
            if not count:
                continue
            start = offsets[p, block]
            choice = (p + block + 1) % count
            if design and block == 0:
                alternatives = np.flatnonzero(
                    types[start : start + count] != original[p, block]
                )
                assert alternatives.size
                choice = alternatives[0]
            selected[p, block] = start + choice
            if count > 1:
                choices[p, 0, molten] = choice
                molten += 1
    if b.framework == "jax":
        choices = b.module.device_put(choices, b.device)
    else:
        choices = b.module.as_tensor(choices, device=b.device)
    return rotamers, graph, choices, selected


def impose(backend, rotamers, graph, choices):
    if backend.framework == "jax":
        from tmol_jax.pack.imposition import impose_top_rotamer_assignments

        return impose_top_rotamer_assignments(backend.pose, rotamers, graph, choices)
    from tmol.pack._impose_rotamers import _impose_top_rotamer_assignments

    return _impose_top_rotamer_assignments(
        backend.pose,
        rotamers,
        graph[7],
        graph[1].long(),
        graph[5].long(),
        graph[8],
        choices,
    )


@pytest.mark.parametrize(
    "name,design",
    [(name, False) for name in IMPOSITION_CASES]
    + [("design", True), ("fixed_design", True)],
)
def test_molecular_imposition_keeps_selected_atoms_and_connectivity(
    backend, name, design
):
    r, g, choices, selected = imposition_fixture(backend, name, design=design)
    original = backend.coords().copy()
    info = backend.pose.pdb_info
    info.atom_occupancy[:] = (
        np.arange(info.atom_occupancy.size).reshape(info.atom_occupancy.shape) / 100
    )
    info.atom_b_factor[:] = 0.5 + np.arange(info.atom_b_factor.size).reshape(
        info.atom_b_factor.shape
    )
    out = impose(backend, r, g, choices)
    xyz, offsets, types = map(
        backend.numpy, (out.coords, out.block_coord_offset, out.block_type_ind)
    )
    source, source_offsets, source_types = map(
        backend.numpy, (r.coords, r.coord_offset_for_rot, r.block_type_ind_for_rot)
    )
    n_atoms = backend.numpy(out.packed_block_types.n_atoms)
    expected = np.zeros_like(xyz)
    for p, block in np.argwhere(selected >= 0):
        rot = selected[p, block]
        assert types[p, block] == source_types[rot]
        old_kind = int(backend.numpy(backend.pose.block_type_ind)[p, block])
        dest = slice(offsets[p, block], offsets[p, block] + n_atoms[types[p, block]])
        old_offset = int(backend.numpy(backend.pose.block_coord_offset)[p, block])
        source_metadata = slice(old_offset, old_offset + n_atoms[old_kind])
        for field, default in (("atom_occupancy", 1.0), ("atom_b_factor", 30.0)):
            expected_metadata = (
                getattr(info, field)[p, source_metadata]
                if types[p, block] == old_kind
                else default
            )
            np.testing.assert_array_equal(
                getattr(out.pdb_info, field)[p, dest], expected_metadata
            )
        n = n_atoms[types[p, block]]
        expected[p, offsets[p, block] : offsets[p, block] + n] = source[
            source_offsets[rot] : source_offsets[rot] + n
        ]
    np.testing.assert_array_equal(xyz, expected)
    np.testing.assert_array_equal(backend.coords(), original)
    np.testing.assert_array_equal(
        backend.numpy(out.inter_residue_connections),
        backend.numpy(backend.pose.inter_residue_connections),
    )
    assert out.packed_block_types is backend.pose.packed_block_types
    assert out.coords.device == backend.pose.coords.device
    if design:
        assert np.any(types != backend.numpy(backend.pose.block_type_ind))
        real_counts = np.where(types >= 0, n_atoms[np.maximum(types, 0)], 0)
        expected_offsets = np.cumsum(real_counts, axis=1) - real_counts
        np.testing.assert_array_equal(offsets, expected_offsets)
        assert xyz.shape[1] == real_counts.sum(axis=1).max()
    else:
        np.testing.assert_array_equal(
            offsets, backend.numpy(backend.pose.block_coord_offset)
        )
        assert xyz.shape == original.shape

    if backend.framework == "jax":
        if name == "mixed_batch" and not design:
            for dtype in (np.float32, np.float64):
                _check_imposition_transforms(backend, dtype)
        elif name == "design" and design:
            for bump in (False, True):
                _check_joint_design_expansion(backend, bump)


def _check_joint_design_expansion(backend, bump):
    import attrs

    jax, jnp = backend.module, backend.module.numpy
    from tmol_jax.pack.imposition import prepare_pose_imposition

    from tests.scientific.molecules import MolecularCase
    from tests.scientific.test_correlated_graph import joint_fixture
    from tmol_jax.pack.interaction_graph import build_interaction_graph
    from tmol_jax.pack.rotamer_set import ConformerContext
    from tmol_jax.pose.pose_stack_builder import PoseStackBuilder

    b = backend
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


def _check_imposition_transforms(backend, dtype):
    import attrs

    jax, jnp = backend.module, backend.module.numpy
    from tmol_jax.pack.imposition import prepare_pose_imposition

    b = backend
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
