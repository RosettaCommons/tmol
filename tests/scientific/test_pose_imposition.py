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
