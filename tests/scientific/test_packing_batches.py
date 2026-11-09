"""Bounded public packing preserves physical choices and molecular metadata."""

from importlib import import_module
from dataclasses import fields

import numpy as np
import pytest

from .test_packing_execution import check_complete_packing
from .constraint_execution import Constraints
from .fragment_molecules import fragment_fixture
from .test_fragment_input import _build, _physical_atoms


@pytest.mark.parametrize("bump", [False, True])
def test_packing_bounds_mixed_pose_batches(backend, monkeypatch, bump):
    monkeypatch.setenv("TMOL_PACK_MAX_POSES_PER_CHUNK", "1")
    seen = []
    preset = backend.use_beta2016

    def prepare():
        result = preset()
        render = backend.score_function.render_rotamer_scoring_module

        def observe(pose, *args, **kwargs):
            seen.append(pose.n_poses)
            return render(pose, *args, **kwargs)

        monkeypatch.setattr(
            backend.score_function, "render_rotamer_scoring_module", observe
        )
        return result

    monkeypatch.setattr(backend, "use_beta2016", prepare)
    check_complete_packing(backend, "mixed_batch", bump=bump)
    # The existing independent bump invariant renders the full original table
    # after packing; only the first two calls belong to the bounded packer.
    assert seen[:2] == [1, 1]
    assert len(seen) == (3 if bump else 2)

    if backend.framework == "jax" and not bump:
        from tmol_core.pack.streaming import complete_table_fits_window
        from tmol_jax.pack import streaming
        from .rotamer_execution import RotamerExecution
        from .test_complete_preset import CASES
        from .test_canonical_molecular_input import canonical_input, rebuild_canonical

        for name in ("ligand", "mixed_batch", "d_peptide"):
            backend.molecule(CASES[name], terms=())
            # Public canonical input retains the complete chemistry palette, including
            # parameter tables for residues absent from this particular pose.
            co, canonical = canonical_input(CASES[name], backend.parameter_database)
            rebuild_canonical(backend, co, canonical)
            rotamers = RotamerExecution(
                backend, family="genbonded", sampled=True
            ).rotamers
            backend.use_beta2016()
            scorer = backend.score_function.render_rotamer_scoring_module(
                backend.pose, rotamers
            )
            bound = streaming.complete_table_bound(scorer, rotamers.coords)
            assert bound is not None
            values, indices = scorer(rotamers.coords)
            assert values.size <= bound[1]
            # Independently measure the production call with device tables as operands.
            # An extra enclosing jit embeds those tables as constants instead, changing
            # the storage contract. Native scratch is an explicit FFI result.
            evaluate = scorer.prepare(rotamers.coords)
            executable = evaluate.func.lower(
                rotamers.coords, **evaluate.keywords
            ).compile()
            memory = executable.memory_analysis()
            assert memory.output_size_in_bytes + memory.temp_size_in_bytes <= bound[2]
            assert values.nbytes + indices.nbytes <= bound[2]
            assert streaming.use_complete_table(scorer, rotamers.coords)
            assert not streaming.use_complete_table(
                scorer, rotamers.coords, score_window=1
            )
            assert not complete_table_fits_window(
                None, bound[1], bound[2], values.dtype.itemsize
            )


def test_packing_chunks_preserve_fragment_constraints_and_fixed_atoms(
    backend, monkeypatch
):
    monkeypatch.setenv("TMOL_PACK_MAX_POSES_PER_CHUNK", "1")
    _, atoms, parameters = fragment_fixture(repeated=True, mixed=False)
    pose, context = _build(backend, atoms, parameters)
    front = import_module("tmol_jax" if backend.framework == "jax" else "tmol")
    backend.pose = front.PoseStackBuilder.from_poses([pose, pose], backend.device)
    backend.parameter_database = context.parameter_database
    backend.use_beta2016()
    constraints = np.array([[[p, 0, 0], [p, 0, 1]] for p in range(2)], np.int32)
    Constraints(backend).attach(
        [("harmonic", constraints, np.array([[1.3, 0.4], [1.5, 0.6]], np.float32))]
    )
    backend.score_function.set_weight(front.ScoreType.constraint, 0.7)
    task = front.PackerTask(backend.pose, front.PackerPalette())
    task.disable_packing_by_block_mask(
        Constraints(backend).array(np.ones(task.is_real_block.shape, bool))
    )
    original = backend.pose
    original_atoms = _physical_atoms(backend, original)
    original_constraints = backend.numpy(
        original.constraint_set.constraint_atoms
    ).copy()
    before = backend.coords().copy()
    result = front.pack_rotamers(original, backend.score_function, task, seed=17)
    np.testing.assert_array_equal(backend.numpy(result.coords), before)
    np.testing.assert_array_equal(
        backend.numpy(result.constraint_set.constraint_atoms), original_constraints
    )
    np.testing.assert_array_equal(
        result.pdb_info.residue_labels, original.pdb_info.residue_labels
    )
    assert _physical_atoms(backend, result) == original_atoms
    assert len(result.split_block_mapping.entries) == len(
        original.split_block_mapping.entries
    )
    for actual, expected in zip(
        result.split_block_mapping.entries, original.split_block_mapping.entries
    ):
        for field in fields(expected):
            np.testing.assert_array_equal(
                getattr(actual, field.name), getattr(expected, field.name)
            )
    backend.pose = result
    if backend.framework == "torch":
        backend._render_torch()
    values, gradients = backend.channel_values_and_gradients(backend.coords())
    assert (
        len(values) == 26 and np.isfinite(values).all() and np.isfinite(gradients).all()
    )
    assert np.linalg.norm(gradients) > 1e-3
