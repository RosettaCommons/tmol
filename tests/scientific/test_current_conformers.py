"""Input-conformation coverage across real molecular identities and design."""

import numpy as np
import pytest

from .molecules import CASES, MolecularCase
from .test_molecular_sampling import block_mask, sampling_task


def current_samplers(backend):
    if backend.framework == "jax":
        from tmol_jax.pack import IncludeCurrentSampler, FallbackSampler
    else:
        from tmol.pack import IncludeCurrentSampler, FallbackSampler
    return IncludeCurrentSampler(), FallbackSampler()


def mixed_task(backend):
    backend.molecule(
        MolecularCase(
            CASES["ligand"].parameter_files,
            (("ARG", "LYS", "DARG", "LG1"),),
            ((1, 1, 1, 1),),
        ),
        terms=("ref",),
    )
    return sampling_task(backend)


def selected_positions(backend, task, result):
    counts, choices, metadata = result
    assert counts.device == choices.device == backend.device
    counts, choices = map(backend.numpy, (counts, choices))
    assert counts.dtype == choices.dtype == np.int32
    np.testing.assert_array_equal(choices, np.flatnonzero(counts))
    assert metadata == {"copy_input_coordinates": True}
    positions = backend.numpy(task.cons_bt_block)[choices]
    originals = backend.numpy(task.per_block_orig_block_type)
    types = backend.numpy(task.cons_bt_block_type)[choices]
    np.testing.assert_array_equal(types, originals[0, positions])
    return positions


def test_include_current_keeps_real_input_identities_and_respects_masks(backend):
    task, _, concrete = mixed_task(backend)
    current, _ = current_samplers(backend)
    task.add_conformer_sampler(current)
    before = backend.coords().copy()
    fixed = concrete(task)
    np.testing.assert_array_equal(
        selected_positions(
            backend, fixed, current.create_samples_for_poses(backend.pose, fixed)
        ),
        [0, 1, 2, 3],
    )
    task.disable_sampler_by_block_mask(
        current, block_mask(backend, [[False, True, False, False]])
    )
    task.disable_packing_by_block_mask(
        block_mask(backend, [[False, False, True, False]])
    )
    fixed = concrete(task)
    np.testing.assert_array_equal(
        selected_positions(
            backend, fixed, current.create_samples_for_poses(backend.pose, fixed)
        ),
        [0, 3],
    )
    np.testing.assert_array_equal(backend.coords(), before)


@pytest.mark.parametrize("fixed_protein", (False, True))
def test_fallback_fills_actual_sampler_gaps_and_fixed_positions(backend, fixed_protein):
    task, dunbrack, concrete = mixed_task(backend)
    _, fallback = current_samplers(backend)
    task.add_conformer_sampler(fallback)
    if fixed_protein:
        task.disable_packing_by_block_mask(
            block_mask(backend, [[True, False, False, False]])
        )
    fixed = concrete(task)
    np.testing.assert_array_equal(
        selected_positions(
            backend, fixed, fallback.create_samples_for_poses(backend.pose, fixed)
        ),
        [0, 1, 2, 3],
    )
    built = dunbrack.create_samples_for_poses(backend.pose, fixed)[0]
    selected = fallback.create_samples_for_poses(
        backend.pose, fixed, built_rotamer_counts=built
    )
    np.testing.assert_array_equal(
        selected_positions(backend, fixed, selected), [0, 3] if fixed_protein else [3]
    )
    # An uncovered, explicitly disabled fallback contributes nothing. Fixed
    # positions must still have an input state even when this sampler is off.
    task.disable_sampler_by_block_mask(
        fallback, block_mask(backend, [[True, True, True, True]])
    )
    fixed = concrete(task)
    selected = fallback.create_samples_for_poses(
        backend.pose, fixed, built_rotamer_counts=built
    )
    np.testing.assert_array_equal(
        selected_positions(backend, fixed, selected), [0] if fixed_protein else []
    )
    for sampler in task.conformer_samplers:
        task.disable_sampler_by_block_mask(
            sampler, block_mask(backend, [[True, True, True, True]])
        )
    fixed = concrete(task)
    np.testing.assert_array_equal(
        selected_positions(
            backend,
            fixed,
            fallback.create_samples_for_poses(
                backend.pose, fixed, built_rotamer_counts=built
            ),
        ),
        [0, 1, 2, 3],
    )


def test_fallback_does_not_reintroduce_a_designed_original_identity(backend):
    task, dunbrack, concrete = mixed_task(backend)
    _, fallback = current_samplers(backend)
    task.add_conformer_sampler(fallback)
    # Re-enable choices at the first ARG position, then design it to LYS.
    types = backend.numpy(task.per_block_considered_block_types)
    blocks = backend.pose.packed_block_types.active_block_types
    allowed = backend.numpy(task.per_block_is_block_type_allowed).copy()
    allowed[0, 0] = [
        index >= 0 and blocks[index].name == "LYS" for index in types[0, 0]
    ]
    assert allowed[0, 0].sum() == 1
    task.per_block_is_block_type_allowed = block_mask(backend, allowed)
    fixed = concrete(task)
    built = dunbrack.create_samples_for_poses(backend.pose, fixed)[0]
    rows = backend.numpy(fixed.cons_bt_block) == 0
    assert backend.numpy(built)[rows].sum() > 0
    selected = fallback.create_samples_for_poses(
        backend.pose, fixed, built_rotamer_counts=built
    )
    np.testing.assert_array_equal(selected_positions(backend, fixed, selected), [3])
