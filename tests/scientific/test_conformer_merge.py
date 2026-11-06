"""Merging actual molecular samplers retains original sample identity."""

import numpy as np

from .test_current_conformers import mixed_task, current_samplers


def test_mixed_molecular_samples_preserve_source_rows_and_fallback(backend):
    task, dunbrack, concrete = mixed_task(backend)
    current, fallback = current_samplers(backend)
    task.add_conformer_sampler(current)
    task.add_conformer_sampler(fallback)
    fixed = concrete(task)
    samples = [
        sampler.create_samples_for_poses(backend.pose, fixed)
        for sampler in (dunbrack, current)
    ]
    samples.append(
        fallback.create_samples_for_poses(
            backend.pose, fixed, built_rotamer_counts=samples[0][0] + samples[1][0]
        )
    )
    if backend.framework == "jax":
        from tmol_jax.pack.merge import merge_conformer_samples as merge
    else:
        from tmol.pack.rotamer._build_rotamers import _merge_conformer_samples as merge
    merged_counts, sampler_ids, choices, masks, destinations = merge(samples)
    counts = [backend.numpy(sample[0]) for sample in samples]
    records = [
        (choice, sampler, row)
        for choice in range(len(counts[0]))
        for sampler in range(len(samples))
        for row in range(
            sum(counts[sampler][:choice]), sum(counts[sampler][: choice + 1])
        )
    ]
    np.testing.assert_array_equal(backend.numpy(merged_counts), np.sum(counts, axis=0))
    np.testing.assert_array_equal(
        backend.numpy(choices), [record[0] for record in records]
    )
    np.testing.assert_array_equal(
        backend.numpy(sampler_ids), [record[1] for record in records]
    )
    for i, (mask, destination) in enumerate(zip(masks, destinations)):
        np.testing.assert_array_equal(
            backend.numpy(destination),
            [row for row, record in enumerate(records) if record[1] == i],
        )
        np.testing.assert_array_equal(
            backend.numpy(mask), [record[1] == i for record in records]
        )
        assert destination.device == mask.device == backend.device
    # IncludeCurrent covers the libraryless ligand, so the fallback adds no
    # duplicate conformation. Library chi and the input remain distinct rows.
    assert not backend.numpy(samples[-1][0]).any()
    assert backend.numpy(merged_counts).sum() == sum(
        backend.numpy(sample[0]).sum() for sample in samples[:2]
    )
