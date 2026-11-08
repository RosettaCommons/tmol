"""Palette sampler defaults are owned by tasks and cover real molecular input."""

from importlib import import_module
from types import SimpleNamespace
import numpy as np
import pytest

from .molecules import DOMAIN_CASES


def packing_api(backend):
    root = import_module("tmol_jax" if backend.framework == "jax" else "tmol")
    samplers = import_module(root.__name__ + ".pack")
    return SimpleNamespace(
        PackerPalette=root.PackerPalette,
        PackerTask=root.PackerTask,
        pack_rotamers=root.pack_rotamers,
        IncludeCurrentSampler=samplers.IncludeCurrentSampler,
    )


@pytest.mark.parametrize("name", ("ligand", "d_peptide", "dna"))
def test_default_task_preserves_unsampled_molecular_coordinates(backend, name):
    from .test_molecular_sampling import sampling_task, block_mask

    b = backend.molecule(DOMAIN_CASES[name], terms=())
    api = packing_api(b)
    palette = api.PackerPalette()
    task = api.PackerTask(b.pose, palette)
    assert [s.sampler_name() for s in task.conformer_samplers] == ["FallbackSampler"]
    _, inactive, _ = sampling_task(b)
    task.add_conformer_sampler(inactive)
    if name == "d_peptide":
        task.disable_sampler_by_block_mask(
            inactive, block_mask(b, np.ones(b.pose.block_type_ind.shape, bool))
        )
    samplers = tuple(task.conformer_samplers)
    original = b.pose
    before = b.coords().copy()
    b.use_beta2016()
    energy, gradient = b.channel_values_and_gradients(before)
    packed = api.pack_rotamers(original, b.score_function, task, seed=412)
    assert tuple(task.conformer_samplers) == samplers
    np.testing.assert_array_equal(b.numpy(original.coords), before)
    np.testing.assert_array_equal(b.numpy(packed.coords), before)
    np.testing.assert_array_equal(
        b.numpy(packed.block_type_ind), b.numpy(original.block_type_ind)
    )
    np.testing.assert_array_equal(
        b.numpy(packed.inter_residue_connections),
        b.numpy(original.inter_residue_connections),
    )
    b.pose = packed
    b.use_beta2016()
    after_energy, after_gradient = b.channel_values_and_gradients(before)
    assert np.isfinite(after_energy).all() and np.isfinite(after_gradient).all()
    np.testing.assert_allclose(after_energy, energy, rtol=3e-5, atol=3e-5)
    np.testing.assert_allclose(after_gradient, gradient, rtol=3e-5, atol=3e-5)


def test_custom_palette_default_samplers_keep_instance_and_task_ownership(backend):
    b = backend.molecule(DOMAIN_CASES["ligand"], terms=())
    api = packing_api(b)
    same, distinct = api.IncludeCurrentSampler(), api.IncludeCurrentSampler()
    defaults = [same, same, distinct]

    class Palette(api.PackerPalette):
        def default_conformer_samplers(self):
            return defaults

    palette = Palette()
    a, other = (api.PackerTask(b.pose, palette) for _ in range(2))
    assert a.conformer_samplers == [same, distinct]
    assert other.conformer_samplers == [same, distinct]
    assert a.conformer_samplers is not other.conformer_samplers
    mask = np.ones(b.numpy(a.is_real_block).shape, bool)
    mask = (
        b.module.device_put(mask, b.device)
        if b.framework == "jax"
        else b.module.as_tensor(mask, device=b.device)
    )
    a.disable_sampler_by_block_mask(same, mask)
    np.testing.assert_array_equal(
        b.numpy(a.per_block_conformer_sampler_allowed)[..., 0], False
    )
    np.testing.assert_array_equal(
        b.numpy(a.per_block_conformer_sampler_allowed)[..., 1], True
    )
    np.testing.assert_array_equal(
        b.numpy(other.per_block_conformer_sampler_allowed), True
    )
    defaults.clear()
    assert a.conformer_samplers == [same, distinct]
    assert other.conformer_samplers == [same, distinct]
