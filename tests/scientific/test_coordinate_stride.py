"""Unused coordinate capacity cannot change another pose's physics or forces."""

from contextlib import nullcontext

import attrs
import numpy as np

from .molecules import BACKBONE_CASES


def test_coordinate_capacity_and_replacement_preserve_all_term_values_and_forces(
    backend,
):
    b = backend.molecule(BACKBONE_CASES["mixed"], terms=())
    precision = b.module.enable_x64() if b.framework == "jax" else nullcontext()
    with precision:
        coordinates = b.coords().astype(np.float64)
        if b.framework == "jax":
            pose = b.pose.replace(coords=b.module.device_put(coordinates, b.device))
        else:
            pose = attrs.evolve(
                b.pose, coords=b.module.as_tensor(coordinates, device=b.device)
            )
        b.pose = pose
        b.use_beta2016()
        values, gradients = b.channel_values_and_gradients(coordinates)
        padded = np.pad(coordinates, ((0, 0), (0, 7), (0, 0)))
        if b.framework == "jax":
            b.pose = pose.replace(coords=b.module.device_put(padded, b.device))
        else:
            b.pose = attrs.evolve(
                pose, coords=b.module.as_tensor(padded, device=b.device)
            )
            b._render_torch()
        other_values, other_gradients = b.channel_values_and_gradients(padded)
        np.testing.assert_allclose(other_values, values, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(
            other_gradients[..., : coordinates.shape[1], :],
            gradients,
            rtol=2e-10,
            atol=2e-10,
        )
        np.testing.assert_array_equal(
            other_gradients[..., coordinates.shape[1] :, :], 0
        )
        b.pose = pose
        if b.framework == "torch":
            b._render_torch()
        again_values, again_gradients = b.channel_values_and_gradients(coordinates)
        np.testing.assert_allclose(again_values, values, rtol=2e-10, atol=2e-10)
        np.testing.assert_allclose(again_gradients, gradients, rtol=2e-10, atol=2e-10)
